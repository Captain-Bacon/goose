use super::api_client::{ApiClient, AuthMethod};
use super::base::{ConfigKey, MessageStream, Provider, ProviderDef, ProviderMetadata};
use super::errors::ProviderError;
use super::openai_compatible::handle_status_openai_compat;
use super::retry::ProviderRetry;
use super::utils::{ImageFormat, RequestLog};
use crate::config::GooseMode;
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use crate::providers::formats::openai::{create_request, response_to_streaming_message};
use anyhow::{Error, Result};
use async_stream::try_stream;
use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::TryStreamExt;
use reqwest::Response;
use rmcp::model::Tool;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::pin;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;
use url::Url;

const MLX_PROVIDER_NAME: &str = "mlx";
pub const MLX_HOST: &str = "localhost";
pub const MLX_TIMEOUT: u64 = 600;
pub const MLX_DEFAULT_PORT: u16 = 5757;
pub const MLX_DEFAULT_MODEL: &str = "mlx-community/Qwen3.5-35B-A3B-4bit";
pub const MLX_KNOWN_MODELS: &[&str] = &[
    MLX_DEFAULT_MODEL,
    "mlx-community/Qwen3.5-27B-4bit",
    "mlx-community/Qwen3.5-35B-A3B-8bit",
];
pub const MLX_DOC_URL: &str = "https://github.com/vllm-project/vllm-mlx";
pub const MLX_DEFAULT_IDLE_TIMEOUT: u64 = 15;

fn model_to_profile(model_name: &str) -> &str {
    match model_name {
        "mlx-community/Qwen3.5-27B-4bit" => "27b",
        "mlx-community/Qwen3.5-35B-A3B-8bit" => "35b-8bit",
        _ => "35b",
    }
}

struct MlxLifecycle {
    last_request: Instant,
    mlx_script: PathBuf,
    active_profile: Option<String>,
    port: u16,
    idle_timeout: Duration,
}

impl MlxLifecycle {
    async fn ensure_server_running(&mut self, model_name: &str) -> Result<(), ProviderError> {
        self.last_request = Instant::now();
        let needed_profile = model_to_profile(model_name).to_string();

        if self.is_server_up().await {
            // Server is running — check if it's the right profile
            if self.active_profile.as_deref() == Some(&needed_profile) {
                return Ok(());
            }
            // Wrong model loaded, restart with the right one
            tracing::info!(
                "MLX server running with wrong profile (have {:?}, need '{}'), restarting",
                self.active_profile, needed_profile
            );
        }

        tracing::info!("Starting MLX server with profile '{}'", needed_profile);
        self.start_server(&needed_profile).await?;
        self.active_profile = Some(needed_profile);
        Ok(())
    }

    async fn is_server_up(&self) -> bool {
        let url = format!("http://localhost:{}/v1/models", self.port);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build();
        match client {
            Ok(c) => c.get(&url).send().await.is_ok(),
            Err(_) => false,
        }
    }

    async fn start_server(&self, profile: &str) -> Result<(), ProviderError> {
        let output = tokio::process::Command::new(&self.mlx_script)
            .arg(profile)
            .output()
            .await
            .map_err(|e| {
                ProviderError::RequestFailed(format!(
                    "Failed to run mlx script at {}: {}",
                    self.mlx_script.display(),
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ProviderError::RequestFailed(format!(
                "mlx script failed (exit {}): {}",
                output.status,
                stderr.trim()
            )));
        }

        tracing::info!("MLX server started with profile '{}'", profile);
        Ok(())
    }

    fn is_idle(&self) -> bool {
        self.last_request.elapsed() >= self.idle_timeout
    }
}

#[derive(serde::Serialize)]
pub struct MlxProvider {
    #[serde(skip)]
    api_client: ApiClient,
    model: ModelConfig,
    name: String,
    #[serde(skip)]
    lifecycle: Arc<Mutex<MlxLifecycle>>,
}

impl MlxProvider {
    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let host: String = config
            .get_param("MLX_HOST")
            .unwrap_or_else(|_| MLX_HOST.to_string());

        let port: u16 = config
            .get_param("MLX_PORT")
            .unwrap_or(MLX_DEFAULT_PORT);

        let timeout: Duration =
            Duration::from_secs(config.get_param("MLX_TIMEOUT").unwrap_or(MLX_TIMEOUT));

        let idle_minutes: u64 = config
            .get_param("MLX_IDLE_TIMEOUT")
            .unwrap_or(MLX_DEFAULT_IDLE_TIMEOUT);

        let mlx_script: PathBuf = config
            .get_param::<String>("MLX_SCRIPT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                // crates/goose -> repo root
                p.pop();
                p.pop();
                p.push("mlx");
                p
            });

        let base = if host.starts_with("http://") || host.starts_with("https://") {
            host.clone()
        } else {
            format!("http://{}:{}", host, port)
        };

        let base_url =
            Url::parse(&base).map_err(|e| anyhow::anyhow!("Invalid MLX base URL: {e}"))?;

        let api_client =
            ApiClient::with_timeout(base_url.to_string(), AuthMethod::NoAuth, timeout)?;

        let lifecycle = Arc::new(Mutex::new(MlxLifecycle {
            last_request: Instant::now(),
            mlx_script,
            active_profile: None,
            port,
            idle_timeout: Duration::from_secs(idle_minutes * 60),
        }));

        // Spawn idle watcher
        let lc = Arc::clone(&lifecycle);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                // Check idle under lock (cheap), then release before network calls
                let (idle, script, port) = {
                    let lc = lc.lock().await;
                    (lc.is_idle(), lc.mlx_script.clone(), lc.port)
                };
                if idle {
                    let url = format!("http://localhost:{}/v1/models", port);
                    let up = reqwest::Client::builder()
                        .timeout(Duration::from_secs(2))
                        .build()
                        .map(|c| c.get(&url).send())
                        .ok();
                    let is_up = match up {
                        Some(fut) => fut.await.is_ok(),
                        None => false,
                    };
                    if is_up {
                        tracing::info!("Stopping MLX server (idle timeout)");
                        let _ = tokio::process::Command::new(&script)
                            .arg("stop")
                            .output()
                            .await;
                        // Clear active profile so next request triggers a fresh start
                        let mut lc = lc.lock().await;
                        lc.active_profile = None;
                    }
                }
            }
        });

        Ok(Self {
            api_client,
            model,
            name: MLX_PROVIDER_NAME.to_string(),
            lifecycle,
        })
    }
}

impl ProviderDef for MlxProvider {
    type Provider = Self;

    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            MLX_PROVIDER_NAME,
            "MLX",
            "Local inference on Apple Silicon via vllm-mlx",
            MLX_DEFAULT_MODEL,
            MLX_KNOWN_MODELS.to_vec(),
            MLX_DOC_URL,
            vec![
                ConfigKey::new("MLX_HOST", true, false, Some(MLX_HOST), false),
                ConfigKey::new(
                    "MLX_PORT",
                    false,
                    false,
                    Some("5757"),
                    false,
                ),
                ConfigKey::new(
                    "MLX_TIMEOUT",
                    false,
                    false,
                    Some(&MLX_TIMEOUT.to_string()),
                    false,
                ),
                ConfigKey::new(
                    "MLX_IDLE_TIMEOUT",
                    false,
                    false,
                    Some("15"),
                    false,
                ),
            ],
        )
    }

    fn from_env(
        model: ModelConfig,
        _extensions: Vec<crate::config::ExtensionConfig>,
    ) -> BoxFuture<'static, Result<Self::Provider>> {
        Box::pin(Self::from_env(model))
    }
}

#[async_trait]
impl Provider for MlxProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        session_id: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        // Ensure server is running with the right model before making the request
        {
            let mut lc = self.lifecycle.lock().await;
            lc.ensure_server_running(&model_config.model_name).await?;
        }

        let config = crate::config::Config::global();
        let goose_mode = config.get_goose_mode().unwrap_or(GooseMode::Auto);
        let filtered_tools = if goose_mode == GooseMode::Chat {
            &[]
        } else {
            tools
        };

        let payload = create_request(
            model_config,
            system,
            messages,
            filtered_tools,
            &ImageFormat::OpenAi,
            true,
        )?;
        let mut log = RequestLog::start(model_config, &payload)?;

        let response = self
            .with_retry(|| async {
                let resp = self
                    .api_client
                    .response_post(Some(session_id), "v1/chat/completions", &payload)
                    .await?;
                handle_status_openai_compat(resp).await
            })
            .await
            .inspect_err(|e| {
                let _ = log.error(e);
            })?;
        stream_mlx(response, log)
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        // Return known models — no need to start the server just to list models
        Ok(MLX_KNOWN_MODELS.iter().map(|s| s.to_string()).collect())
    }
}

fn stream_mlx(response: Response, mut log: RequestLog) -> Result<MessageStream, ProviderError> {
    let stream = response.bytes_stream().map_err(std::io::Error::other);

    Ok(Box::pin(try_stream! {
        let stream_reader = StreamReader::new(stream);
        let framed = FramedRead::new(stream_reader, LinesCodec::new())
            .map_err(Error::from);

        let message_stream = response_to_streaming_message(framed);
        pin!(message_stream);
        while let Some(message) = message_stream.next().await {
            let (message, usage) = message.map_err(|e|
                ProviderError::RequestFailed(format!("Stream decode error: {}", e))
            )?;
            log.write(&message, usage.as_ref().map(|f| f.usage).as_ref())?;
            yield (message, usage);
        }
    }))
}
