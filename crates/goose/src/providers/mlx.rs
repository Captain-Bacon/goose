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
use serde_json::Value;
use std::time::Duration;
use tokio::pin;
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

#[derive(serde::Serialize)]
pub struct MlxProvider {
    #[serde(skip)]
    api_client: ApiClient,
    model: ModelConfig,
    name: String,
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

        let base = if host.starts_with("http://") || host.starts_with("https://") {
            host.clone()
        } else {
            format!("http://{}:{}", host, port)
        };

        let base_url =
            Url::parse(&base).map_err(|e| anyhow::anyhow!("Invalid MLX base URL: {e}"))?;

        let api_client =
            ApiClient::with_timeout(base_url.to_string(), AuthMethod::NoAuth, timeout)?;

        Ok(Self {
            api_client,
            model,
            name: MLX_PROVIDER_NAME.to_string(),
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
                ConfigKey::new("MLX_HOST", false, false, Some(MLX_HOST), false),
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
        let response = self
            .api_client
            .request(None, "v1/models")
            .response_get()
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to fetch models: {}", e)))?;

        if !response.status().is_success() {
            return Err(ProviderError::RequestFailed(format!(
                "Failed to fetch models: HTTP {}",
                response.status()
            )));
        }

        let json_response = response.json::<Value>().await.map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to parse response: {}", e))
        })?;

        let models = json_response
            .get("data")
            .and_then(|m| m.as_array())
            .ok_or_else(|| {
                ProviderError::RequestFailed("No data array in response".to_string())
            })?;

        let mut model_names: Vec<String> = models
            .iter()
            .filter_map(|model| model.get("id").and_then(|n| n.as_str()).map(String::from))
            .collect();

        model_names.sort();
        Ok(model_names)
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
