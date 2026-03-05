# Upstream Patches & Workarounds

Bugs we've hit that are fixed locally but not yet in stable releases. Check periodically — once a fix lands in a release, the workaround can be removed.

---

## 1. mlx-lm: GatedDeltaNet dtype mismatch (segfault)

**Symptom:** vllm-mlx segfaults during generation with Qwen 3.5 models.

**Root cause:** `A_log` parameter stored as F32, but Metal kernel assumes all inputs are F16. Wrong pointer arithmetic → out-of-bounds read → segfault.

**Fix:** PR #902 in ml-explore/mlx-lm (merged, not yet in a PyPI release as of 2026-03-05).

**Our workaround:** Installed mlx-lm 0.30.8 from git main instead of PyPI.

**How to check if resolved:**
```bash
# Check installed version
./vllm-mlx-env/bin/pip show mlx-lm

# Check latest PyPI release
./vllm-mlx-env/bin/pip index versions mlx-lm

# If PyPI version > 0.30.7, you can switch back:
./vllm-mlx-env/bin/pip install --upgrade mlx-lm
```

**Tracking:** https://github.com/ml-explore/mlx-lm/pull/902

---

## 2. vllm-mlx: Qwen 3.5 model loading (vision tower weights)

**Symptom:** vllm-mlx refuses to load any Qwen 3.5 model — all are natively multimodal with vision tower weights baked in.

**Root cause:** Weight loading uses `strict=True` by default, fails on unexpected vision tower keys.

**Fix:** PR #127 in otarkhan/vllm-mlx adds `strict=False` fallback.

**Our workaround:** Installed vllm-mlx from the PR branch (`feature/qwen3.5-support`).

**How to check if resolved:**
```bash
# Check if merged into main
./vllm-mlx-env/bin/pip show vllm-mlx  # note installed version

# Reinstall from main and test:
# ./vllm-mlx-env/bin/pip install git+https://github.com/otarkhan/vllm-mlx.git@main
```

**Tracking:** https://github.com/otarkhan/vllm-mlx/pull/127

---

## 3. vllm-mlx: Duplicate reasoning_content in streaming JSON

**Symptom:** Both `reasoning` and `reasoning_content` fields appear in streamed chat completion chunks, confusing downstream consumers.

**Root cause:** Pydantic models in `vllm_mlx/api/models.py` have both a `reasoning` field and a `@computed_field reasoning_content` property. Both serialise.

**Fix:** Added `exclude=True` to the `reasoning` Field in `AssistantMessage` and `ChatCompletionChunkDelta`. **This is a manual patch to site-packages** — not from any PR.

**Our workaround:** Hand-edited the installed package. Will be lost if vllm-mlx is reinstalled.

**How to check if resolved:**
```bash
# Check if there's a newer vllm-mlx release or if a PR addresses this
# Search: https://github.com/otarkhan/vllm-mlx/issues?q=reasoning_content

# After any reinstall of vllm-mlx, verify the fix is still present:
grep -n "exclude=True" ./vllm-mlx-env/lib/python3.12/site-packages/vllm_mlx/api/models.py
# Should show the reasoning field with exclude=True
```

**Tracking:** No upstream PR filed yet. If this breaks after a reinstall, re-apply manually — see session handover bead `general-1js` for details.

---

## 4. Qwen 3.5 quantised models: thinking behaviour depends on context

**Symptom:** In plain chat (no tools), model outputs `Thinking Process:\n1. Analyze...` as plain text — no `<think>` tags. In tool-calling context, model correctly uses `<think>...</think>` tags.

**Root cause:** The mlx-community 4-bit quantised Qwen 3.5 models don't follow `<think>` tag format for plain chat, but do for tool calls. This is a model behaviour quirk, not a serving bug.

**Current state:** `enable_thinking=True` (the vllm-mlx default) is restored. With tools present, thinking mode works — reasoning is surfaced alongside tool calls. For plain chat, the model's verbose plain-text reasoning will consume tokens. This is acceptable for Goose's agentic use case.

**If plain chat becomes a problem:** Patch `vllm_mlx/engine/simple.py` line 385 to force `enable_thinking = False`. This will suppress all reasoning (including useful tool-call reasoning). Trade-off: cleaner plain chat responses, but no visible reasoning for tool calls.

---

## 5. vllm-mlx: Streaming tool call parser misses closing tag split across chunks

**Symptom:** Tool calls detected in non-streaming mode but silently swallowed in streaming. All SSE chunks have `content: null, tool_calls: null`.

**Root cause:** `qwen_tool_parser.py` line 138 checked `"</tool_call>" in delta_text` — but the tokenizer splits `</tool_call>` across multiple chunks (e.g. `"</"` then `"tool_call>"`). Neither chunk alone contains the full closing tag.

**Our workaround:** Changed to check `current_text` (the accumulated text) instead of `delta_text`. Added guard `not closing_in_prev` to emit only once.

**Fragile:** Manual patch to site-packages. Lost on vllm-mlx reinstall.

**File:** `vllm_mlx/tool_parsers/qwen_tool_parser.py`

---

## 6. vllm-mlx: Qwen tool parser doesn't handle XML-param format

**Symptom:** Tool calls parsed in non-streaming mode (which has a separate code path) but fail in streaming. The Qwen 3.5 chat template produces `<tool_call><function=name><parameter=arg>value</parameter></function></tool_call>` format, but the parser only handles `<tool_call>{"name":"...","arguments":{...}}</tool_call>` JSON format.

**Root cause:** Missing regex pattern for XML-param format in `QwenToolParser.extract_tool_calls()`.

**Our workaround:** Added `XML_PARAM_PATTERN` and `PARAM_PATTERN` regexes to parse `<function=name>` and `<parameter=name>value</parameter>` tags. Parameter values are JSON-parsed to handle arrays/objects (Goose's `tool_graph` parameter).

**Fragile:** Manual patch to site-packages. Lost on vllm-mlx reinstall.

**File:** `vllm_mlx/tool_parsers/qwen_tool_parser.py`

---

## 7. vllm-mlx: Reasoning parser defaults to reasoning when no think tags present

**Symptom:** When `enable_thinking=False` and model produces tool calls without `<think>` tags, the streaming reasoning parser classifies all output (including tool call XML) as `reasoning_content`. Tool parser never sees the content.

**Root cause:** `think_parser.py` line 140 — Case 3 returns `DeltaMessage(reasoning=delta_text)` when no tags seen. Should return `DeltaMessage(content=delta_text)`.

**Our workaround:** Changed Case 3 in `think_parser.py` to default to content instead of reasoning.

**Note:** This patch IS active — `start-mlx-server.sh` now uses `--reasoning-parser qwen3`. Critical for nothink profiles where thinking is off but the reasoning parser is also off.

**Fragile:** Manual patch to site-packages. Lost on vllm-mlx reinstall.

**File:** `vllm_mlx/reasoning/think_parser.py`

---

## 8. vllm-mlx: enable_thinking toggle via environment variable

**Symptom:** `enable_thinking` is hardcoded in `simple.py` — always True unless "coder" is in the model name. No way to toggle thinking per server instance.

**Root cause:** Line 385 of `engine/simple.py` derives `enable_thinking` solely from the model name string.

**Our workaround:** Added `VLLM_MLX_ENABLE_THINKING` env var check. When set to `"0"`, forces `enable_thinking=False` (injects pre-closed empty `<think></think>` block via chat template, steering model to skip reasoning). When `"1"` or unset, original behaviour preserved.

**Also added:** `import os` at top of file (was not previously imported).

**Used by:** The `mlx` command script (`~/GitHub/goose/mlx`) sets this env var based on profile selection (`-nothink` suffix).

**Fragile:** Manual patch to site-packages. Lost on vllm-mlx reinstall.

**File:** `vllm_mlx/engine/simple.py` (lines 10, 383-390)

---

## Maintenance

When checking these, work through them top to bottom. If a PyPI release includes the fix, switch to the release version and remove the entry from this file.

After any `pip install` or venv rebuild, re-check patches #2, #3, #4, #5, #6, #7, and #8 — they're installed from branches/manual edits and will be overwritten.
