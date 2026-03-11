# ComfyUI-Z-Image-Utilities

A collection of utility nodes for ComfyUI designed for the [Z-Image](https://github.com/Tongyi-MAI/Z-Image) workflow, with LLM-powered prompt enhancement and an integrated sampling node.

![ComfyUI-Z-Image-Utilities](https://i.imgur.com/4pdoapv.png)

## Features

- Three provider modes: OpenRouter, local OpenAI-compatible API servers, and direct Hugging Face model loading
- Optional image-aware prompt enhancement for vision-capable models
- Standalone prompt enhancer or all-in-one integrated KSampler workflow
- Optional session history with manual `session_id` control; blank `session_id` runs are stateless
- Direct-model quantization support with cache unloading tools
- Detailed debugging through the `debug_log` output, console logs, and `z_image_debug.log`

---

## Installation

1. Clone the repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Koko-boya/ComfyUI-Z-Image-Utilities.git
```

2. Install the extra dependencies required only for the Direct provider:

```bash
pip install bitsandbytes huggingface-hub
```

Optional for faster Hugging Face downloads:

```bash
pip install hf_transfer
```

ComfyUI already provides `torch`, `transformers`, and `accelerate` in a normal installation.

3. Restart ComfyUI.

---

## Included Nodes

| Node | Description |
|------|-------------|
| `Z-Image API Config` | Configure OpenRouter, local API, or direct Hugging Face model loading |
| `Z-Image Options` | Optional inference settings with per-option enable toggles |
| `Z-Image Prompt Enhancer` | Enhance prompts with text-only or vision-capable LLMs |
| `Z-Image Integrated KSampler` | Prompt enhancement plus encoding, sampling, decoding, and optional auto-save |
| `Z-Image Unload Models` | Unload cached direct models from memory |
| `Z-Image Clear Sessions` | Clear saved chat sessions |

---

## Quick Start

Standalone enhancement workflow:

```text
[Z-Image API Config] -> [Z-Image Prompt Enhancer] -> [CLIP Text Encode] -> [KSampler]
                                ^
                    [Z-Image Options] (optional)
```

Integrated generation workflow:

```text
[Checkpoint Loader] -> [Z-Image Integrated KSampler] -> [Preview Image]
         |                         ^
         +--> model / clip / vae   |
                         [Z-Image API Config]
```

If you use the integrated sampler in `image_to_image` mode, connect at least one image input.

---

## Provider Setup

### OpenRouter

- `provider`: `openrouter`
- `model`: any compatible OpenRouter chat model
- `api_key`: your OpenRouter API key

If you want to send an image into `Z-Image Prompt Enhancer`, use a vision-capable model. The current default model value in the node is `qwen/qwen3-235b-a22b:free`.

### Local API Server

The local provider expects an OpenAI-compatible `/v1/chat/completions` endpoint.

- `provider`: `local`
- `model`: model name exposed by your server
- `local_endpoint`: server URL such as `http://localhost:11434/v1`

Common endpoint examples:

| Server | Endpoint |
|--------|----------|
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| vLLM | `http://localhost:8000/v1` |
| text-generation-webui | `http://localhost:5000/v1` |

### Direct Hugging Face Loading

- `provider`: `direct`
- `model`: Hugging Face repo ID
- `quantization`: `4bit`, `8bit`, or `none`
- `device`: `auto`, `cuda`, `cpu`, or `mps`
- `llm_path`: optional local model folder root
- `auto_download_fallback`: if enabled, try `llm_path` first and download if not found

Downloaded direct models are stored under `ComfyUI/models/LLM/Z-Image/`.

If you want to use image input with the Direct provider, choose a vision-language model. Text-only models will reject image input.

---

## Node Reference

### Z-Image API Config

Configures the LLM backend connection.

| Parameter | Notes |
|-----------|-------|
| `provider` | `openrouter`, `local`, or `direct` |
| `model` | OpenRouter model ID, local server model name, or Hugging Face repo ID |
| `api_key` | Used for OpenRouter |
| `local_endpoint` | Used for the local provider |
| `llm_path` | Optional manual model root for the Direct provider |
| `auto_download_fallback` | Direct provider only |
| `quantization` | Direct provider only |
| `device` | Direct provider only |

Output: `config`

### Z-Image Options

Provides optional inference settings. Values are only sent when their matching enable toggle is on.

| Toggle | Value |
|--------|-------|
| `enable_temperature` | `temperature` |
| `enable_top_p` | `top_p` |
| `enable_top_k` | `top_k` |
| `enable_seed` | `seed` |
| `enable_repeat_penalty` | `repeat_penalty` |
| `enable_max_tokens` | `max_tokens` |
| Always available | `debug_mode` |

Notes:

- If `enable_temperature` is off, the provider default temperature is used.
- If `enable_max_tokens` is off, the provider default max token limit is used.
- Support for `seed`, `top_k`, and similar settings depends on the backend model/provider.

Output: `options`

### Z-Image Prompt Enhancer

Enhances a text prompt into a longer image-generation prompt.

| Parameter | Notes |
|-----------|-------|
| `config` | From `Z-Image API Config` |
| `prompt` | Input prompt text |
| `prompt_template` | `auto`, `chinese`, `english`, or `custom` |
| `options` | Optional `ZIMAGE_OPTIONS` input |
| `image` | Optional image input for vision-capable models |
| `retry_count` | Retry attempts on API failure |
| `max_output_length` | Character limit; `0` means unlimited |
| `session_id` | Reuse the same value for persistent conversation history |
| `reset_session` | Clears the named session before use |
| `keep_model_loaded` | Direct provider only; keeps the loaded HF model cached |
| `utf8_sanitize` | Replaces some Unicode punctuation with ASCII-safe characters |
| `custom_system_prompt` | Required only when `prompt_template` is `custom`; must include `{prompt}` |

Notes:

- Blank `session_id` means stateless execution.
- With `debug_mode` enabled, the full report is returned in `debug_log` and also written to logs.

Outputs: `enhanced_prompt`, `debug_log`

### Z-Image Integrated KSampler

All-in-one node for prompt enhancement and image generation.

Required inputs:

- `model`
- `clip`
- `vae`
- `config`
- `positive_prompt`
- `negative_prompt`
- `generation_mode`
- `width`
- `height`
- `seed`
- `steps`
- `cfg`
- `sampler_name`
- `scheduler`

Important optional inputs/settings:

| Parameter | Notes |
|-----------|-------|
| `options` | Optional LLM options |
| `image_1` to `image_4` | Optional images for vision enhancement and `image_to_image` mode |
| `latent` | Optional latent override |
| `prompt_template` | `auto`, `chinese`, `english`, or `custom` |
| `enable_prompt_enhance` | If off, the prompt is used as-is |
| `batch_size` | Number of images to generate |
| `auraflow_shift` | Applies AuraFlow sampling shift when greater than `0` |
| `cfg_norm_strength` | Applies CFGNorm when greater than `0` |
| `enable_clean_gpu` | Clears GPU cache before and after sampling |
| `enable_clean_ram` | Runs garbage collection after completion |
| `auto_save_folder` | Empty disables auto-save; relative paths resolve inside ComfyUI output |
| `output_prefix` | Filename prefix for auto-saved images |
| `custom_system_prompt` | Used with `prompt_template = custom` |
| `instruction` | Vision-language instruction text for `image_to_image` mode |
| `denoise` | Standard KSampler denoise strength |

Notes:

- `image_to_image` mode requires at least one connected image.
- The starting latent is prepared at the requested `width` and `height`.
- This node always calls the prompt enhancer statelessly.

Outputs: `images`, `latent`, `enhanced_prompt`, `debug_log`

### Z-Image Unload Models

Unloads cached Direct-provider models.

| Parameter | Notes |
|-----------|-------|
| `unload_all` | When `true`, unloads all cached direct models |
| `passthrough` | Optional passthrough input for graph chaining |

Output: `status`

### Z-Image Clear Sessions

Clears saved prompt-enhancer session history.

| Parameter | Notes |
|-----------|-------|
| `clear_all` | Clears every saved session |
| `session_id` | Clears one specific session when `clear_all` is `false` |

Output: `status`

---

## Troubleshooting

| Issue | What to check |
|-------|---------------|
| Empty or failed response | Verify API key, endpoint, model name, and `retry_count` |
| Rate limits from OpenRouter | The node retries automatically, but free models can still hit upstream `429` limits |
| Image input fails on Direct provider | Use a vision-language model instead of a text-only model |
| Local provider errors | Make sure the server exposes an OpenAI-compatible `/v1` endpoint |
| Out of memory | Use `4bit`, a smaller model, or unload cached direct models |
| Repetitive output | Enable `repeat_penalty` in `Z-Image Options` |
| Need deeper logs | Turn on `debug_mode` and inspect `debug_log` or `z_image_debug.log` |

---

## Credits

- System prompt source: [Z-Image Turbo Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py)
- Author: [Koko-boya](https://github.com/Koko-boya)

References:

- [comfyui-ollama](https://github.com/stavsap/comfyui-ollama)
- [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL)
- [ComfyUI-EBU-LMStudio](https://github.com/burnsbert/ComfyUI-EBU-LMStudio)

## License

Apache License 2.0. See [LICENSE](LICENSE).
