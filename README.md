# ComfyUI-Z-Image-Utilities

A collection of utility nodes for ComfyUI designed specifically for the [Z-Image](https://github.com/Tongyi-MAI/Z-Image) model.

![ComfyUI-Z-Image-Utilities](https://i.imgur.com/n2Jh9PD.png)

## Installation

1. Navigate to your ComfyUI custom nodes directory and clone the repository:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yourusername/ComfyUI-Z-Image-Utilities.git
```

2. Restart ComfyUI

---

## Included Utilities

| Utility | Description | Status |
|---------|-------------|--------|
| [Prompt Enhancer](#prompt-enhancer) | LLM-powered prompt enhancement using the official Z-Image system prompt | ✅ Available |

---

## Prompt Enhancer

Transform simple prompts into detailed visual descriptions optimized for Z-Image.

This node uses the official prompt enhancement system prompt from [Z-Image Turbo Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py) to expand your prompts via an LLM. Z-Image works best with long, detailed prompts, and this node automates that process.

### Example

**Input:**
```
a cat
```

**Output:**
```
A domestic shorthair cat with orange and white fur sits on a wooden floor. 
The cat has green eyes and is looking directly at the camera with an alert 
expression. Soft natural light from a nearby window illuminates the scene 
from the left, creating gentle shadows. The background shows a blurred 
living room interior with warm tones.
```

### Features

- **3 Backend Options** — Choose what works best for you:
  - **OpenRouter** (Cloud) — Free tier available, no setup required
  - **Local API Server** — Ollama, LM Studio, vLLM, text-generation-webui, etc.
  - **Direct Local Model** — Load models directly via HuggingFace transformers (NEW!)
- **Auto Model Download** — Direct local models download automatically from HuggingFace
- **Quantization Support** — 4-bit/8-bit quantization to run large models on consumer GPUs
- **Bilingual** — Automatically detects and handles Chinese and English prompts
- **Reliable** — Smart retry logic with exponential backoff and rate limit handling
- **Transparent** — Debug output shows full API request/response details
- **CLIP integration** — Optional direct conditioning output for streamlined workflows

### Setup

**Option 1: OpenRouter (Cloud)**
- Get a free API key from [OpenRouter](https://openrouter.ai/keys)
- No installation required

**Option 2: Local API Server**
- Install a local LLM server (examples below)
- Make sure it exposes an OpenAI-compatible API endpoint

**Option 3: Direct Local Model (NEW!)**
- No server needed - models load directly in ComfyUI
- Install dependencies:
```bash
pip install torch transformers accelerate bitsandbytes huggingface-hub
```
- Models auto-download from HuggingFace on first use
- Supports quantization for lower VRAM usage

#### Local API Server Options

| Server | Default Port | Installation |
|--------|--------------|--------------|
| [Ollama](https://ollama.ai/) | 11434 | `curl -fsSL https://ollama.ai/install.sh \| sh` |
| [LM Studio](https://lmstudio.ai/) | 1234 | Download from website |
| [vLLM](https://github.com/vllm-project/vllm) | 8000 | `pip install vllm` |
| [text-generation-webui](https://github.com/oobabooga/text-generation-webui) | 5000 | Follow repo instructions |

**Example: Quick start with Ollama**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., Qwen 2.5)
ollama pull qwen2.5:14b

# The server starts automatically at http://localhost:11434
```

### Nodes

#### Z-Image LLM API Config (OpenRouter / Local)

Configure your API connection once and reuse it across your workflow.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `provider` | Select `openrouter` or `local` | `openrouter` |
| `model` | Model name/ID | OpenRouter: `qwen/qwen3-235b-a22b:free`<br>Local: `qwen2.5:14b` |
| `api_key` | Your OpenRouter API key (not needed for local) | `sk-or-v1-xxxxx` |
| `local_endpoint` | Local LLM server endpoint (used when provider=local) | `http://localhost:11434/v1` |

**Output:** `api_config`

**Example Configurations:**

*OpenRouter (Free):*
```
provider: openrouter
model: qwen/qwen3-235b-a22b:free
api_key: sk-or-v1-xxxxx
local_endpoint: (ignored)
```

*Ollama (Local):*
```
provider: local
model: qwen2.5:14b
api_key: (not needed)
local_endpoint: http://localhost:11434/v1
```

*LM Studio (Local):*
```
provider: local
model: any-model-name
api_key: (not needed)
local_endpoint: http://localhost:1234/v1
```

---

#### Z-Image Direct Local Model Config (NEW!)

Load models directly from HuggingFace without needing a separate API server.

| Parameter | Description | Example |
|-----------|-------------|---------|
| `repo_id` | HuggingFace repository ID | `Qwen/Qwen2.5-7B-Instruct` |
| `quantization` | Memory optimization mode | `4bit` (recommended), `8bit`, or `none` |
| `device` | Where to load the model | `auto` (uses CUDA if available) |

**Output:** `api_config`

**Example Configuration:**

```
repo_id: Qwen/Qwen2.5-7B-Instruct
quantization: 4bit
device: auto
```

**Recommended Models:**
- `Qwen/Qwen2.5-7B-Instruct` (7B params, good balance)
- `Qwen/Qwen2.5-14B-Instruct` (14B params, better quality)
- `Qwen/Qwen2.5-3B-Instruct` (3B params, faster, lower quality)

**Memory Requirements (approximate):**
| Model Size | Full Precision | 8-bit | 4-bit |
|------------|---------------|-------|-------|
| 3B | ~12GB | ~6GB | ~3GB |
| 7B | ~28GB | ~14GB | ~7GB |
| 14B | ~56GB | ~28GB | ~14GB |

---

#### Z-Image Prompt Enhancer

The core enhancement node.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api_config` | Configuration from the API Router node | — |
| `prompt` | Your input text | — |
| `output_language` | `auto`, `english`, or `chinese` | `auto` |
| `temperature` | Creativity level (0.1–1.5) | 0.7 |
| `max_tokens` | Maximum output length (256–8192) | 2048 |
| `retry_count` | Retry attempts on failure (0–10) | 1 |

**Outputs:** `enhanced_prompt`, `debug_log`

---

#### Z-Image Prompt Enhancer + CLIP

Same as above, but outputs CLIP conditioning directly.

**Additional Input:** `clip` — CLIP model from your checkpoint loader

**Outputs:** `conditioning`, `enhanced_prompt`, `debug_log`

### Example Workflows

**Standard:**
```
[API Router] → [Prompt Enhancer] → [CLIP Text Encode] → [KSampler]
```

**Streamlined:**
```
[Checkpoint] → [Prompt Enhancer + CLIP] → [KSampler]
                        ↑
                  [API Router]
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Empty response errors | **OpenRouter:** Verify API key and increase `retry_count`<br>**Local API:** Check server is running and endpoint is correct<br>**Direct Local:** Check debug log for errors |
| Rate limiting | The node respects `Retry-After` headers automatically; wait a few minutes if persistent |
| Connection errors (API) | 1. Verify server is running (`ollama list` or check LM Studio)<br>2. Check endpoint URL matches server port<br>3. Ensure firewall allows local connections |
| Model not found (API) | Make sure the model is downloaded (`ollama pull <model>` for Ollama) |
| Out of memory (Direct Local) | 1. Use `4bit` quantization instead of `8bit` or `none`<br>2. Try a smaller model (e.g., 3B instead of 7B)<br>3. Close other GPU applications |
| Transformers not installed | Run: `pip install torch transformers accelerate bitsandbytes huggingface-hub` |
| Model download failed | 1. Check internet connection<br>2. Verify HuggingFace repo ID is correct<br>3. Check disk space in ComfyUI models directory |
| CUDA errors | 1. Update GPU drivers<br>2. Update PyTorch: `pip install --upgrade torch`<br>3. Try `device: cpu` for testing (slower) |
| Unexpected output | Check `debug_log` for full request/response details |

---

## Roadmap

More Z-Image utilities coming soon.

*Have a suggestion? Open an issue!*

---

## Credits

- **System prompt:** [Z-Image Turbo Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py) by Tongyi-MAI
- **Author:** Kokoboy

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.