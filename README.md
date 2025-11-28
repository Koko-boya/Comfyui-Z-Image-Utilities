# ComfyUI-ZImage-Utilities

A powerful prompt enhancement utility for ComfyUI that uses the FREE Qwen API via OpenRouter (or any other OpenRouter model) to transform simple prompts into detailed, aesthetically pleasing visual descriptions.

## Features

- **Free & Powerful**: Uses the Qwen model via OpenRouter's free tier by default.
- **Prompt Enhancement**: Turns basic ideas into "logic cage" optimized prompts for better image generation.
- **Bilingual Support**: Automatically detects and handles Chinese and English prompts.
- **Strict Error Handling**: Robust error management with clear feedback.
- **Smart Rate Limiting**: Automatically respects `Retry-After` headers to avoid API bans.
- **Detailed Logging**: Comprehensive debug logs for troubleshooting.
- **CLIP Integration**: Optional node to directly output CLIP conditioning.

## Nodes

### 1. Z-Image OpenRouter API Router
Configuration node for the API connection.
- **api_key**: Your OpenRouter API key.
- **model**: The model ID to use (default: `qwen/qwen3-235b-a22b:free`).

### 2. Z-Image Prompt Enhancer
The core node that takes a prompt and returns the enhanced version.
- **Inputs**:
    - `qwen_api`: Connection from the API Router node.
    - `prompt`: Your input text.
    - `output_language`: Force "english", "chinese", or "auto" (default).
    - `temperature`: Creativity control (default: 0.7).
    - `max_tokens`: Maximum length of the generated prompt.
    - `retry_count`: How many times to retry on failure.
- **Outputs**:
    - `enhanced_prompt`: The final text.
    - `debug_log`: Technical details of the generation process.

### 3. Z-Image Prompt Enhancer + CLIP
Same as above, but also encodes the text using a CLIP model.
- **Inputs**: Same as above, plus a `clip` input.
- **Outputs**:
    - `conditioning`: Ready-to-use conditioning for KSampler.
    - `enhanced_prompt`: The text.
    - `debug_log`: Technical details.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/ComfyUI-ZImage-Utilities.git
   ```
2. Restart ComfyUI.

## Configuration

1. Get a free API key from [OpenRouter](https://openrouter.ai/keys).
2. In ComfyUI, add the **Z-Image OpenRouter API Router** node.
3. Paste your API key into the `api_key` field.

## Usage Example

**Basic Workflow:**
```
[Z-Image OpenRouter API Router] 
       |
       v
[Z-Image Prompt Enhancer] <-- [Primitive String (Your Prompt)]
       |
       v
[CLIP Text Encode] --> [KSampler]
```

**Advanced Workflow (with CLIP):**
```
[Checkpoint Loader] --> [Z-Image Prompt Enhancer + CLIP] --> [KSampler]
                                   ^
                                   |
                     [Z-Image OpenRouter API Router]
```

## License

MIT
