# ComfyUI-ZImage-Utilities

A collection of utility nodes for ComfyUI that enhance Z-Image generation quality through intelligent prompt optimization using free LLM APIs etc

![ComfyUI-ZImage-Utilities](https://i.imgur.com/n2Jh9PD.png)

## What It Does

Transforms simple prompts into detailed, precise visual descriptions using the **"Logic Cage"** methodology—a structured approach that preserves your intent while adding professional-grade aesthetic and compositional details.

**Example:**
```
Input: "a cat"

Output: "A domestic shorthair cat with orange and white fur sits on a wooden floor. 
The cat has green eyes and is looking directly at the camera with an alert expression. 
Soft natural light from a nearby window illuminates the scene from the left, creating 
gentle shadows. The background shows a blurred living room interior with warm tones..."
```

## Features

✅ **100% Free** - Uses OpenRouter's free tier models (Qwen by default)  
✅ **Bilingual** - Auto-detects and handles Chinese/English prompts  
✅ **Reliable** - Smart retry logic with exponential backoff  
✅ **Transparent** - Detailed debug logs for every generation  
✅ **CLIP Integration** - Direct conditioning output for streamlined workflows

## Installation

1. Clone into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/yourusername/ComfyUI-ZImage-Utilities.git
   ```

2. Restart ComfyUI

3. Get a free API key from [OpenRouter](https://openrouter.ai/keys)

## Nodes

### Z-Image OpenRouter API Router
Configure your API connection. Set once, use everywhere.

**Parameters:**
- `api_key` - Your OpenRouter API key
- `model` - Model ID (default: `qwen/qwen3-235b-a22b:free`)

---

### Z-Image Prompt Enhancer
Core enhancement node. Transforms prompts using the Logic Cage methodology.

**Inputs:**
- `qwen_api` - API config from Router node
- `prompt` - Your input text
- `output_language` - `auto`, `english`, or `chinese`
- `temperature` - Creativity (0.1-1.5, default: 0.7)
- `max_tokens` - Output length (256-8192, default: 2048)
- `retry_count` - Retries on failure (0-10, default: 1)

**Outputs:**
- `enhanced_prompt` - The improved text
- `debug_log` - Detailed generation info

---

### Z-Image Prompt Enhancer + CLIP
Same as above, but directly outputs CLIP conditioning for KSamplers.

**Additional Input:**
- `clip` - CLIP model from checkpoint

**Outputs:**
- `conditioning` - Ready for KSampler
- `enhanced_prompt` - The text
- `debug_log` - Generation info

## Usage

### Basic Workflow
```
[API Router] → [Prompt Enhancer] → [CLIP Text Encode] → [KSampler]
                      ↑
                [Your Prompt]
```

### Streamlined Workflow
```
[Checkpoint] → [Prompt Enhancer + CLIP] → [KSampler]
                          ↑
                    [API Router]
```

## The Logic Cage Methodology

The enhancement process follows a strict logical sequence:

1. **Lock Core Elements** - Subject, quantity, actions, colors, text are preserved
2. **Generative Reasoning** - If needed, constructs a complete visualizable solution
3. **Aesthetic Injection** - Adds composition, lighting, materials, color schemes
4. **Text Precision** - Handles all text elements with exact quoting and placement

This ensures outputs are faithful to your intent while being dramatically more detailed and visually specific.

## Credits

- **Prompt Template**: Adapted from [Z-Image Turbo Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py)
- **Author**: Kokoboy
