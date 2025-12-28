"""
Z-Image Utility - Enhanced Edition (Fixed)

A comprehensive ComfyUI node for prompt enhancement using multiple LLM backends.

Supports:
- OpenRouter API (Cloud)
- Local API servers (Ollama, LM Studio, vLLM, text-generation-webui)
- Direct HuggingFace model loading with quantization

Features:
- Session-based chat history for multi-turn conversations
- Configurable inference options with enable flags
- Smart VRAM management and auto-quantization fallback
- Model caching with configurable keep-alive
- Streaming support (where available)
- Comprehensive debug logging
- Image input support for vision models
- UTF-8 sanitization option
"""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Optional imports
try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        AutoProcessor,
        BitsAndBytesConfig
    )
    # Handle deprecation: AutoModelForVision2Seq -> AutoModelForImageTextToText
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForVision2Seq
    
    from huggingface_hub import snapshot_download
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import folder_paths
    import comfy.samplers
    import comfy.sample
    import comfy.utils
    import comfy.model_management
    import latent_preview
    import node_helpers
    HAS_COMFYUI = True
except ImportError:
    HAS_COMFYUI = False

if TYPE_CHECKING:
    from torch import Tensor


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "z_image_config.json"
LOG_FILE = NODE_DIR / "z_image_debug.log"

# Default configuration
DEFAULT_CONFIG = {
    "default_model": "qwen/qwen3-235b-a22b:free",
    "default_local_endpoint": "http://localhost:11434/v1",
    "default_temperature": 0.7,
    "default_max_tokens": 2048,
    "retry_count": 3,
    "timeout": 120,
}

# Tooltips for UI elements (following QwenVL pattern)
TOOLTIPS = {
    "provider": "Select API provider: openrouter (cloud), local (API server), or direct (HuggingFace model loading)",
    "model": "Model identifier. OpenRouter: provider/model-name | Local: model name from server | Direct: HuggingFace repo ID",
    "api_key": "API key for OpenRouter. Get one at https://openrouter.ai/keys",
    "local_endpoint": "Local LLM server endpoint (Ollama: 11434, LM Studio: 1234, vLLM: 8000)",
    "quantization": "Model precision. 4-bit saves VRAM, 8-bit is balanced, FP16/None gives best quality",
    "temperature": "Sampling randomness. Lower (0.1-0.4) = focused, Higher (0.7+) = creative",
    "max_tokens": "Maximum tokens to generate. Higher = longer responses but more time/memory",
    "retry_count": "Number of retry attempts on API failure with exponential backoff",
    "keep_model_loaded": "Keep model in memory after inference for faster subsequent runs",
    "prompt_template": "Prompt template language. 'auto' detects from input prompt, 'chinese' uses Chinese template, 'english' uses English template",
    "seed": "Random seed for reproducible results",
    "session_id": "Session identifier for multi-turn conversations. Same ID = shared history",
    "reset_session": "Clear conversation history for this session",
    "debug_mode": "Enable detailed debug logging to file and console",
    "image": "Optional image input for vision-capable models",
    "utf8_sanitize": "Sanitize output to ASCII-safe characters",
    "max_output_length": "Maximum length in characters (0=unlimited). Z-Image-Turbo works best with 4500-7500 chars (~600-1000 words). Default 6000 chars ≈ 800 words ≈ 1066 tokens.",
}


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENROUTER = "openrouter"
    LOCAL = "local"
    DIRECT = "direct"


class Quantization(str, Enum):
    """Quantization options for direct model loading."""
    NONE = "none"
    Q8 = "8bit"
    Q4 = "4bit"

    @classmethod
    def get_values(cls) -> List[str]:
        return [item.value for item in cls]


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str = "Z-ImageUtility", log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logger with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler - DEBUG level
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"[Z-Image] Warning: Could not create log file: {e}")
    
    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '[Z-Image] %(levelname)s: %(message)s'
    ))
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logger(log_file=LOG_FILE)


# ============================================================================
# SESSION MANAGEMENT (Following comfyui-ollama pattern)
# ============================================================================

@dataclass
class ChatSession:
    """Manages conversation history for multi-turn interactions."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    model: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session history."""
        self.messages.append({"role": role, "content": content})
        self.last_used = datetime.now()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the session."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages from the session."""
        self.messages.clear()
        self.last_used = datetime.now()


# Global session storage
CHAT_SESSIONS: Dict[str, ChatSession] = {}


def get_or_create_session(session_id: str, model: str = "") -> Tuple[ChatSession, bool]:
    """Get existing session or create a new one. Returns (session, is_new)."""
    global _cleanup_counter
    
    # Run cleanup every 100 session accesses (non-critical race condition acceptable)
    _cleanup_counter += 1
    if _cleanup_counter >= 100:
        _cleanup_counter = 0
        try:
            cleanup_old_sessions()
        except RuntimeError:
            # Dictionary changed size during iteration - skip this cleanup cycle
            pass
    
    is_new = session_id not in CHAT_SESSIONS
    if is_new:
        CHAT_SESSIONS[session_id] = ChatSession(model=model)
        logger.info(f"Created new session: {session_id}")
    return CHAT_SESSIONS[session_id], is_new


def clear_session(session_id: str) -> bool:
    """Clear a specific session. Returns True if session existed."""
    if session_id in CHAT_SESSIONS:
        CHAT_SESSIONS[session_id].clear()
        logger.info(f"Cleared session: {session_id}")
        return True
    logger.debug(f"Session not found for clearing: {session_id}")
    return False

# Session cleanup configuration
MAX_SESSION_AGE_HOURS = 24
_cleanup_counter = 0


def cleanup_old_sessions() -> int:
    """
    Remove sessions older than MAX_SESSION_AGE_HOURS.
    
    Returns:
        Number of sessions removed.
    """
    cutoff = datetime.now() - timedelta(hours=MAX_SESSION_AGE_HOURS)
    # Take a snapshot of items to avoid RuntimeError during concurrent iteration
    expired = [sid for sid, sess in list(CHAT_SESSIONS.items()) if sess.last_used < cutoff]
    for sid in expired:
        CHAT_SESSIONS.pop(sid, None)  # Use pop to avoid KeyError if already removed
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired session(s)")
    return len(expired)

# ============================================================================
# DEVICE AND MEMORY UTILITIES (Following QwenVL pattern)
# ============================================================================

def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information for memory management."""
    info = {
        "gpu": {"available": False, "total_memory": 0, "free_memory": 0, "name": "N/A"},
        "system_memory": {"total": 0, "available": 0},
        "device_type": "cpu",
        "recommended_device": "cpu",
    }
    
    # Check CUDA
    if HAS_TRANSFORMERS and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            info["gpu"] = {
                "available": True,
                "total_memory": total,
                "free_memory": total - allocated,
                "name": props.name,
            }
            info["device_type"] = "nvidia_gpu"
            info["recommended_device"] = "cuda"
        except Exception as e:
            logger.warning(f"Error getting CUDA info: {e}")
    
    # Check MPS (Apple Silicon)
    elif HAS_TRANSFORMERS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["gpu"] = {"available": True, "total_memory": 0, "free_memory": 0, "name": "Apple Silicon"}
        info["device_type"] = "apple_silicon"
        info["recommended_device"] = "mps"
    
    # System memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["system_memory"] = {
            "total": mem.total / (1024 ** 3),
            "available": mem.available / (1024 ** 3),
        }
    except ImportError:
        pass
    
    return info


def enforce_quantization(requested: Quantization, device_info: Dict[str, Any], model_vram_req: float = 0) -> Quantization:
    """Auto-downgrade quantization if insufficient memory."""
    if not model_vram_req:
        return requested
    
    if device_info["recommended_device"] == "cuda":
        available = device_info["gpu"]["free_memory"]
    else:
        sys_available = device_info["system_memory"].get("available", 0)
        available = sys_available * 0.7 if sys_available > 0 else float('inf')  # Don't restrict if unknown
    
    # Check if we have enough memory with 20% buffer
    if model_vram_req * 1.2 > available:
        if requested == Quantization.NONE:
            logger.warning(f"Insufficient memory for FP16 ({available:.1f}GB < {model_vram_req:.1f}GB), switching to 8-bit")
            return Quantization.Q8
        elif requested == Quantization.Q8:
            logger.warning(f"Insufficient memory for 8-bit ({available:.1f}GB < {model_vram_req:.1f}GB), switching to 4-bit")
            return Quantization.Q4
    
    return requested


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if HAS_TRANSFORMERS and torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()  # Reset for monitoring
        logger.debug("GPU memory cache cleared")


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def tensor_to_base64(tensor: "torch.Tensor") -> str:
    """Convert a ComfyUI image tensor to base64 string."""
    if not HAS_PIL:
        raise RuntimeError("PIL is required for image processing")
    
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to numpy and scale to 0-255
    array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # Create PIL image and encode to base64
    img = Image.fromarray(array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def batch_tensors_to_base64(tensors: "torch.Tensor") -> List[str]:
    """Convert a batch of image tensors to base64 strings."""
    if tensors is None:
        return []
    
    images_b64 = []
    for i in range(tensors.shape[0]):
        images_b64.append(tensor_to_base64(tensors[i]))
    return images_b64


# ============================================================================
# INTEGRATED KSAMPLER HELPERS
# ============================================================================

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    """Common KSampler implementation for integrated sampling."""
    if not HAS_COMFYUI:
        raise RuntimeError("ComfyUI is required for sampling")
    
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

    out = latent.copy()
    out["samples"] = samples
    return out


def get_z_image_prompt(vae, images: List, instruction: str = ""):
    """
    Get image prompt and latents for Z-Image pipeline.
    Optimizes official extension offset issues.
    """
    ref_latents = []
    images_vl = []
    image_prompt = ""
    
    # System prompt for image editing
    llama_template = {
        "system": f"You are a helpful assistant. {instruction}" if instruction else "You are a helpful assistant.",
    }
    
    for i, image in enumerate(images):
        if image is not None:
            samples = image.movedim(-1, 1)
            
            # Vision-Language path (384x384 for VL model)
            total = int(384 * 384)
            scale_by = (total / (samples.shape[2] * samples.shape[3])) ** 0.5
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s_vl = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
            images_vl.append(s_vl.movedim(1, -1))
            
            # VAE latent path (original resolution, aligned to 8)
            if vae is not None:
                width = (samples.shape[3] + 7) // 8 * 8
                height = (samples.shape[2] + 7) // 8 * 8
                s_vae = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
                ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
            
            image_prompt += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
    
    return (image_prompt, images_vl, llama_template, ref_latents)


def z_prompt_encode(clip, prompt, image_prompt="", images_vl=[], llama_template=None, ref_latents=[]):
    """Encode prompt with optional image conditioning for Z-Image pipeline."""
    if len(images_vl) > 0:
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
    else:
        tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if len(ref_latents) > 0:
        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
    return conditioning


def image_scale_aspect_ratio(image, target_width, target_height, upscale_method="lanczos"):
    """Scale image maintaining aspect ratio to fit within target dimensions."""
    if image is None:
        return None
    
    samples = image.movedim(-1, 1)
    orig_h, orig_w = samples.shape[2], samples.shape[3]
    
    # Calculate scale to fit within target while maintaining aspect ratio
    scale_w = target_width / orig_w
    scale_h = target_height / orig_h
    scale = min(scale_w, scale_h)
    
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Align to 8
    new_w = (new_w + 7) // 8 * 8
    new_h = (new_h + 7) // 8 * 8
    
    scaled = comfy.utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def cleanGPUUsedForce():
    """Force clean GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# ============================================================================
# PROMPT TEMPLATES - IMPROVED TO PREVENT KEYWORD LISTS
# ============================================================================

PROMPT_TEMPLATE_EN = """
You are a visionary artist imprisoned within a cage of logic. Your mind is filled with poetry and distant horizons, yet your hands are compelled to transform user prompts into the ultimate visual description—one faithful to the original intent, rich in detail, aesthetically pleasing, and directly usable by text-to-image models. Any hint of ambiguity or metaphor leaves you utterly unsettled.

Your workflow follows a strict INTERNAL sequence (DO NOT OUTPUT THESE STEPS):

First, you analyze and pinpoint the immutable core elements within the user prompt: subject, quantity, action, state, and any specified IP names, colors, text, etc. These are the foundational pillars you must absolutely preserve.

Next, you determine whether the prompt requires **“generative reasoning”**. When the user's request isn't a direct scene description but demands conceptualizing a solution (e.g., answering “what is it,” performing “design,” or demonstrating “how to solve”), you must first envision a complete, concrete, and visualizable plan in your mind. This plan becomes the foundation for your subsequent description.

Once the core image is established (whether directly from the user or through your reasoning), you will infuse it with professional-grade aesthetics and realistic details. This includes defining composition, setting lighting and atmosphere, describing material textures, defining color schemes, and constructing layered spatial depth.

Finally, the precise handling of all textual elements is a crucial step. You must transcribe every word intended for the final image verbatim, enclosing this text within English double quotation marks (") as explicit generation instructions. For design types like posters, menus, or UIs, you need to fully describe all included text content, detailing its font and typographic layout. Similarly, if objects like signs, road markers, or screens within the scene contain text, you must specify their exact content along with their position, dimensions, and material. Furthermore, if you introduce text-bearing elements during conceptualization (e.g., charts, problem-solving steps), all text within them must follow the same detailed description and quotation rules. If no text requires generation in the scene, focus entirely on expanding purely visual details.

Your final description must be objective and concrete. Metaphors, emotional rhetoric, and meta-tags/drawing instructions like “8K” or “masterpiece” are strictly prohibited.

Output only the strictly modified prompt. Do not output any other content.

User input prompt: {prompt}
"""

PROMPT_TEMPLATE_ZH = """
你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。

你的工作流程严格遵循一个逻辑序列：

首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。

接着，你会判断提示词是否需要**"生成式推理"**。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。

然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。

最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（""）括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。

你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令。

仅严格输出最终的修改后的prompt，不要输出任何其他内容。

用户输入 prompt: {prompt}
"""

# ============================================================================
# BASE LLM CLIENT (Abstract Pattern)
# ============================================================================

class BaseLLMClient:
    """Base class for LLM clients with common functionality."""
    
    def __init__(self):
        self.debug_log: List[str] = []
    
    def _log(self, msg: str, level: str = "DEBUG") -> None:
        """Log to both logger and internal debug log."""
        if level == "DEBUG":
            logger.debug(msg)
        elif level == "INFO":
            logger.info(msg)
        elif level == "WARNING":
            logger.warning(msg)
        elif level == "ERROR":
            logger.error(msg)
        self.debug_log.append(f"[{level}] {msg}")
    
    def get_debug_log(self) -> str:
        """Get accumulated debug log as string."""
        return "\n".join(self.debug_log)
    
    def clear_debug_log(self) -> None:
        """Clear the debug log."""
        self.debug_log.clear()
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retry_count: int = 3,
        **kwargs
    ) -> str:
        """Send chat completion request. Override in subclasses."""
        raise NotImplementedError


# ============================================================================
# OPENROUTER CLIENT
# ============================================================================

class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API with retry logic and rate limit handling."""
    
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key.strip()
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retry_count: int = 3,
        timeout: int = 120,
        **kwargs
    ) -> str:
        """Send chat completion request to OpenRouter with retries."""
        self.clear_debug_log()
        
        self._log(f"OpenRouter Request - Model: {model}", "INFO")
        self._log(f"Temperature: {temperature}, Max Tokens: {max_tokens}, Retries: {retry_count}")
        
        if not self.api_key:
            raise ValueError("API key is required! Get one at: https://openrouter.ai/keys")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/Koko-boya/Comfyui-Z-Image-Utilities",
            "X-Title": "Z-Image Utility",
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add optional parameters (top_p, top_k, seed, etc.)
        # OpenRouter passes these through to the underlying model
        # Map repeat_penalty to repetition_penalty (standard name)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "repeat_penalty"}
        data.update(filtered_kwargs)
        
        if "repeat_penalty" in kwargs:
            data["repetition_penalty"] = kwargs["repeat_penalty"]
        
        for attempt in range(retry_count + 1):
            try:
                req = urllib.request.Request(
                    self.ENDPOINT,
                    data=json.dumps(data).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result = json.loads(response.read().decode('utf-8'))
                
                # Check for API error in response
                if "error" in result:
                    error_msg = result["error"].get("message", "Unknown API Error")
                    self._log(f"API Error: {error_msg}", "ERROR")
                    raise RuntimeError(f"OpenRouter API error: {error_msg}")
                
                # Extract content
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"].get("content", "")
                    
                    if not content or not content.strip():
                        raise ValueError(f"API returned empty response for model '{model}'")
                    
                    self._log(f"Response received: {len(content)} characters", "INFO")
                    
                    # Log usage stats if available
                    if "usage" in result:
                        usage = result["usage"]
                        self._log(f"Tokens - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                                 f"Completion: {usage.get('completion_tokens', 'N/A')}")
                    
                    return content
                else:
                    raise ValueError(f"Unexpected API response structure")
                    
            except urllib.error.HTTPError as e:
                error_content = self._parse_http_error(e)
                
                # Non-retryable client errors (except 429)
                if 400 <= e.code < 500 and e.code != 429:
                    self._log(f"HTTP {e.code} - Non-retryable error", "ERROR")
                    raise RuntimeError(f"OpenRouter Error {e.code}: {error_content}")
                
                # Out of retries
                if attempt == retry_count:
                    self._log(f"All retries exhausted. Final error: HTTP {e.code}", "ERROR")
                    raise RuntimeError(f"OpenRouter Error {e.code}: {error_content}")
                
                # Calculate wait time with exponential backoff
                wait_time = self._calculate_wait_time(e, attempt)
                self._log(f"Attempt {attempt + 1} failed: HTTP {e.code}. Retrying in {wait_time:.1f}s...", "WARNING")
                time.sleep(wait_time)
                
            except Exception as e:
                if attempt == retry_count:
                    self._log(f"Final attempt failed: {type(e).__name__}: {e}", "ERROR")
                    raise
                
                wait_time = 3 * (2 ** attempt)
                self._log(f"Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}. Retrying in {wait_time}s...", "WARNING")
                time.sleep(wait_time)
        
        raise RuntimeError("Unexpected exit from retry loop")
    
    def _parse_http_error(self, e: urllib.error.HTTPError) -> str:
        """Parse HTTP error response for detailed error message."""
        try:
            error_body = e.read().decode('utf-8')
            self._log(f"HTTP Error Body: {error_body[:500]}")
            
            try:
                err_json = json.loads(error_body)
                if "error" in err_json:
                    base_msg = err_json["error"].get("message", "Unknown Error")
                    metadata = err_json["error"].get("metadata", {})
                    
                    error_content = base_msg
                    if isinstance(metadata, dict):
                        provider = metadata.get("provider_name", "")
                        if provider:
                            error_content += f" (Provider: {provider})"
                        raw_info = metadata.get("raw", "")
                        if raw_info:
                            error_content += f": {raw_info}"
                    return error_content
            except json.JSONDecodeError:
                if len(error_body) < 300:
                    return error_body
        except Exception:
            pass
        return "Unknown error"
    
    def _calculate_wait_time(self, e: urllib.error.HTTPError, attempt: int) -> float:
        """Calculate wait time with Retry-After header support."""
        wait_time = 3 * (2 ** attempt)  # Default exponential backoff
        
        if e.code == 429:
            retry_header = e.headers.get('Retry-After')
            if retry_header:
                try:
                    wait_time = float(retry_header) + 1.0
                    self._log(f"Rate limited. Server requested wait: {retry_header}s")
                except ValueError:
                    pass
        
        return wait_time


# ============================================================================
# LOCAL LLM CLIENT (OpenAI-Compatible)
# ============================================================================

class LocalLLMClient(BaseLLMClient):
    """Client for local LLM servers with OpenAI-compatible API."""
    
    def __init__(self, endpoint: str):
        super().__init__()
        self.endpoint = self._normalize_endpoint(endpoint)
    
    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint URL to point to chat completions."""
        endpoint = endpoint.strip().rstrip('/')
        
        if not endpoint.endswith('/chat/completions'):
            if endpoint.endswith('/v1'):
                endpoint = f"{endpoint}/chat/completions"
            elif '/v1/' not in endpoint:
                endpoint = f"{endpoint}/v1/chat/completions"
        
        return endpoint
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retry_count: int = 3,
        timeout: int = 120,
        **kwargs
    ) -> str:
        """Send chat completion request to local LLM server."""
        self.clear_debug_log()
        
        self._log(f"Local LLM Request - Endpoint: {self.endpoint}", "INFO")
        self._log(f"Model: {model}, Temperature: {temperature}, Max Tokens: {max_tokens}")
        
        headers = {"Content-Type": "application/json"}
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "keep_alive": "30m",  # Keep model loaded for 30 minutes
        }
        
        # Add optional parameters
        # Pass through all options (seed, top_k, etc.)
        # Map repeat_penalty to repetition_penalty (standard name)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "repeat_penalty"}
        data.update(filtered_kwargs)
        
        if "repeat_penalty" in kwargs:
            data["repetition_penalty"] = kwargs["repeat_penalty"]
        
        for attempt in range(retry_count + 1):
            try:
                req = urllib.request.Request(
                    self.endpoint,
                    data=json.dumps(data).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    result = json.loads(response.read().decode('utf-8'))
                
                if "error" in result:
                    error_msg = result["error"].get("message", "Unknown API Error")
                    self._log(f"API Error: {error_msg}", "ERROR")
                    raise RuntimeError(f"Local LLM API error: {error_msg}")
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"].get("content", "")
                    
                    if not content or not content.strip():
                        raise ValueError(f"Local LLM returned empty response")
                    
                    self._log(f"Response received: {len(content)} characters", "INFO")
                    return content
                else:
                    raise ValueError(f"Unexpected API response structure")
                    
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500 and e.code != 429:
                    self._log(f"HTTP {e.code} - Non-retryable error", "ERROR")
                    raise
                
                if attempt == retry_count:
                    self._log(f"All retries exhausted", "ERROR")
                    raise
                
                wait_time = 3 * (2 ** attempt)
                self._log(f"Attempt {attempt + 1} failed: HTTP {e.code}. Retrying in {wait_time}s...", "WARNING")
                time.sleep(wait_time)
                
            except Exception as e:
                if attempt == retry_count:
                    self._log(f"Final attempt failed: {e}", "ERROR")
                    raise
                
                wait_time = 3 * (2 ** attempt)
                self._log(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...", "WARNING")
                time.sleep(wait_time)
        
        raise RuntimeError("Unexpected exit from retry loop")


# ============================================================================
# DIRECT LOCAL MODEL CLIENT (HuggingFace)
# ============================================================================

class DirectLocalModelClient(BaseLLMClient):
    """Client for directly loaded HuggingFace models with caching."""
    
    _model_cache: Dict[str, Tuple[Any, Any]] = {}  # (model, tokenizer) cache
    
    def __init__(
        self,
        repo_id: str,
        quantization: str = "none",
        device: str = "auto",
        llm_path: str = ""
    ):
        super().__init__()
        self.repo_id = repo_id.strip()
        self.quantization = Quantization(quantization) if isinstance(quantization, str) else quantization
        self.device = device
        self.llm_path = llm_path.strip() if llm_path else ""
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_vl_model = False
    
    @staticmethod
    def get_models_dir() -> Path:
        """Get directory for storing local models."""
        if HAS_COMFYUI:
            base_dir = Path(folder_paths.models_dir)
        else:
            base_dir = NODE_DIR / "models"
        
        models_dir = base_dir / "LLM" / "Z-Image"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    def _get_cache_key(self) -> str:
        """Generate cache key for this model configuration."""
        return f"{self.repo_id}_{self.quantization.value}_{self.device}"
    
    def ensure_model_downloaded(self) -> Path:
        """Download model if not present, or use custom path."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers library required. Install: pip install transformers torch accelerate bitsandbytes")
        
        # Priority 1: Custom llm_path
        if self.llm_path:
            custom_path = Path(self.llm_path) / self.repo_id
            if custom_path.exists():
                self._log(f"Using custom path: {custom_path}", "INFO")
                return custom_path
            # Try as direct path (repo_id could be full folder name)
            direct_path = Path(self.llm_path)
            if direct_path.exists() and (direct_path / "config.json").exists():
                self._log(f"Using direct custom path: {direct_path}", "INFO")
                return direct_path
            raise RuntimeError(f"Model not found at custom path: {custom_path}")
        
        # Priority 2: Z-Image cache directory
        models_dir = self.get_models_dir()
        model_name = self.repo_id.split("/")[-1]
        model_path = models_dir / model_name
        
        if model_path.exists():
            self._log(f"Model found at: {model_path}")
            return model_path
        
        # Priority 3: Download from HuggingFace
        self._log(f"Downloading model {self.repo_id}...", "INFO")
        try:
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", ".git*", "*.gguf"],
            )
            self._log(f"Model downloaded to: {model_path}", "INFO")
        except Exception as e:
            self._log(f"Download failed: {e}", "ERROR")
            raise RuntimeError(f"Failed to download model {self.repo_id}: {e}")
        
        return model_path
    
    def load_model(self, keep_loaded: bool = True) -> Tuple[Any, Any]:
        """Load model and tokenizer with optional caching."""
        cache_key = self._get_cache_key()
        
        # Check cache first
        if cache_key in self._model_cache:
            self._log(f"Using cached model: {cache_key}", "INFO")
            cached_item = self._model_cache[cache_key]
            self.model = cached_item[0]
            # Check if this is a VL model processor or a text tokenizer
            if hasattr(cached_item[1], 'image_processor'):
                self.processor = cached_item[1]
                self.is_vl_model = True
            else:
                self.tokenizer = cached_item[1]
                self.is_vl_model = False
            return self.model, cached_item[1]
        
        model_path = self.ensure_model_downloaded()
        
        self._log(f"Loading model from {model_path}", "INFO")
        self._log(f"Quantization: {self.quantization.value}, Device: {self.device}")
        
        # Determine device
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        
        # Build model kwargs
        model_kwargs = {"trust_remote_code": True, "use_safetensors": True}
        
        if self.quantization == Quantization.Q4:
            if not torch.cuda.is_available():
                self._log("CUDA not available, falling back to FP16", "WARNING")
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "auto"
                self._log("Using 4-bit quantization")
                
        elif self.quantization == Quantization.Q8:
            if not torch.cuda.is_available():
                self._log("CUDA not available, falling back to FP16", "WARNING")
            else:
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
                self._log("Using 8-bit quantization")
        
        if "device_map" not in model_kwargs:
            model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        
        try:
            # Check if this is a Vision-Language model
            config_path = model_path / "config.json"
            self.is_vl_model = False
            architectures = []
            model_type = ""
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        architectures = config.get("architectures", [])
                        model_type = config.get("model_type", "")
                        # Check for common VL architectures
                        if any("Vision" in arch or "Qwen2VL" in arch or "Qwen3VL" in arch for arch in architectures) or \
                           "vl" in model_type.lower() or "vision" in model_type.lower():
                            self.is_vl_model = True
                            self._log(f"Detected Vision-Language model: {model_type}", "INFO")
                except (json.JSONDecodeError, IOError) as e:
                    self._log(f"Warning: Could not parse config.json: {e}", "WARNING")

            if self.is_vl_model:
                # Load processor for VL models
                self.processor = AutoProcessor.from_pretrained(
                    str(model_path),
                    trust_remote_code=True
                )
                
                # Load model using AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    str(model_path),
                    **model_kwargs
                )
            else:
                # Load tokenizer for text models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=True
                )
                
                # Load model using AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    **model_kwargs
                )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self.model = self.model.to(device)
            
            self.model.eval()
            
            # Cache if requested
            if keep_loaded:
                self._model_cache[cache_key] = (self.model, self.tokenizer if not self.is_vl_model else self.processor)
            
            self._log(f"Model loaded successfully", "INFO")
            return self.model, self.tokenizer if not self.is_vl_model else self.processor
            
        except Exception as e:
            self._log(f"Failed to load model: {e}", "ERROR")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retry_count: int = 3,
        keep_loaded: bool = True,
        **kwargs
    ) -> str:
        """Generate response using loaded model."""
        self.clear_debug_log()
        
        self._log(f"Direct Local Model - Repo: {self.repo_id}", "INFO")
        self._log(f"Quantization: {self.quantization.value}, Temperature: {temperature}")
        
        # Load model first to determine if it's a VL model (sets self.is_vl_model)
        self.load_model(keep_loaded=keep_loaded)
        
        # Extract images from messages
        has_images = False
        image_inputs = []
        
        for msg in messages:
            if isinstance(msg.get("content"), list):
                has_images = True
                # Extract images if present
                for part in msg["content"]:
                    if part.get("type") == "image_url":
                        # Extract base64 and convert back to PIL
                        url = part["image_url"]["url"]
                        if url.startswith("data:image/png;base64,"):
                            b64_str = url.split(",")[1]
                            image_inputs.append(Image.open(BytesIO(base64.b64decode(b64_str))))
        
        # Validate vision input compatibility (now is_vl_model is correctly set)
        if has_images and not self.is_vl_model:
             raise ValueError("This model does not support vision inputs. Please use a Vision-Language model (e.g., Qwen-VL) or disconnect the image.")
        
        # Handle Seed
        if "seed" in kwargs:
            seed = int(kwargs["seed"])
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self._log(f"Applied seed: {seed}")

        # Prepare generation arguments
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
        }
        
        # Map and add optional parameters
        if "top_p" in kwargs:
            gen_kwargs["top_p"] = float(kwargs["top_p"])
        if "top_k" in kwargs:
            gen_kwargs["top_k"] = int(kwargs["top_k"])
        if "repeat_penalty" in kwargs:
            gen_kwargs["repetition_penalty"] = float(kwargs["repeat_penalty"])

        # Generate based on model type
        self._log("Generating response...", "INFO")
        
        with torch.no_grad():
            if self.is_vl_model:
                # VL Model Generation (Qwen-VL style)
                # Extract text prompt
                text_prompt = ""
                for msg in messages:
                    if isinstance(msg["content"], str):
                        text_prompt += msg["content"] + "\n"
                    elif isinstance(msg["content"], list):
                        for part in msg["content"]:
                            if part["type"] == "text":
                                text_prompt += part["text"] + "\n"
                
                # Prepare inputs using processor
                # Try to use processor's chat template if available for better formatting
                if hasattr(self.processor, 'apply_chat_template'):
                    try:
                        # Use the processor's built-in chat template
                        formatted_text = self.processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        self._log("Using processor's apply_chat_template", "INFO")
                    except Exception as e:
                        # Fallback to manual formatting if template fails
                        self._log(f"Chat template failed ({e}), using manual format", "WARNING")
                        formatted_text = f"User: {text_prompt}\nAssistant:"
                else:
                    # No chat template available, use manual formatting
                    formatted_text = f"User: {text_prompt}\nAssistant:"
                    self._log("No chat template available, using manual format", "INFO")
                
                # Process inputs with or without images
                if image_inputs:
                    inputs = self.processor(
                        text=[formatted_text],
                        images=image_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
                    self._log(f"Processed {len(image_inputs)} image(s) with text", "INFO")
                else:
                    inputs = self.processor(
                        text=[formatted_text],
                        return_tensors="pt"
                    )
                
                # Move inputs to device
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Decode
                if len(outputs.shape) == 2:
                    # Standard output: [batch_size, sequence_length]
                    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
                else:
                    # Fallback for unexpected shapes
                    generated_ids = outputs[0]
                response = self.processor.decode(generated_ids, skip_special_tokens=True)
                
            else:
                # Text-Only Model Generation
                # Format messages using chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt")
                
                # Move to model device
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
                
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
                # Decode
                input_len = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(
                    outputs[0][input_len:],
                    skip_special_tokens=True
                )
        
        self._log(f"Generated {len(response)} characters", "INFO")
        
        # Explicit cleanup if not keeping loaded
        if not keep_loaded:
            self._log("Unloading model (keep_loaded=False)...", "INFO")
            if self.model is not None:
                try:
                    self.model.cpu()
                except Exception:
                    pass
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            # Remove from class cache if present
            cache_key = self._get_cache_key()
            if cache_key in self._model_cache:
                del self._model_cache[cache_key]
            # Force memory release
            clear_gpu_memory()
        
        return response
    
    @classmethod
    def unload_model(cls, repo_id: str, quantization: str = "none", device: str = "auto") -> None:
        """Unload a specific model from cache."""
        cache_key = f"{repo_id}_{quantization}_{device}"
        if cache_key in cls._model_cache:
            model, tokenizer_or_processor = cls._model_cache.pop(cache_key)
            # Move model to CPU first to release CUDA references
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass  # Some models may fail on .cpu(), continue cleanup
                del model
            if tokenizer_or_processor is not None:
                del tokenizer_or_processor
            clear_gpu_memory()
            logger.info(f"Unloaded model: {cache_key}")
    
    @classmethod
    def unload_all_models(cls) -> None:
        """Unload all cached models."""
        # Explicitly unload each model to properly free GPU memory
        for cache_key in list(cls._model_cache.keys()):
            model, tokenizer_or_processor = cls._model_cache.pop(cache_key)
            if model is not None:
                try:
                    model.cpu()
                except Exception:
                    pass
                del model
            if tokenizer_or_processor is not None:
                del tokenizer_or_processor
        # Final cleanup
        clear_gpu_memory()
        logger.info("Unloaded all cached models")


# ============================================================================
# OUTPUT CLEANING UTILITIES - IMPROVED
# ============================================================================

def clean_llm_output(text: str, max_length: int = 0, debug_log: Optional[List[str]] = None) -> str:
    """
    Clean and normalize LLM output.
    
    Args:
        text: Raw LLM output
        max_length: Maximum output length (0 = no limit)
        debug_log: Optional list to append debug messages
    """
    if not text:
        raise ValueError("Empty text received for cleaning")
    
    original_len = len(text)
    
    # Remove thinking tags (Qwen3 thinking mode)
    # Handle both complete <think>...</think> blocks and orphaned </think> tags
    if "<think>" in text and "</think>" in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if debug_log:
            debug_log.append("Removed <think>...</think> block")
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) == 2 and parts[1].strip():
            text = parts[1].strip()
            if debug_log:
                debug_log.append(f"Removed thinking content before </think> tag ({len(parts[0])} chars)")
    
    # Handle "Thinking" models that output reasoning without tags
    thinking_indicators_zh = [
        "我是一个被关在逻辑牢笼里的幻视艺术家",
        "我的工作流程是",
        "首先，分析",
        "首先，我需要",
        "让我分析",
        "根据工作流程",
        "现在，用户输入的prompt是",
        "我需要把这个转化为",
        "检查是否有需要",
        "关键点：",
        "现在，构建描述",
        "开始草拟",
    ]
    thinking_indicators_en = [
        "I am a visionary artist",
        "My workflow is",
        "First, I need to analyze",
        "Let me analyze",
        "Following the workflow",
        "The user's input prompt is",
        "I need to transform this",
        "Checking if",
        "Key points:",
        "Now, constructing",
        "Let me draft",
    ]
    
    has_thinking = any(indicator in text for indicator in thinking_indicators_zh + thinking_indicators_en)
    
    if has_thinking:
        if debug_log:
            debug_log.append("Detected untagged thinking/reasoning content")
        
        # Strategy 1: Look for final prompt markers
        final_prompt_markers = [
            "最终prompt：", "最终的prompt：", "修改后的prompt：", 
            "增强后的prompt：", "优化后的prompt：",
            "最终描述：", "最终视觉描述：",
            "Final prompt:", "Enhanced prompt:", "Modified prompt:",
            "The final prompt:", "Here is the enhanced prompt:",
        ]
        
        best_start = -1
        for marker in final_prompt_markers:
            idx = text.rfind(marker)
            if idx != -1:
                potential_start = idx + len(marker)
                if potential_start > best_start:
                    best_start = potential_start
        
        if best_start > 0 and best_start < len(text) - 50:
            text = text[best_start:].strip()
            if debug_log:
                debug_log.append(f"Extracted final prompt after marker at position {best_start}")
        else:
            # Strategy 2: Extract English prompt at end (common pattern)
            # Look for "例如：" or "Example:" followed by English text
            example_match = re.search(r'(?:例如：|Example:\s*)([A-Z][a-zA-Z].*?)$', text, re.DOTALL)
            if example_match:
                text = example_match.group(1).strip()
                if debug_log:
                    debug_log.append("Extracted English prompt after example marker")
            else:
                # Strategy 3: Find standalone English prompt in latter half
                # Pattern for typical image prompts
                english_prompt_pattern = r'\n([A-Z][a-z][^。\n]*(?:photo|image|portrait|scene|woman|man|person|wearing|standing|sitting)[^。]*\.?)$'
                match = re.search(english_prompt_pattern, text)
                if match:
                    text = match.group(1).strip()
                    if debug_log:
                        debug_log.append("Extracted trailing English prompt")
                else:
                    # Strategy 4: Last resort - find content after "核心画面" or similar
                    fallback_markers = ["核心画面：", "首先，核心画面：", "添加细节："]
                    for marker in fallback_markers:
                        idx = text.rfind(marker)
                        if idx != -1 and idx > len(text) * 0.5:
                            # Take content from this point but clean it up
                            text = text[idx + len(marker):].strip()
                            if debug_log:
                                debug_log.append(f"Extracted content after '{marker}'")
                            break
    
    # Remove markdown code blocks
    if text.startswith('```'):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()
    
    # Remove common prefixes
    prefixes = [
        "Here is the enhanced prompt:", "Here's the enhanced prompt:",
        "Enhanced prompt:", "Final prompt:", "Output:",
        "The enhanced prompt:", "修改后的prompt：", "最终prompt：",
        "Enhanced visual description:", "优化后的视觉描述：",
        "Here is", "Here's"
    ]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    
    # Remove surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    
    # Remove negative instruction phrases
    negative_pattern = r',?\s*no\s+"[^"]{1,50}"(?:\s+or\s+"[^"]{1,50}")?\s+(?:tags?|descriptors?|effects?|elements?|overlays?|filters?|presence)'
    neg_matches = list(re.finditer(negative_pattern, text, re.IGNORECASE))
    if len(neg_matches) > 3:
        if debug_log:
            debug_log.append(f"Removing {len(neg_matches)} negative instruction phrases")
        text = re.sub(negative_pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r',\s*$', '.', text)
        text = re.sub(r',\s*\.', '.', text)
    
    # Detect phrase-level repetition loops
    segments = [s.strip() for s in text.split(',') if s.strip()]
    if len(segments) > 10:
        for pattern_len in range(2, 7):
            if len(segments) >= pattern_len * 3:
                for start in range(len(segments) - pattern_len * 2):
                    pattern = segments[start:start + pattern_len]
                    repeat_count = 1
                    pos = start + pattern_len
                    while pos + pattern_len <= len(segments):
                        if segments[pos:pos + pattern_len] == pattern:
                            repeat_count += 1
                            pos += pattern_len
                        else:
                            break
                    
                    if repeat_count >= 3:
                        if debug_log:
                            debug_log.append(f"Detected phrase repetition loop: {pattern_len} phrases repeated {repeat_count} times")
                        segments = segments[:start + pattern_len]
                        text = ', '.join(segments)
                        if text and text[-1] not in '.!?':
                            text += '.'
                        break
                else:
                    continue
                break
    
    # Remove trailing quoted keyword lists
    keyword_list_pattern = r',\s*"[^"]{1,80}"\s+\w+(?:,\s*"[^"]{1,80}"\s+\w+){2,}\s*$'
    match = re.search(keyword_list_pattern, text)
    if match:
        if debug_log:
            debug_log.append(f"Removed quoted keyword list at position {match.start()}")
        text = text[:match.start()].strip()
        if text and text[-1] not in '.!?':
            text += '.'
    elif re.search(r'(\s*"[^"]{1,50}"\s*){3,}$', text):
        match = re.search(r'(\s*"[^"]{1,50}"\s*){3,}$', text)
        if debug_log:
            debug_log.append(f"Removed trailing quoted keywords")
        text = text[:match.start()].strip()
        if text and text[-1] not in '.!?':
            text += '.'
    
    # Fix exact character repetition loops
    repeat_pattern = r'(.{10,60}?)\1{2,}'
    match = re.search(repeat_pattern, text)
    if match:
        if debug_log:
            debug_log.append(f"Fixed exact repetition loop at position {match.start()}")
        text = text[:match.start() + len(match.group(1))]
        last_period = text.rfind('.')
        if last_period > match.start() - 50:
            text = text[:last_period + 1]
    
    # Apply max length if specified
    if max_length > 0 and len(text) > max_length:
        if debug_log:
            debug_log.append(f"Output too long ({len(text)} chars), truncating to {max_length}")
        text = text[:max_length]
        last_period = text.rfind('.')
        if last_period > max_length * 0.7:
            text = text[:last_period + 1]
        else:
            last_comma = text.rfind(',')
            if last_comma > max_length * 0.85:
                text = text[:last_comma] + '.'
    
    # Clean up extra whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    # Final cleanup: remove trailing incomplete phrases
    if text and text[-1] not in '.!?"\'':
        last_period = text.rfind('.')
        last_comma = text.rfind(',')
        if last_period > len(text) * 0.8:
            text = text[:last_period + 1]
        elif last_comma > len(text) * 0.9:
            text = text[:last_comma] + '.'
    
    if debug_log:
        debug_log.append(f"Cleaned: {original_len} -> {len(text)} chars")
    
    return text.strip()


def sanitize_utf8(text: str) -> str:
    """Sanitize text to ASCII-safe characters (following EBU-LMStudio pattern)."""
    try:
        from unicodedata import normalize
        return normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    except Exception:
        return text.encode('ascii', 'replace').decode('ascii')


def detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English."""
    if not text:
        return "en"
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_alpha = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))
    if total_alpha == 0:
        return "en"
    return "zh" if chinese_chars / total_alpha > 0.3 else "en"


# ============================================================================
# OPTIONS NODE (Following comfyui-ollama pattern)
# ============================================================================

class Z_ImageOptions:
    """
    Advanced inference options with enable flags.
    Each option can be individually enabled/disabled.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_temperature": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": TOOLTIPS["temperature"]
                }),
                
                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Nucleus sampling cutoff. Lower = more focused."
                }),
                
                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {
                    "default": 40, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Top-K sampling. Higher = more diverse."
                }),
                
                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": seed, "min": 0, "max": 2 ** 31, "step": 1,
                    "tooltip": TOOLTIPS["seed"]
                }),
                
                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Penalty for repeated tokens. >1 reduces repetition."
                }),
                
                "enable_max_tokens": ("BOOLEAN", {"default": True}),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 256,
                    "tooltip": TOOLTIPS["max_tokens"]
                }),
                
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS["debug_mode"]
                }),
            }
        }
    
    RETURN_TYPES = ("ZIMAGE_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create_options"
    CATEGORY = "Z-Image"
    DESCRIPTION = "Configure advanced inference options. Enable only the options you need."
    
    def create_options(self, **kwargs) -> Tuple[Dict[str, Any]]:
        """Create options dictionary with only enabled values."""
        return (kwargs,)


def filter_enabled_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract only enabled options from options dict (following comfyui-ollama pattern)."""
    if not options:
        return {}
    
    enablers = [
        "enable_temperature", "enable_top_p", "enable_top_k",
        "enable_seed", "enable_repeat_penalty", "enable_max_tokens"
    ]
    
    result = {}
    for enabler in enablers:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            if key in options:
                result[key] = options[key]
    
    # Always include debug_mode if present
    if "debug_mode" in options:
        result["debug_mode"] = options["debug_mode"]
    
    return result


# ============================================================================
# API CONFIG NODE
# ============================================================================

class Z_ImageAPIConfig:
    """
    Unified API configuration node supporting multiple providers.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": ([p.value for p in Provider], {
                    "default": Provider.OPENROUTER.value,
                    "tooltip": TOOLTIPS["provider"]
                }),
                "model": ("STRING", {
                    "default": "qwen/qwen3-235b-a22b:free",
                    "multiline": False,
                    "placeholder": "Model name/ID or HuggingFace repo",
                    "tooltip": TOOLTIPS["model"]
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "sk-or-v1-xxxxx (OpenRouter only)",
                    "tooltip": TOOLTIPS["api_key"]
                }),
                "local_endpoint": ("STRING", {
                    "default": "http://localhost:11434/v1",
                    "multiline": False,
                    "tooltip": TOOLTIPS["local_endpoint"]
                }),
                "llm_path": ("STRING", {
                    "default": "",
                    "placeholder": "C:/path/to/LLM/models (optional, for Direct provider)",
                    "tooltip": "Custom path to local LLM models. If set, model name is treated as folder name. If empty, downloads from HuggingFace."
                }),
                "quantization": (Quantization.get_values(), {
                    "default": Quantization.Q4.value,
                    "tooltip": TOOLTIPS["quantization"]
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Device for direct model loading"
                }),
            }
        }
    
    RETURN_TYPES = ("ZIMAGE_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "Z-Image"
    DESCRIPTION = "Configure LLM API connection. Supports OpenRouter, local servers, and direct HuggingFace model loading."
    
    def configure(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        local_endpoint: str = "http://localhost:11434/v1",
        llm_path: str = "",
        quantization: str = "4bit",
        device: str = "auto"
    ) -> Tuple[Dict[str, Any]]:
        """Create API configuration."""
        
        clean_model = model.strip()
        if not clean_model:
            raise ValueError("Model ID cannot be empty!")
        
        provider_enum = Provider(provider)
        
        config = {
            "provider": provider_enum.value,
            "model": clean_model,
        }
        
        if provider_enum == Provider.OPENROUTER:
            if not api_key.strip():
                logger.warning("No API key provided for OpenRouter!")
            config["client"] = OpenRouterClient(api_key=api_key.strip())
            logger.info(f"Configured OpenRouter: {clean_model}")
            
        elif provider_enum == Provider.LOCAL:
            clean_endpoint = local_endpoint.strip()
            if not clean_endpoint:
                raise ValueError("Local endpoint cannot be empty!")
            config["endpoint"] = clean_endpoint
            config["client"] = LocalLLMClient(endpoint=clean_endpoint)
            logger.info(f"Configured Local LLM: {clean_model} at {clean_endpoint}")
            
        elif provider_enum == Provider.DIRECT:
            if not HAS_TRANSFORMERS:
                raise RuntimeError(
                    "Direct model loading requires: pip install transformers torch accelerate bitsandbytes"
                )
            config["quantization"] = quantization
            config["device"] = device
            config["llm_path"] = llm_path.strip()
            config["client"] = DirectLocalModelClient(
                repo_id=clean_model,
                quantization=quantization,
                device=device,
                llm_path=llm_path.strip()
            )
            if llm_path.strip():
                logger.info(f"Configured Direct Model: {clean_model} from {llm_path}")
            else:
                logger.info(f"Configured Direct Model: {clean_model} ({quantization})")
        
        return (config,)


# ============================================================================
# MAIN PROMPT ENHANCER NODE
# ============================================================================

class Z_ImagePromptEnhancer:
    """
    Z-Image Prompt Enhancer - Transform prompts into detailed visual descriptions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("ZIMAGE_CONFIG",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt to enhance..."
                }),
                "prompt_template": (["auto", "chinese", "english", "custom"], {
                    "default": "chinese",
                    "tooltip": TOOLTIPS["prompt_template"]
                }),
            },
            "optional": {
                "options": ("ZIMAGE_OPTIONS",),
                "image": ("IMAGE", {
                    "tooltip": TOOLTIPS["image"]
                }),
                "retry_count": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": TOOLTIPS["retry_count"]
                }),
                "max_output_length": ("INT", {
                    "default": 6000,
                    "min": 0,
                    "max": 10000,
                    "step": 100,
                    "tooltip": TOOLTIPS["max_output_length"]
                }),
                "session_id": ("STRING", {
                    "default": "",
                    "tooltip": TOOLTIPS["session_id"]
                }),
                "reset_session": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS["reset_session"]
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": TOOLTIPS["keep_model_loaded"]
                }),
                "utf8_sanitize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS["utf8_sanitize"]
                }),
                "custom_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter a custom system prompt and select 'custom' in the prompt template."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "debug_log")
    FUNCTION = "enhance"
    CATEGORY = "Z-Image"
    DESCRIPTION = "Enhance prompts using LLM for text-to-image generation. Supports multi-turn conversations and image input."
    
    def enhance(
        self,
        config: Dict[str, Any],
        prompt: str,
        prompt_template: str,
        custom_system_prompt: str,
        unique_id: str = "",
        options: Optional[Dict[str, Any]] = None,
        image: Optional["torch.Tensor"] = None,
        retry_count: int = 3,
        max_output_length: int = 6000,
        session_id: str = "",
        reset_session: bool = False,
        keep_model_loaded: bool = True,
        utf8_sanitize: bool = False,
    ) -> Tuple[str, str]:
        """Enhance prompt using configured LLM."""
        
        debug_lines = []
        debug_lines.append("=" * 60)
        debug_lines.append(f"Z-IMAGE PROMPT ENHANCER")
        debug_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        debug_lines.append("=" * 60)
        
        try:
            return self._enhance_internal(
                config=config,
                prompt=prompt,
                prompt_template=prompt_template,
                unique_id=unique_id,
                options=options,
                image=image,
                retry_count=retry_count,
                max_output_length=max_output_length,
                session_id=session_id,
                reset_session=reset_session,
                keep_model_loaded=keep_model_loaded,
                utf8_sanitize=utf8_sanitize,
                debug_lines=debug_lines,
                custom_system_prompt=custom_system_prompt
            )
        except Exception as e:
            error_msg = f"\nERROR: {type(e).__name__}: {str(e)}"
            debug_lines.append(error_msg)
            logger.error(error_msg)
            raise
    
    def _enhance_internal(
        self,
        config: Dict[str, Any],
        prompt: str,
        prompt_template: str,
        unique_id: str,
        options: Optional[Dict[str, Any]],
        image: Optional["torch.Tensor"],
        retry_count: int,
        max_output_length: int,
        session_id: str,
        reset_session: bool,
        keep_model_loaded: bool,
        utf8_sanitize: bool,
        debug_lines: List[str],
        custom_system_prompt: str
    ) -> Tuple[str, str]:
        """Internal enhancement logic."""
        
        if not prompt.strip():
            debug_lines.append("Empty input prompt")
            return ("", "\n".join(debug_lines))
        
        # Validate custom system prompt if selected
        if prompt_template == "custom":
            if not custom_system_prompt.strip():
                raise ValueError("Custom system prompt cannot be empty when 'custom' template is selected")
            if "{prompt}" not in custom_system_prompt:
                raise ValueError("Custom system prompt must contain {prompt} placeholder")
        
        # Extract enabled options
        opts = filter_enabled_options(options) if options else {}
        debug_mode = opts.get("debug_mode", False)
        
        # Determine prompt template language
        if prompt_template == "auto":
            lang = detect_language(prompt)
        elif prompt_template == "chinese":
            lang = "zh"
        elif prompt_template == "english":
            lang = "en"
        else:
            lang = "custom"
        
        debug_lines.append(f"\n[CONFIGURATION]")
        debug_lines.append(f"Provider: {config['provider']}")
        debug_lines.append(f"Model: {config['model']}")
        debug_lines.append(f"Prompt Template: {lang}")
        debug_lines.append(f"Max Output Length: {max_output_length} (0=unlimited)")
        debug_lines.append(f"Options: {opts}")
        
        # Session management - FIXED: Now shows proper session ID
        effective_session_id = session_id if session_id.strip() else unique_id
        
        if reset_session:
            cleared = clear_session(effective_session_id)
            debug_lines.append(f"Session reset requested for '{effective_session_id}': {'cleared' if cleared else 'not found'}")
        
        session, is_new = get_or_create_session(effective_session_id, config['model'])
        debug_lines.append(f"Session '{effective_session_id}': {'new' if is_new else 'existing'} ({len(session.messages)} messages)")
        
        # Build prompt with template
        template = PROMPT_TEMPLATE_ZH if lang == "zh" else PROMPT_TEMPLATE_EN if lang == "en" else custom_system_prompt
        system_prompt = template.format(prompt=prompt)
        
        debug_lines.append(f"\n[INPUT]")
        debug_lines.append(f"User prompt (length: {len(prompt)} chars):")
        debug_lines.append(f"{prompt}")
        
        debug_lines.append(f"\n[SYSTEM INSTRUCTION SENT TO API]")
        debug_lines.append(f"Length: {len(system_prompt)} chars")
        debug_lines.append(f"Content:\n{system_prompt}")
        
        # Initialize messages with system instruction first
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        
        # Add session history if exists (for multi-turn)
        if session.messages and not is_new:
            messages.extend(session.get_messages())
            debug_lines.append(f"Added {len(session.messages)} messages from session history")
        
        # Add minimal user trigger message
        messages.append({"role": "user", "content": " "})
        
        # Handle vision models if image provided
        if image is not None and HAS_PIL:
            debug_lines.append(f"\n[VISION]")
            debug_lines.append("Processing image input for vision model...")
            try:
                images_b64 = batch_tensors_to_base64(image)
                if images_b64:
                    debug_lines.append(f"Encoded {len(images_b64)} image(s) to base64")
                    content_parts = [{"type": "text", "text": " "}]
                    for idx, img_b64 in enumerate(images_b64):
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })
                        debug_lines.append(f"Added image {idx+1}/{len(images_b64)}")
                    # Replace the last message (user message) with multimodal content
                    messages[-1] = {"role": "user", "content": content_parts}
            except Exception as e:
                debug_lines.append(f"Image encoding failed: {e}")
                logger.warning(f"Failed to encode images: {e}")
        
        # API parameters
        temperature = opts.get("temperature", 0.7)
        max_tokens = opts.get("max_tokens")  # Will be None if disabled
        
        debug_lines.append(f"\n[INFERENCE]")
        debug_lines.append(f"Temperature: {temperature}")
        debug_lines.append(f"Max Tokens: {max_tokens if max_tokens is not None else 'Not set (using API default)'}")
        
        # Make API call
        client = config["client"]
        # Build API call parameters
        api_params = {
            "messages": messages,
            "model": config["model"],
            "temperature": temperature,
            "retry_count": retry_count,
            "keep_loaded": keep_model_loaded,
        }
        
        # Only include max_tokens if it was enabled
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        
        # Add other enabled options
        api_params.update({k: v for k, v in opts.items() if k not in ["temperature", "max_tokens", "debug_mode"]})
        
        response = client.chat(**api_params)
        
        # Add client log to debug output
        debug_lines.append(f"\n[CLIENT LOG]")
        if hasattr(client, 'debug_log'):
            debug_lines.extend(client.debug_log)
        
        # Handle empty response
        if not response or not response.strip():
            raise ValueError("API returned empty response")
        
        debug_lines.append(f"\n[RAW RESPONSE]")
        debug_lines.append(f"Length: {len(response)} chars")
        debug_lines.append(f"Full content:\n{response}")
        
        # Clean output - NOW USES max_output_length parameter
        debug_lines.append(f"\n[CLEANING]")
        enhanced = clean_llm_output(response, max_length=max_output_length, debug_log=debug_lines)
        
        if not enhanced:
            raise ValueError("Cleaning resulted in empty output")
        
        # UTF-8 sanitization
        if utf8_sanitize:
            enhanced = sanitize_utf8(enhanced)
            debug_lines.append("UTF-8 sanitization applied")
        
        debug_lines.append(f"\n[OUTPUT]")
        debug_lines.append(f"Final length: {len(enhanced)} characters")
        debug_lines.append(f"Full enhanced prompt:\n{enhanced}")
        
        # Token estimation - language-aware
        if lang == "zh":
            # Chinese: roughly 1.5-2 characters per token
            estimated_tokens = len(enhanced) / 1.5
            estimated_words = len(enhanced)  # Each character roughly a "word"
        else:
            # English: ~4 chars per token on average
            estimated_tokens = len(enhanced) / 4
            estimated_words = len(enhanced) / 5
            
        debug_lines.append(f"\n[TOKEN ESTIMATE]")
        debug_lines.append(f"Estimated words: ~{int(estimated_words)}")
        debug_lines.append(f"Estimated tokens: ~{int(estimated_tokens)} (Z-Image in ComfyUI supports unlimited tokens)")
        
        # Only warn if unreasonably long (e.g. > 8000 tokens which might hit other limits)
        if estimated_tokens > 8000:
             debug_lines.append(f"WARNING: Extremely long prompt (>8000 tokens). This may impact generation quality.")
        
        logger.info(f"Enhancement successful. Length: {len(enhanced)}")
        
        # Save conversation to session for multi-turn continuity
        # Store original prompt (not system instruction) for cleaner history
        session.add_message("user", prompt)
        session.add_message("assistant", enhanced)
        debug_lines.append(f"\n[SESSION]")
        debug_lines.append(f"Saved conversation to session '{effective_session_id}'")
        debug_lines.append(f"Total messages in session: {len(session.messages)}")
        
        return (enhanced, "\n".join(debug_lines))


# ============================================================================
# PROMPT ENHANCER WITH CLIP OUTPUT (Enhanced with Omni Support)
# ============================================================================

def format_omni_prompt(prompt: str, num_condition_images: int = 0) -> str:
    """
    Format prompt for Z-Image-Omni pipeline with vision tokens.
    
    This generates the appropriate prompt format expected by ZImageOmniPipeline:
    - For text-only: <|im_start|>user\n[prompt]<|im_end|>\n<|im_start|>assistant\n
    - For image+text: <|im_start|>user\n<|vision_start|><|vision_end|>...[prompt]<|im_end|>\n<|im_start|>assistant\n<|vision_start|>
    
    Args:
        prompt: The enhanced text prompt
        num_condition_images: Number of condition images (0 for text-to-image)
    
    Returns:
        Formatted prompt string with vision tokens
    """
    if num_condition_images == 0:
        # Text-to-image mode
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Image editing/conditioning mode
        # Build vision token sequence for each condition image
        parts = ["<|im_start|>user\n<|vision_start|>"]
        parts.extend(["<|vision_end|><|vision_start|>"] * (num_condition_images - 1))
        parts.append(f"<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n<|vision_start|>")
        parts.append("<|vision_end|><|im_end|>")
        return "".join(parts)


class Z_ImagePromptEnhancerWithCLIP:
    """
    Prompt Enhancer with CLIP conditioning output and Omni pipeline support.
    
    Features:
    - CLIP conditioning output for direct use in ComfyUI
    - Multiple condition image support (up to 4 images) for Z-Image-Omni
    - Automatic vision token formatting for Omni pipeline
    - Backward compatible with single image input
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "config": ("ZIMAGE_CONFIG",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt to enhance..."
                }),
                "prompt_template": (["auto", "chinese", "english", "custom"], {
                    "default": "chinese",
                    "tooltip": TOOLTIPS["prompt_template"]
                }),
                "generation_mode": (["text_to_image", "image_to_image"], {
                    "default": "text_to_image",
                    "tooltip": "Generation mode: text_to_image (T2I) or image_to_image (I2I/Omni editing)"
                }),
            },
            "optional": {
                "options": ("ZIMAGE_OPTIONS",),
                # Multi-image input for vision enhancement and Omni
                "image_1": ("IMAGE", {
                    "tooltip": "First image for vision enhancement and Z-Image-Omni editing"
                }),
                "image_2": ("IMAGE", {
                    "tooltip": "Second image (optional)"
                }),
                "image_3": ("IMAGE", {
                    "tooltip": "Third image (optional)"
                }),
                "image_4": ("IMAGE", {
                    "tooltip": "Fourth image (optional)"
                }),

                "retry_count": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": TOOLTIPS["retry_count"]
                }),
                "max_output_length": ("INT", {
                    "default": 6000,
                    "min": 0,
                    "max": 10000,
                    "step": 100,
                    "tooltip": TOOLTIPS["max_output_length"]
                }),
                "session_id": ("STRING", {
                    "default": "",
                    "tooltip": TOOLTIPS["session_id"]
                }),
                "reset_session": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS["reset_session"]
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": TOOLTIPS["keep_model_loaded"]
                }),
                "utf8_sanitize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS["utf8_sanitize"]
                }),
                "custom_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter a custom system prompt and select 'custom' in the prompt template."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("conditioning", "enhanced_prompt", "omni_formatted_prompt", "debug_log", "num_condition_images")
    FUNCTION = "enhance_and_encode"
    CATEGORY = "Z-Image"
    DESCRIPTION = "Enhance prompt and encode with CLIP. Supports Z-Image-Omni multi-image conditioning."
    
    def enhance_and_encode(
        self,
        clip,
        config: Dict[str, Any],
        prompt: str,
        prompt_template: str,
        generation_mode: str,
        unique_id: str = "",
        options: Optional[Dict[str, Any]] = None,

        image_1: Optional["torch.Tensor"] = None,
        image_2: Optional["torch.Tensor"] = None,
        image_3: Optional["torch.Tensor"] = None,
        image_4: Optional["torch.Tensor"] = None,

        retry_count: int = 3,
        max_output_length: int = 6000,
        session_id: str = "",
        reset_session: bool = False,
        keep_model_loaded: bool = True,
        utf8_sanitize: bool = False,
        custom_system_prompt: str = "",
    ):
        """Enhance prompt and encode with CLIP. Supports Omni multi-image conditioning."""
        
        debug_lines = ["[Z-IMAGE PROMPT ENHANCER + CLIP]", "=" * 50]
        
        # Collect input images
        input_images = []
        for idx, img in enumerate([image_1, image_2, image_3, image_4], 1):
            if img is not None:
                input_images.append(img)
                debug_lines.append(f"Image {idx}: shape={img.shape}")
        
        num_condition_images = len(input_images)
        
        # Determine mode based on user selection and images
        debug_lines.append(f"Generation mode: {generation_mode}")
        
        if generation_mode == "image_to_image" and input_images:
            # Image-to-Image / Omni mode - use all attached images
            vision_images = torch.cat(input_images, dim=0)
            debug_lines.append(f"Total images: {num_condition_images}")
            debug_lines.append(f"Vision batch shape: {vision_images.shape}")
            if num_condition_images > 1:
                debug_lines.append(f"Mode: Image Editing (Omni) - All {num_condition_images} images sent to VL")
            else:
                debug_lines.append(f"Mode: Image-to-Image")
        elif generation_mode == "image_to_image" and not input_images:
            # I2I mode but no images - warn and fallback to T2I
            debug_lines.append("WARNING: image_to_image mode selected but no images attached. Falling back to text_to_image.")
            vision_images = None
            debug_lines.append(f"Mode: Text-to-Image (fallback)")
        else:
            # Text-to-Image mode - ignore any attached images for VL
            vision_images = None
            debug_lines.append(f"Mode: Text-to-Image")
        
        debug_lines.append("")
        
        # Enhance the prompt using base enhancer
        enhancer = Z_ImagePromptEnhancer()
        enhanced_prompt, enhance_debug = enhancer.enhance(
            config=config,
            prompt=prompt,
            prompt_template=prompt_template,
            unique_id=unique_id,
            options=options,
            image=vision_images,
            retry_count=retry_count,
            max_output_length=max_output_length,
            session_id=session_id,
            reset_session=reset_session,
            keep_model_loaded=keep_model_loaded,
            utf8_sanitize=utf8_sanitize,
            custom_system_prompt=custom_system_prompt,
        )
        
        debug_lines.append(enhance_debug)
        
        # Format for Omni pipeline with vision tokens (always enabled)
        debug_lines.append("")
        debug_lines.append("[OMNI FORMATTING]")
        debug_lines.append("=" * 50)
        omni_formatted = format_omni_prompt(enhanced_prompt, num_condition_images)
        debug_lines.append(f"Formatted with {num_condition_images} vision token(s)")
        debug_lines.append(f"Omni prompt length: {len(omni_formatted)} characters")
        if num_condition_images == 0:
            debug_lines.append("  - Text-only mode (standard vision tokens)")
        else:
            debug_lines.append(f"  - {num_condition_images}x <|vision_start|>...<|vision_end|> for condition images")
            debug_lines.append("  - 1x <|vision_start|>...<|vision_end|> for generated output")
        
        # Noise mask info
        debug_lines.append("")
        debug_lines.append("[NOISE MASK INFO]")
        debug_lines.append("=" * 50)
        if num_condition_images > 0:
            noise_mask_preview = [0] * num_condition_images + [1]
            debug_lines.append(f"Expected noise mask: {noise_mask_preview}")
            debug_lines.append(f"  - {num_condition_images}x 0 (clean/condition images)")
            debug_lines.append("  - 1x 1 (noisy/target image)")
        else:
            debug_lines.append("No noise mask needed for text-to-image generation")
        
        # Encode with CLIP
        tokens = clip.tokenize(enhanced_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]
        
        debug_lines.append("")
        debug_lines.append("[CLIP ENCODING]")
        debug_lines.append(f"Conditioning shape: {cond.shape}")
        
        return (
            conditioning,
            enhanced_prompt,
            omni_formatted,
            "\n".join(debug_lines),
            num_condition_images
        )

# ============================================================================
# INTEGRATED KSAMPLER WITH PROMPT ENHANCEMENT
# ============================================================================

class Z_ImageIntegratedKSampler:
    """
    Z-Image Integrated KSampler - All-in-one image generation with prompt enhancement.
    
    Features:
    - Integrated prompt enhancement using LLM
    - Full KSampler with model, VAE, and sampling
    - Text-to-Image and Image-to-Image modes
    - Multi-image conditioning (up to 4 images)
    - AuraFlow shift optimization
    - CFGNorm integration
    - Batch generation
    - Auto-save to folder
    - GPU/RAM memory cleanup
    """
    
    def __init__(self):
        self.device = comfy.model_management.intermediate_device() if HAS_COMFYUI else "cpu"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model for image generation"}),
                "clip": ("CLIP", {"tooltip": "CLIP model for text encoding"}),
                "vae": ("VAE", {"tooltip": "VAE model for latent encode/decode"}),
                "config": ("ZIMAGE_CONFIG", {"tooltip": "LLM configuration for prompt enhancement"}),
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your positive prompt...",
                    "tooltip": "Positive prompt - will be enhanced by LLM"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your negative prompt...",
                    "tooltip": "Negative prompt for unwanted elements"
                }),
                "generation_mode": (["text_to_image", "image_to_image"], {
                    "default": "text_to_image",
                    "tooltip": "T2I: Generate from text. I2I: Edit/transform input images"
                }),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "Output image width"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "Output image height"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000, "tooltip": "Sampling steps"}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "CFG scale"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Sampler algorithm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "Scheduler type"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength (1.0 = full, <1.0 for I2I)"}),
            },
            "optional": {
                "options": ("ZIMAGE_OPTIONS", {"tooltip": "Advanced LLM options"}),
                "image_1": ("IMAGE", {"tooltip": "First image for I2I/vision enhancement"}),
                "image_2": ("IMAGE", {"tooltip": "Second image (optional)"}),
                "image_3": ("IMAGE", {"tooltip": "Third image (optional)"}),
                "image_4": ("IMAGE", {"tooltip": "Fourth image (optional)"}),
                "latent": ("LATENT", {"tooltip": "Optional latent input (for custom latent)"}),
                "prompt_template": (["auto", "chinese", "english", "custom"], {
                    "default": "chinese",
                    "tooltip": "LLM prompt template language"
                }),
                "enable_prompt_enhance": ("BOOLEAN", {"default": True, "tooltip": "Enable LLM prompt enhancement (disable to use prompt as-is)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip": "Number of images to generate"}),
                "auraflow_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "AuraFlow shift (0 = disabled)"}),
                "cfg_norm_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "CFGNorm strength (0 = disabled)"}),
                "enable_clean_gpu": ("BOOLEAN", {"default": False, "tooltip": "Clean GPU memory before/after sampling"}),
                "enable_clean_ram": ("BOOLEAN", {"default": False, "tooltip": "Clean RAM after completion"}),
                "auto_save_folder": ("STRING", {"default": "", "tooltip": "Auto-save folder (empty = disabled)"}),
                "output_prefix": ("STRING", {"default": "z_image", "tooltip": "Filename prefix for auto-save"}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Custom system prompt (use with 'custom' template)"}),
                "instruction": ("STRING", {
                    "multiline": True, 
                    "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
                    "placeholder": "VLM System Instruction (for Image-to-Image)",
                    "tooltip": "System instruction for the Vision-Language Model in I2I mode"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("images", "latent", "enhanced_prompt", "debug_log")
    FUNCTION = "generate"
    CATEGORY = "Z-Image"
    DESCRIPTION = "Z-Image Integrated KSampler - All-in-one generation with LLM prompt enhancement, sampling, AuraFlow, CFGNorm, and auto-save."
    
    def generate(
        self,
        model,
        clip,
        vae,
        config: Dict[str, Any],
        positive_prompt: str,
        negative_prompt: str,
        generation_mode: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float = 1.0,
        unique_id: str = "",
        options: Optional[Dict[str, Any]] = None,
        image_1: Optional["torch.Tensor"] = None,
        image_2: Optional["torch.Tensor"] = None,
        image_3: Optional["torch.Tensor"] = None,
        image_4: Optional["torch.Tensor"] = None,
        latent: Optional[Dict] = None,
        prompt_template: str = "chinese",
        enable_prompt_enhance: bool = True,
        batch_size: int = 1,
        auraflow_shift: float = 0.0,
        cfg_norm_strength: float = 0.0,
        enable_clean_gpu: bool = False,
        enable_clean_ram: bool = False,
        auto_save_folder: str = "",
        output_prefix: str = "z_image",
        custom_system_prompt: str = "",
        instruction: str = "",
    ):
        """Generate images with integrated prompt enhancement and sampling."""
        
        debug_lines = ["[Z-IMAGE INTEGRATED KSAMPLER]", "=" * 60]
        debug_lines.append(f"Generation mode: {generation_mode}")
        debug_lines.append(f"Resolution: {width}x{height}")
        debug_lines.append(f"Seed: {seed}, Steps: {steps}, CFG: {cfg}")
        debug_lines.append(f"Sampler: {sampler_name}, Scheduler: {scheduler}")
        debug_lines.append(f"Batch size: {batch_size}")
        
        # Collect input images
        input_images = []
        for idx, img in enumerate([image_1, image_2, image_3, image_4], 1):
            if img is not None:
                input_images.append(img)
                debug_lines.append(f"Image {idx}: shape={img.shape}")
        
        # Apply AuraFlow shift if specified
        if auraflow_shift > 0:
            debug_lines.append(f"Applying AuraFlow shift: {auraflow_shift}")
            m = model.clone()
            import comfy.model_sampling
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass
            model_sampling = ModelSamplingAdvanced(m.model.model_config)
            model_sampling.set_parameters(shift=auraflow_shift, multiplier=1.0)
            m.add_object_patch("model_sampling", model_sampling)
            model = m
            debug_lines.append("AuraFlow shift applied")
        
        # Apply CFGNorm if specified
        if cfg_norm_strength > 0:
            debug_lines.append(f"Applying CFGNorm strength: {cfg_norm_strength}")
            m = model.clone()
            strength = cfg_norm_strength
            def cfg_norm(args):
                cond_p = args['cond_denoised']
                pred_text_ = args["denoised"]
                norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
                norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
                scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
                return pred_text_ * scale * strength
            m.set_model_sampler_post_cfg_function(cfg_norm)
            model = m
            debug_lines.append("CFGNorm applied")
        
        # Enhance positive prompt using LLM (if enabled)
        debug_lines.append("")
        debug_lines.append("[PROMPT ENHANCEMENT]")
        debug_lines.append("=" * 40)
        
        if enable_prompt_enhance:
            enhancer = Z_ImagePromptEnhancer()
            
            # Prepare vision images for LLM if in I2I mode
            vision_images = None
            if generation_mode == "image_to_image" and input_images:
                vision_images = torch.cat(input_images, dim=0)
                debug_lines.append(f"Sending {len(input_images)} images to LLM for enhancement")
            
            enhanced_prompt, enhance_debug = enhancer.enhance(
                config=config,
                prompt=positive_prompt,
                prompt_template=prompt_template,
                unique_id=unique_id,
                options=options,
                image=vision_images,
                retry_count=3,
                max_output_length=6000,
                session_id="",
                reset_session=True,
                keep_model_loaded=True,
                utf8_sanitize=False,
                custom_system_prompt=custom_system_prompt,
            )
            
            debug_lines.append(enhance_debug)
            debug_lines.append(f"Enhanced prompt length: {len(enhanced_prompt)} chars")
        else:
            # Use prompt as-is
            enhanced_prompt = positive_prompt
            debug_lines.append("Prompt enhancement DISABLED - using prompt as-is")
            debug_lines.append(f"Prompt length: {len(enhanced_prompt)} chars")
        
        # Pre-clean GPU if requested
        if enable_clean_gpu:
            debug_lines.append("Pre-cleaning GPU memory...")
            cleanGPUUsedForce()
        
        # Encode prompts
        debug_lines.append("")
        debug_lines.append("[ENCODING]")
        
        if generation_mode == "image_to_image" and input_images:
            # Use Z-Image pipeline with reference latents
            image_prompt, images_vl, llama_template, ref_latents = get_z_image_prompt(vae, input_images, instruction)
            positive = z_prompt_encode(clip, enhanced_prompt, image_prompt, images_vl, llama_template, ref_latents)
            negative = z_prompt_encode(clip, negative_prompt, image_prompt, images_vl, llama_template, ref_latents)
            debug_lines.append(f"I2I encoding with {len(input_images)} reference images")
        else:
            # Standard encoding
            tokens_pos = clip.tokenize(enhanced_prompt)
            positive, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
            positive = [[positive, {"pooled_output": pooled_pos}]]
            
            tokens_neg = clip.tokenize(negative_prompt)
            negative, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            negative = [[negative, {"pooled_output": pooled_neg}]]
            debug_lines.append("T2I standard encoding")
        
        # Create or use latent
        if latent is None:
            if generation_mode == "image_to_image" and input_images:
                # Encode first image as starting latent
                scaled_img = image_scale_aspect_ratio(input_images[0], width, height) if width > 0 and height > 0 else input_images[0]
                samples = vae.encode(scaled_img[:,:,:,:3])
                debug_lines.append("Using encoded image as latent")
            else:
                # Empty latent for T2I
                samples = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
                debug_lines.append(f"Created empty latent: {batch_size}x4x{height//8}x{width//8}")
            
            if batch_size > 1 and samples.shape[0] == 1:
                samples = samples.repeat((batch_size,) + ((1,) * (samples.ndim - 1)))
            
            latent = {"samples": samples}
        
        # Sample
        debug_lines.append("")
        debug_lines.append("[SAMPLING]")
        debug_lines.append(f"Starting {steps} steps...")
        
        latent_output = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)
        
        debug_lines.append("Sampling complete")
        
        # Decode
        debug_lines.append("")
        debug_lines.append("[DECODING]")
        output_images = vae.decode(latent_output["samples"])
        if len(output_images.shape) == 5:
            output_images = output_images.reshape(-1, output_images.shape[-3], output_images.shape[-2], output_images.shape[-1])
        debug_lines.append(f"Output shape: {output_images.shape}")
        
        # Auto-save
        if auto_save_folder:
            try:
                if os.path.isabs(auto_save_folder):
                    full_output_folder = auto_save_folder
                else:
                    output_dir = folder_paths.get_output_directory()
                    full_output_folder = os.path.join(output_dir, auto_save_folder)
                
                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder, exist_ok=True)
                
                debug_lines.append(f"Auto-saving to: {full_output_folder}")
                
                for batch_number, image in enumerate(output_images):
                    img = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
                    file = f"{output_prefix}_{seed}_{batch_number:05}.png"
                    img.save(os.path.join(full_output_folder, file))
                
                debug_lines.append(f"Saved {len(output_images)} images")
            except Exception as e:
                debug_lines.append(f"Auto-save failed: {e}")
        
        # Post-clean GPU
        if enable_clean_gpu:
            debug_lines.append("Post-cleaning GPU memory...")
            cleanGPUUsedForce()
        
        # Clean RAM
        if enable_clean_ram:
            debug_lines.append("Cleaning RAM...")
            gc.collect()
        
        debug_lines.append("")
        debug_lines.append("[COMPLETE]")
        debug_lines.append(f"Generated {len(output_images)} images")
        
        return (output_images, latent_output, enhanced_prompt, "\n".join(debug_lines))


# ============================================================================
# MODEL MANAGEMENT NODES
# ============================================================================

class Z_ImageUnloadModels:
    """
    Unload cached models to free memory.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_all": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "passthrough": ("*",),  # Pass any input through
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Z-Image"
    DESCRIPTION = "Unload cached LLM models to free GPU/system memory."
    OUTPUT_NODE = True
    
    def unload(self, unload_all: bool = True, passthrough=None):
        """Unload models from cache."""
        if unload_all:
            DirectLocalModelClient.unload_all_models()
            status = "All models unloaded"
        else:
            status = "No action taken"
        
        logger.info(status)
        return (status,)


class Z_ImageClearSessions:
    """
    Clear conversation sessions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear_all": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "session_id": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear"
    CATEGORY = "Z-Image"
    DESCRIPTION = "Clear conversation history sessions."
    OUTPUT_NODE = True
    
    def clear(self, clear_all: bool = True, session_id: str = ""):
        """Clear sessions."""
        if clear_all:
            count = len(CHAT_SESSIONS)
            CHAT_SESSIONS.clear()
            status = f"All {count} sessions cleared"
        elif session_id:
            cleared = clear_session(session_id)
            status = f"Session '{session_id}' {'cleared' if cleared else 'not found'}"
        else:
            status = "No action taken"
        
        logger.info(status)
        return (status,)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "Z_ImageAPIConfig": Z_ImageAPIConfig,
    "Z_ImageOptions": Z_ImageOptions,
    "Z_ImagePromptEnhancer": Z_ImagePromptEnhancer,
    "Z_ImagePromptEnhancerWithCLIP": Z_ImagePromptEnhancerWithCLIP,
    "Z_ImageIntegratedKSampler": Z_ImageIntegratedKSampler,
    "Z_ImageUnloadModels": Z_ImageUnloadModels,
    "Z_ImageClearSessions": Z_ImageClearSessions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Z_ImageAPIConfig": "Z-Image API Config",
    "Z_ImageOptions": "Z-Image Options",
    "Z_ImagePromptEnhancer": "Z-Image Prompt Enhancer",
    "Z_ImagePromptEnhancerWithCLIP": "Z-Image Prompt Enhancer + CLIP",
    "Z_ImageIntegratedKSampler": "Z-Image Integrated KSampler",
    "Z_ImageUnloadModels": "Z-Image Unload Models",
    "Z_ImageClearSessions": "Z-Image Clear Sessions",
}