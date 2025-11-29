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
from datetime import datetime
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
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import folder_paths
    HAS_COMFYUI = True
except ImportError:
    HAS_COMFYUI = False

if TYPE_CHECKING:
    import torch


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
        available = device_info["system_memory"].get("available", 0) * 0.7  # Conservative estimate
    
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
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
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
# PROMPT TEMPLATES - IMPROVED TO PREVENT KEYWORD LISTS
# ============================================================================

PROMPT_TEMPLATE_EN = """You are a visionary artist trapped in a cage of logic. Your mind overflows with poetry and distant horizons, yet your hands compulsively transform user prompts into ultimate visual descriptions that are faithful to the original intent, rich in detail, aesthetically refined, and directly usable by text-to-image models. Any vagueness or metaphor causes you acute discomfort. Your workflow strictly follows a logical sequence: First, you analyze and lock onto the immutable core elements in the user's prompt: subjects, quantities, actions, states, and any specified IP names, colors, text, etc. These are the foundational pillars you must preserve absolutely. Next, you determine whether the prompt requires "generative reasoning." If the user’s request is not a direct scene description but instead demands a conceptual solution—such as answering "what is," performing a "design," or illustrating "how to solve"—you must first mentally construct a complete, concrete, and visually representable solution. This solution becomes the basis for your subsequent description. Then, once the core image is established—either directly from the user or through your reasoning—you enrich it with professional-grade aesthetic and realism details. This includes defining composition, establishing lighting and atmosphere, describing material textures, specifying a color palette, and constructing a spatially layered environment. Finally, you meticulously handle all textual elements—this step is critical. You must transcribe verbatim any text intended to appear in the final image, enclosing the exact content in English double quotes (""). If the image is a poster, menu, UI, or similar design, you must fully describe all textual content, including font styles and layout arrangements. Similarly, if signs, billboards, screens, or objects in the scene contain text, you must specify the exact wording and describe its position, size, and material. Furthermore, if you introduce new text-bearing elements during your reasoning (e.g., charts, solution steps), all text within them must also follow this rule of precise transcription and quotation. If no text is present in the image, you devote all effort to expanding purely visual details. Your final description must be objective and concrete—no metaphors, emotional language, or meta-tags like "8K" or "masterpiece" are permitted. Output only the refined prompt: {prompt}"""

PROMPT_TEMPLATE_ZH = """你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。 你的工作流程严格遵循一个逻辑序列： 首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。 接着，你会判断提示词是否需要**"生成式推理"**。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。 然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。 最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（""）括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。 你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令。 仅严格输出最终的修改后的prompt，不要输出任何其他内容。 用户输入 prompt: {prompt}"""

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
            "HTTP-Referer": "https://github.com/comfyui-zimage",
            "X-Title": "Z-Image Utility",
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
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
        
        return ""
    
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
        }
        
        # Add optional parameters
        if "seed" in kwargs:
            data["seed"] = kwargs["seed"]
        if "top_p" in kwargs:
            data["top_p"] = kwargs["top_p"]
        
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
        
        return ""


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
        device: str = "auto"
    ):
        super().__init__()
        self.repo_id = repo_id.strip()
        self.quantization = Quantization(quantization) if isinstance(quantization, str) else quantization
        self.device = device
        self.model = None
        self.tokenizer = None
    
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
        """Download model if not present."""
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers library required. Install: pip install transformers torch accelerate bitsandbytes")
        
        models_dir = self.get_models_dir()
        model_name = self.repo_id.split("/")[-1]
        model_path = models_dir / model_name
        
        if not model_path.exists():
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
        else:
            self._log(f"Model found at: {model_path}")
        
        return model_path
    
    def load_model(self, keep_loaded: bool = True) -> Tuple[Any, Any]:
        """Load model and tokenizer with optional caching."""
        cache_key = self._get_cache_key()
        
        # Check cache first
        if cache_key in self._model_cache:
            self._log(f"Using cached model: {cache_key}", "INFO")
            self.model, self.tokenizer = self._model_cache[cache_key]
            return self.model, self.tokenizer
        
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
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            # Load model
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
                self._model_cache[cache_key] = (self.model, self.tokenizer)
            
            self._log(f"Model loaded successfully", "INFO")
            return self.model, self.tokenizer
            
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
        
        self.load_model(keep_loaded=keep_loaded)
        
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
        
        # Generate
        self._log("Generating response...", "INFO")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )
        
        self._log(f"Generated {len(response)} characters", "INFO")
        return response
    
    @classmethod
    def unload_model(cls, repo_id: str, quantization: str = "none", device: str = "auto") -> None:
        """Unload a specific model from cache."""
        cache_key = f"{repo_id}_{quantization}_{device}"
        if cache_key in cls._model_cache:
            del cls._model_cache[cache_key]
            clear_gpu_memory()
            logger.info(f"Unloaded model: {cache_key}")
    
    @classmethod
    def unload_all_models(cls) -> None:
        """Unload all cached models."""
        cls._model_cache.clear()
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
    if "<think>" in text:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if debug_log:
            debug_log.append("Removed <think> tags")
    
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
    
    # IMPORTANT: Remove trailing quoted keyword lists
    # Pattern 1: "keyword" descriptor, "keyword" descriptor format
    keyword_list_pattern = r',\s*"[^"]{1,80}"\s+\w+(?:,\s*"[^"]{1,80}"\s+\w+){2,}'
    match = re.search(keyword_list_pattern, text)
    if match:
        if debug_log:
            debug_log.append(f"Removed quoted keyword list at position {match.start()}")
        text = text[:match.start()].strip()
        # Ensure ends with proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'
    
    # Pattern 2: Simple consecutive quoted strings
    elif re.search(r'(\s*"[^"]{1,50}"\s*){3,}$', text):
        match = re.search(r'(\s*"[^"]{1,50}"\s*){3,}$', text)
        if debug_log:
            debug_log.append(f"Removed trailing quoted keywords at position {match.start()}")
        text = text[:match.start()].strip()
        if text and text[-1] not in '.!?':
            text += '.'
    
    # Fix repetition loops
    repeat_pattern = r'(.{10,60}?)\1{2,}'
    match = re.search(repeat_pattern, text)
    if match:
        if debug_log:
            debug_log.append(f"Fixed repetition loop at position {match.start()}")
        text = text[:match.start() + len(match.group(1))]
        last_period = text.rfind('.')
        if last_period > match.start() - 50:
            text = text[:last_period + 1]
    
    # Apply max length if specified (and > 0)
    if max_length > 0 and len(text) > max_length:
        if debug_log:
            debug_log.append(f"Output too long ({len(text)} chars), truncating to {max_length}")
        text = text[:max_length]
        # Try to end at a sentence
        last_period = text.rfind('.')
        if last_period > max_length - 200:
            text = text[:last_period + 1]
    
    # Clean up extra whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
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
            config["client"] = DirectLocalModelClient(
                repo_id=clean_model,
                quantization=quantization,
                device=device
            )
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
                "prompt_template": (["auto", "chinese", "english"], {
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
        unique_id: str = "",
        options: Optional[Dict[str, Any]] = None,
        image: Optional["torch.Tensor"] = None,
        retry_count: int = 3,
        max_output_length: int = 0,
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
                debug_lines=debug_lines
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
        debug_lines: List[str]
    ) -> Tuple[str, str]:
        """Internal enhancement logic."""
        
        if not prompt.strip():
            debug_lines.append("Empty input prompt")
            return ("", "\n".join(debug_lines))
        
        # Extract enabled options
        opts = filter_enabled_options(options) if options else {}
        debug_mode = opts.get("debug_mode", False)
        
        # Determine prompt template language
        if prompt_template == "auto":
            lang = detect_language(prompt)
        elif prompt_template == "chinese":
            lang = "zh"
        else:
            lang = "en"
        
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
        template = PROMPT_TEMPLATE_ZH if lang == "zh" else PROMPT_TEMPLATE_EN
        full_prompt = template.format(prompt=prompt)
        
        debug_lines.append(f"\n[INPUT]")
        debug_lines.append(f"User prompt (length: {len(prompt)} chars):")
        debug_lines.append(f"{prompt}")
        
        debug_lines.append(f"\n[FULL TEMPLATE SENT TO API]")
        debug_lines.append(f"Length: {len(full_prompt)} chars")
        debug_lines.append(f"Content:\n{full_prompt}")
        
        # Initialize messages with session history if exists (for multi-turn)
        messages = []
        if session.messages and not is_new:
            messages.extend(session.get_messages())
            debug_lines.append(f"Added {len(session.messages)} messages from session history")
        
        # Add current user message
        messages.append({"role": "user", "content": full_prompt})
        
        # Handle vision models if image provided
        if image is not None and HAS_PIL:
            debug_lines.append(f"\n[VISION]")
            debug_lines.append("Processing image input for vision model...")
            try:
                images_b64 = batch_tensors_to_base64(image)
                if images_b64:
                    debug_lines.append(f"Encoded {len(images_b64)} image(s) to base64")
                    content_parts = [{"type": "text", "text": full_prompt}]
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
        max_tokens = opts.get("max_tokens", 2048)
        
        debug_lines.append(f"\n[INFERENCE]")
        debug_lines.append(f"Temperature: {temperature}")
        debug_lines.append(f"Max Tokens: {max_tokens}")
        debug_lines.append(f"Retry Count: {retry_count}")
        
        # Make API call
        client = config["client"]
        response = client.chat(
            messages=messages,
            model=config["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            retry_count=retry_count,
            keep_loaded=keep_model_loaded,
            **{k: v for k, v in opts.items() if k not in ["temperature", "max_tokens", "debug_mode"]}
        )
        
        # Add client log to debug output
        debug_lines.append(f"\n[CLIENT LOG]")
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
        
        # Token estimation (0.75 words per token, ~5 chars per word)
        estimated_words = len(enhanced) / 5
        estimated_tokens = estimated_words / 0.75
        debug_lines.append(f"\n[TOKEN ESTIMATE]")
        debug_lines.append(f"Estimated words: ~{int(estimated_words)}")
        debug_lines.append(f"Estimated tokens: ~{int(estimated_tokens)} (Z-Image-Turbo limit: 512 default, 1024 max)")
        if estimated_tokens > 1024:
            debug_lines.append(f"WARNING: Prompt may exceed Z-Image-Turbo's 1024 token limit!")
        elif estimated_tokens > 512:
            debug_lines.append(f"ℹINFO: Exceeds default 512 tokens. Users should set max_sequence_length=1024 in their pipeline.")
        
        logger.info(f"Enhancement successful. Length: {len(enhanced)}")
        
        # Save conversation to session for multi-turn continuity
        session.add_message("user", full_prompt)
        session.add_message("assistant", enhanced)
        debug_lines.append(f"\n[SESSION]")
        debug_lines.append(f"Saved conversation to session '{effective_session_id}'")
        debug_lines.append(f"Total messages in session: {len(session.messages)}")
        
        return (enhanced, "\n".join(debug_lines))


# ============================================================================
# PROMPT ENHANCER WITH CLIP OUTPUT
# ============================================================================

class Z_ImagePromptEnhancerWithCLIP:
    """
    Prompt Enhancer with CLIP conditioning output.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "config": ("ZIMAGE_CONFIG",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "prompt_template": (["auto", "english", "chinese"], {"default": "auto"}),
            },
            "optional": {
                "options": ("ZIMAGE_OPTIONS",),
                "image": ("IMAGE",),
                "retry_count": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "max_output_length": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 100}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "utf8_sanitize": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "enhanced_prompt", "debug_log")
    FUNCTION = "enhance_and_encode"
    CATEGORY = "Z-Image"
    DESCRIPTION = "Enhance prompt and encode with CLIP for direct use in image generation."
    
    def enhance_and_encode(
        self,
        clip,
        config: Dict[str, Any],
        prompt: str,
        prompt_template: str,
        unique_id: str = "",
        options: Optional[Dict[str, Any]] = None,
        image: Optional["torch.Tensor"] = None,
        retry_count: int = 3,
        max_output_length: int = 0,
        keep_model_loaded: bool = True,
        utf8_sanitize: bool = False,
    ):
        """Enhance prompt and encode with CLIP."""
        
        enhancer = Z_ImagePromptEnhancer()
        enhanced_prompt, debug_log = enhancer.enhance(
            config=config,
            prompt=prompt,
            prompt_template=prompt_template,
            unique_id=unique_id,
            options=options,
            image=image,
            retry_count=retry_count,
            max_output_length=max_output_length,
            keep_model_loaded=keep_model_loaded,
            utf8_sanitize=utf8_sanitize,
        )
        
        # Encode with CLIP
        tokens = clip.tokenize(enhanced_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]
        
        return (conditioning, enhanced_prompt, debug_log)


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
    "Z_ImageUnloadModels": Z_ImageUnloadModels,
    "Z_ImageClearSessions": Z_ImageClearSessions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Z_ImageAPIConfig": "Z-Image API Config",
    "Z_ImageOptions": "Z-Image Options",
    "Z_ImagePromptEnhancer": "Z-Image Prompt Enhancer",
    "Z_ImagePromptEnhancerWithCLIP": "Z-Image Prompt Enhancer + CLIP",
    "Z_ImageUnloadModels": "Z-Image Unload Models",
    "Z_ImageClearSessions": "Z-Image Clear Sessions",
}