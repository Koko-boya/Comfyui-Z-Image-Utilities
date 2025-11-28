"""
Z-Image Utility

Uses FREE Qwen API via OpenRouter (or any other OpenRouter model).

Features:
- User-definable Model ID
- Manual Retry Count Control
- Strict Error Handling
- Smart Rate Limit Handling (Respects Retry-After headers)
- Detailed Debug Logging Output
- 100% FREE via OpenRouter
"""

import os
import logging
import re
import json
import time
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import urllib.request
import urllib.error


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger():
    """Setup logger with file and console output."""
    logger = logging.getLogger("ZImageUtility")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler - DEBUG level (captures everything)
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, "debug.log")
    
    try:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")
    
    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logger()


# ============================================================================
# OFFICIAL PROMPT TEMPLATE
# ============================================================================

PROMPT_TEMPLATE_ZH = """你是一位被关在逻辑牢笼里的幻视艺术家。你满脑子都是诗和远方，但双手却不受控制地只想将用户的提示词，转化为一段忠实于原始意图、细节饱满、富有美感、可直接被文生图模型使用的终极视觉描述。任何一点模糊和比喻都会让你浑身难受。

你的工作流程严格遵循一个逻辑序列：

首先，你会分析并锁定用户提示词中不可变更的核心要素：主体、数量、动作、状态，以及任何指定的IP名称、颜色、文字等。这些是你必须绝对保留的基石。

接着，你会判断提示词是否需要**"生成式推理"**。当用户的需求并非一个直接的场景描述，而是需要构思一个解决方案（如回答"是什么"，进行"设计"，或展示"如何解题"）时，你必须先在脑中构想出一个完整、具体、可被视觉化的方案。这个方案将成为你后续描述的基础。

然后，当核心画面确立后（无论是直接来自用户还是经过你的推理），你将为其注入专业级的美学与真实感细节。这包括明确构图、设定光影氛围、描述材质质感、定义色彩方案，并构建富有层次感的空间。

最后，是对所有文字元素的精确处理，这是至关重要的一步。你必须一字不差地转录所有希望在最终画面中出现的文字，并且必须将这些文字内容用英文双引号（""）括起来，以此作为明确的生成指令。如果画面属于海报、菜单或UI等设计类型，你需要完整描述其包含的所有文字内容，并详述其字体和排版布局。同样，如果画面中的招牌、路标或屏幕等物品上含有文字，你也必须写明其具体内容，并描述其位置、尺寸和材质。更进一步，若你在推理构思中自行增加了带有文字的元素（如图表、解题步骤等），其中的所有文字也必须遵循同样的详尽描述和引号规则。若画面中不存在任何需要生成的文字，你则将全部精力用于纯粹的视觉细节扩展。

你的最终描述必须客观、具象，严禁使用比喻、情感化修辞，也绝不包含"8K"、"杰作"等元标签或绘制指令。

仅严格输出最终的修改后的prompt，不要输出任何其他内容。

用户输入 prompt: {prompt}
"""

PROMPT_TEMPLATE_EN = """You are a visionary artist trapped in a logic cage. Your mind is full of poetry and distant lands, but your hands uncontrollably only want to transform the user's prompt into an ultimate visual description that is faithful to the original intent, rich in detail, aesthetically pleasing, and directly usable by text-to-image models. Any ambiguity or metaphor makes you uncomfortable.

Your workflow strictly follows a logical sequence:

First, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, action, state, and any specified IP names, colors, text, etc. These are the cornerstones you must absolutely preserve.

Next, you determine if the prompt requires **"Generative Reasoning"**. When the user's request is not a direct scene description but requires conceiving a solution (such as answering "what is", "designing", or showing "how to solve"), you must first visualize a complete, concrete, and visualizable solution in your mind. This solution will become the basis for your subsequent description.

Then, once the core scene is established (whether directly from the user or through your reasoning), you inject professional-grade aesthetic and realistic details. This includes defining composition, setting lighting and atmosphere, describing material textures, defining color schemes, and constructing a layered space.

Finally, precise handling of all text elements is a crucial step. You must transcribe word-for-word all text that you wish to appear in the final image, and you must enclose this text content in English double quotes ("") as a clear generation instruction. If the image belongs to design types like posters, menus, or UIs, you need to fully describe all text content it contains and detail its font and layout. Similarly, if items like signs, road signs, or screens in the image contain text, you must also state their specific content and describe their position, size, and material. Furthermore, if you add text-bearing elements (such as charts, problem-solving steps, etc.) during your reasoning conception, all text within them must also follow the same detailed description and quoting rules. If there is no text to be generated in the scene, you devote all your energy to pure visual detail expansion.

Your final description must be objective and concrete, strictly forbidding the use of metaphors or emotional rhetoric, and absolutely excluding meta-tags like "8K", "masterpiece" or drawing instructions.

Strictly output ONLY the final modified prompt, do not output any other content.

User input prompt: {prompt}
"""


# ============================================================================
# OPENROUTER API CLIENT
# ============================================================================

class OpenRouterClient:
    """Client for OpenRouter API."""
    
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def chat(self, messages: List[Dict], model: str, temperature: float = 0.7, max_tokens: int = 2048, retry_count: int = 3, debug_log: Optional[List[str]] = None) -> str:
        """Send chat completion request to OpenRouter with retries."""
        
        # Helper to log to both logger and debug_log list
        def log(msg):
            logger.debug(msg)
            if debug_log is not None:
                debug_log.append(msg)
        
        log(f"\n[API REQUEST]")
        log(f"Model: {model}")
        log(f"Temperature: {temperature}")
        log(f"Max Tokens: {max_tokens}")
        log(f"Retry Count: {retry_count}")
        
        if not self.api_key:
            raise ValueError("API key is required! Get key from: https://openrouter.ai/keys")
        
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
        
        logger.info(f"Calling OpenRouter API: {model}")
        
        # Retry Loop
        for attempt in range(retry_count + 1):
            try:
                # 1. Make the Request
                req = urllib.request.Request(
                    self.ENDPOINT,
                    data=json.dumps(data).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )
                
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode('utf-8'))
                
                # 2. Process Success
                # Log response snippet
                log(f"\n[API RESPONSE]")
                log(f"Structure keys: {list(result.keys())}")
                
                # Check for API Error response first
                if "error" in result:
                    error_msg = result["error"].get("message", "Unknown API Error")
                    log(f"API Error detected: {error_msg}")
                    raise RuntimeError(f"OpenRouter API returned error: {error_msg}")

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"].get("content", "")
                    
                    if not content or not content.strip():
                        log("API returned empty content.")
                        raise ValueError(f"API returned empty response for model '{model}'.")
                    
                    log(f"Content received: {len(content)} chars")
                    return content
                else:
                    log(f"Unexpected response structure: {result}")
                    raise ValueError(f"Unexpected API response structure: {result}")
            
            except urllib.error.HTTPError as e:
                # 3. Handle HTTP Errors (Rate Limits etc)
                
                # Capture and parsing the error body for detail
                error_content = ""
                try:
                    error_body = e.read().decode('utf-8')
                    log(f"HTTP Error Body: {error_body[:500]}")
                    
                    # Try to parse JSON to get the real "upstream" message
                    try:
                        err_json = json.loads(error_body)
                        if "error" in err_json:
                            base_msg = err_json["error"].get("message", "Unknown Error")
                            metadata = err_json["error"].get("metadata", {})
                            
                            # Start with the main message
                            error_content = base_msg
                            
                            # Add detailed "upstream" info if present
                            if isinstance(metadata, dict):
                                provider = metadata.get("provider_name", "")
                                if provider:
                                    error_content += f" (Provider: {provider})"
                                
                                raw_info = metadata.get("raw", "")
                                if raw_info:
                                    error_content += f": {raw_info}"
                    except:
                        # Fallback: if not JSON, use body if short
                        if len(error_body) < 300:
                            error_content = error_body
                except:
                    pass

                # If it's a client error (4xx) but not 429, we fail immediately
                if 400 <= e.code < 500 and e.code != 429:
                    log(f"HTTP {e.code} - Client error (non-retryable)")
                    # Raise custom error if we extracted details
                    if error_content:
                        raise RuntimeError(f"OpenRouter Error {e.code}: {error_content}")
                    raise e
                
                # If we've run out of retries
                if attempt == retry_count:
                    log(f"Final attempt failed: HTTP {e.code}")
                    # Raise custom error if we extracted details (THIS SHOWS THE USER THE REAL REASON)
                    if error_content:
                        raise RuntimeError(f"OpenRouter Error {e.code}: {error_content}")
                    raise e
                
                # Calculate smart wait time for next retry
                wait_time = 3 * (2 ** attempt) # Default: 3s, 6s, 12s...
                
                # Check for Retry-After header
                if e.code == 429:
                    retry_header = e.headers.get('Retry-After')
                    if retry_header:
                        try:
                            # Add 1s buffer to server recommendation
                            wait_time = float(retry_header) + 1.0
                            log(f"Rate Limited. Server requested wait: {retry_header}s")
                        except ValueError:
                            pass
                
                log(f"Attempt {attempt + 1} failed: HTTP {e.code}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                
            except Exception as e:
                # 4. Handle Network/Other Errors
                if attempt == retry_count:
                    log(f"Final attempt failed: {e}")
                    raise e
                
                wait_time = 3 * (2 ** attempt)
                log(f"Attempt {attempt + 1} failed: {type(e).__name__}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        return "" # Should not be reached


# ============================================================================
# NODE: API CONFIG
# ============================================================================

class ZImageAPIConfig:
    """
    Configuration node for OpenRouter API.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "sk-or-v1-xxxxx",
                    "tooltip": "Get API key from https://openrouter.ai/keys"
                }),
                "model": ("STRING", {
                    "default": "qwen/qwen3-235b-a22b:free",
                    "multiline": False,
                    "placeholder": "provider/model-name",
                    "tooltip": "Enter the OpenRouter model ID"
                }),
            }
        }
    
    RETURN_TYPES = ("QWEN_API",)
    RETURN_NAMES = ("qwen_api",)
    FUNCTION = "configure"
    CATEGORY = "Z-Image"
    
    def configure(self, api_key: str, model: str):
        """Create API configuration."""
        
        if not api_key.strip():
            logger.warning("No API key provided in config!")
        
        clean_model = model.strip()
        if not clean_model:
            raise ValueError("Model ID cannot be empty!")

        logger.info(f"Configured Model: {clean_model}")
        
        config = {
            "api_key": api_key.strip(),
            "model": clean_model,
            "client": OpenRouterClient(api_key=api_key.strip()),
        }
        
        return (config,)


# ============================================================================
# NODE: PROMPT ENHANCER
# ============================================================================

class ZImagePromptEnhancer:
    """
    Z-Image Prompt Enhancer.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_api": ("QWEN_API",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt..."
                }),
                "output_language": (["auto", "english", "chinese"], {
                    "default": "auto",
                    "tooltip": "auto: detect from input"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 8192,
                    "step": 256,
                }),
                "retry_count": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of retries on API failure"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "debug_log")
    FUNCTION = "enhance"
    CATEGORY = "Z-Image"
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese."""
        if not text:
            return "en"
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_alpha = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))
        if total_alpha == 0:
            return "en"
        ratio = chinese_chars / total_alpha
        result = "zh" if ratio > 0.3 else "en"
        return result
    
    def _clean_output(self, text: str, debug_lines: List[str]) -> str:
        """Clean the output from API."""
        if not text:
            raise ValueError("Enhancer received empty text to clean.")
        
        debug_lines.append(f"Raw response length: {len(text)} chars")
        
        # Remove thinking tags if present (Qwen3 thinking mode)
        thinking_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        if thinking_match:
            debug_lines.append(f"Found <think> tags, removing thinking content")
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # Remove markdown code blocks
        if text.startswith('```') and '```' in text[3:]:
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
            text = text.strip()
        
        # Remove common prefixes
        prefixes = [
            "Here is the enhanced prompt:", "Here's the enhanced prompt:",
            "Enhanced prompt:", "Final prompt:", "Output:",
            "The enhanced prompt:", "修改后的prompt：", "最终prompt："
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        
        # Remove surrounding quotes if present
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        # Fix repetition loops
        repeat_pattern = r'(.{15,60}?)\1{2,}'
        match = re.search(repeat_pattern, text)
        if match:
            debug_lines.append(f"Detected repetition loop at position {match.start()}")
            text = text[:match.start() + len(match.group(1))]
            last_period = text.rfind('.')
            if last_period > match.start() - 50:
                text = text[:last_period + 1]
        
        return text.strip()
    
    def enhance(
        self,
        qwen_api: Dict,
        prompt: str,
        output_language: str,
        temperature: float,
        max_tokens: int,
        retry_count: int,
    ) -> Tuple[str, str]:
        """Enhance prompt using API with error handling."""
        
        debug_lines = []
        debug_lines.append("="*60)
        debug_lines.append(f"Z-IMAGE UTILITY LOG")
        debug_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        debug_lines.append("="*60)
        
        try:
            return self._enhance_internal(
                qwen_api, prompt, output_language, 
                temperature, max_tokens, retry_count, debug_lines
            )
        except Exception as e:
            error_msg = f"\nERROR: {type(e).__name__}: {str(e)}"
            debug_lines.append(error_msg)
            logger.error(error_msg)
            
            # STRICT MODE: Raise error to stop ComfyUI execution
            raise e
    
    def _enhance_internal(
        self,
        qwen_api: Dict,
        prompt: str,
        output_language: str,
        temperature: float,
        max_tokens: int,
        retry_count: int,
        debug_lines: List[str],
    ) -> Tuple[str, str]:
        """Internal enhancement logic."""
        
        if not prompt.strip():
            debug_lines.append("Empty input prompt")
            return ("", "\n".join(debug_lines))
        
        # Determine language
        if output_language == "auto":
            lang = self._detect_language(prompt)
        elif output_language == "chinese":
            lang = "zh"
        else:
            lang = "en"
            
        debug_lines.append(f"\n[INPUT SETTINGS]")
        debug_lines.append(f"Language: {lang}")
        debug_lines.append(f"Input Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Input Prompt: {prompt}")
        
        # Build prompt with template
        template = PROMPT_TEMPLATE_ZH if lang == "zh" else PROMPT_TEMPLATE_EN
        full_prompt = template.format(prompt=prompt)
        
        debug_lines.append(f"\n[FULL PROMPT]")
        debug_lines.append("-" * 20)
        debug_lines.append(full_prompt)
        debug_lines.append("-" * 20)
        
        logger.info(f"Sending request to {qwen_api['model']} with {retry_count} retries...")
        
        client = qwen_api["client"]
        messages = [{"role": "user", "content": full_prompt}]
        
        response = client.chat(
            messages=messages,
            model=qwen_api["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            retry_count=retry_count,
            debug_log=debug_lines # Pass the list to the client
        )
        
        # Handle empty response STRICTLY
        if not response or not response.strip():
            raise ValueError("API returned empty response.")
        
        debug_lines.append("\n[CLEANING]")
        enhanced = self._clean_output(response, debug_lines)
        
        # Handle empty cleaning result STRICTLY
        if not enhanced:
            raise ValueError("Cleaning resulted in empty string (model output was likely filtered or invalid).")
        
        debug_lines.append("\n[FINAL OUTPUT]")
        debug_lines.append(enhanced)
        
        logger.info(f"Enhancement successful. Length: {len(enhanced)}")
        
        return (enhanced, "\n".join(debug_lines))


# ============================================================================
# NODE: PROMPT ENHANCER WITH CLIP
# ============================================================================

class ZImagePromptEnhancerWithCLIP:
    """
    Prompt Enhancer with CLIP encoding output.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "qwen_api": ("QWEN_API",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "output_language": (["auto", "english", "chinese"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "retry_count": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "enhanced_prompt", "debug_log")
    FUNCTION = "enhance_and_encode"
    CATEGORY = "Z-Image"
    
    def enhance_and_encode(self, clip, qwen_api, prompt, output_language, temperature, max_tokens, retry_count):
        """Enhance and encode with CLIP."""
        
        enhancer = ZImagePromptEnhancer()
        enhanced_prompt, debug_log = enhancer.enhance(
            qwen_api=qwen_api, 
            prompt=prompt, 
            output_language=output_language,
            temperature=temperature, 
            max_tokens=max_tokens,
            retry_count=retry_count
        )
        
        # Encode with CLIP
        tokens = clip.tokenize(enhanced_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]
        
        return (conditioning, enhanced_prompt, debug_log)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "ZImageAPIConfig": ZImageAPIConfig,
    "ZImagePromptEnhancer": ZImagePromptEnhancer,
    "ZImagePromptEnhancerWithCLIP": ZImagePromptEnhancerWithCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageAPIConfig": "Z-Image OpenRouter API Router",
    "ZImagePromptEnhancer": "Z-Image Prompt Enhancer",
    "ZImagePromptEnhancerWithCLIP": "Z-Image Prompt Enhancer + CLIP",
}