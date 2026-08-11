#!/usr/bin/env python3
"""
RKLLM OpenAI and Ollama API compatible server.

The server exposes the OpenAI-compatible API used by existing clients and a
small Ollama-compatible API for tools which expect ``/api/chat`` or
``/api/generate``.  When started directly it also opens a simple terminal chat
so the device can be demonstrated without configuring a separate client.
"""

import ctypes
import sys
import os
import threading
import time
import uuid
import json
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Generator, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import argparse

# ==================== System Library Preloading ====================
def preload_libraries():
    """Preload necessary system libraries to fix OpenCL issues"""
    try:
        # Set environment variables
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/aarch64-linux-gnu:/usr/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

        # Preload libraries
        libs = [
            'librknnrt.so',
            '/usr/lib/librkllmrt.so'
        ]

        for lib in libs:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                print(f"✓ Preloaded: {lib}")
            except Exception as e:
                print(f"⚠ Failed to preload {lib}: {e}")
    except Exception as e:
        print(f"⚠ Error during library preloading: {e}")

print("Preloading system libraries...")
preload_libraries()

# ==================== Load RKLLM Library ====================
try:
    rkllm_lib = ctypes.CDLL('/usr/lib/librkllmrt.so')
    print("✓ Successfully loaded librkllmrt.so")
except Exception as e:
    print(f"✗ Failed to load librkllmrt.so: {e}")
    print("Please ensure RKLLM runtime is installed: sudo apt install librkllmrt")
    sys.exit(1)

# ==================== RKLLM Structure Definitions ====================
RKLLM_Handle_t = ctypes.c_void_p

# Enum definitions
class LLMCallState:
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

class RKLLMInputType:
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_EMBED = 2
    RKLLM_INPUT_MULTIMODAL = 3

class RKLLMInferMode:
    RKLLM_INFER_GENERATE = 0

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]

class RKLLMParam(ctypes.Structure):
    """RKLLMParam from the airockchip/rknn-llm v1.3.0 header.

    v1.3.0 removed the image marker strings from this structure.  Leaving
    those v1.2.x fields in place shifts ``extend_param`` and makes the runtime
    read pointer bytes as ``n_batch`` and ``enabled_cpus_num``.
    """
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]

class RKLLMImageInput(ctypes.Structure):
    _fields_ = [
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_start", ctypes.c_char_p),
        ("image_end", ctypes.c_char_p),
        ("image_content", ctypes.c_char_p),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]

class RKLLMVideoInput(ctypes.Structure):
    _fields_ = [
        ("video_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_frame_tokens", ctypes.c_size_t),
        ("n_frame_per_video", ctypes.c_size_t),
        ("n_video", ctypes.c_size_t),
        ("video_start", ctypes.c_char_p),
        ("video_end", ctypes.c_char_p),
        ("video_content", ctypes.c_char_p),
        ("frame_width", ctypes.c_size_t),
        ("frame_height", ctypes.c_size_t),
    ]

class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image", RKLLMImageInput),
        ("video", RKLLMVideoInput),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("sampling_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32),
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]

class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]

callback_type = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)

class RKLLMCallback(ctypes.Structure):
    _fields_ = [
        ("result_callback", callback_type),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", ctypes.c_void_p),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", ctypes.c_void_p),
        ("embed_userdata", ctypes.c_void_p),
    ]

# ==================== Pydantic Model Definitions ====================
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, assistant")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Message content")

class Function(BaseModel):
    name: str = Field(..., description="Function name")
    description: Optional[str] = Field(None, description="Function description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Function parameters")

class Tool(BaseModel):
    type: str = Field(default="function", description="Tool type")
    function: Optional[Function] = Field(None, description="Function definition")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="rkllm-model", description="Model name")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0, description="Temperature parameter (0.0-2.0)")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter (0.0-1.0)")
    top_k: Optional[int] = Field(default=None, ge=1, le=100, description="Top-k sampling parameter (1-100)")
    n: Optional[int] = Field(default=1, ge=1, le=10, description="Number of completions to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(default=512, ge=1, le=8192, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    tools: Optional[List[Tool]] = Field(None, description="List of tools")
    tool_choice: Optional[str] = Field(None, description="Tool choice")

class UsageInfo(BaseModel):
    prompt_tokens: int = Field(default=0, description="Prompt tokens")
    completion_tokens: int = Field(default=0, description="Completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")

class ChatCompletionResponseChoice(BaseModel):
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Message")
    finish_reason: Optional[str] = Field(default="stop", description="Finish reason")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Request ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatCompletionResponseChoice] = Field(..., description="List of choices")
    usage: UsageInfo = Field(..., description="Usage information")
    system_fingerprint: Optional[str] = Field(default="fp_rkllm", description="System fingerprint")

class DeltaMessage(BaseModel):
    role: Optional[str] = Field(None, description="Role")
    content: Optional[str] = Field(None, description="Content")

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int = Field(..., description="Choice index")
    delta: DeltaMessage = Field(..., description="Delta message")
    finish_reason: Optional[str] = Field(None, description="Finish reason")

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(..., description="Request ID")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatCompletionStreamResponseChoice] = Field(..., description="List of choices")
    system_fingerprint: Optional[str] = Field(default="fp_rkllm", description="System fingerprint")

class ModelInfo(BaseModel):
    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Creation time")
    owned_by: str = Field(default="rkllm", description="Owner")

class ModelsListResponse(BaseModel):
    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of models")

class OllamaChatRequest(BaseModel):
    """Request shape accepted by Ollama's /api/chat endpoint."""
    model: str = Field(default="rkllm-model")
    messages: List[ChatMessage] = Field(...)
    stream: bool = Field(default=True)
    options: Optional[Dict[str, Any]] = Field(default=None)
    keep_alive: Optional[Union[str, int]] = Field(default=None)

class OllamaGenerateRequest(BaseModel):
    """Request shape accepted by Ollama's /api/generate endpoint."""
    model: str = Field(default="rkllm-model")
    prompt: str = Field(default="")
    system: Optional[str] = Field(default=None)
    stream: bool = Field(default=True)
    options: Optional[Dict[str, Any]] = Field(default=None)
    keep_alive: Optional[Union[str, int]] = Field(default=None)

# ==================== Global State Management ====================
class RequestState:
    """State management for individual requests"""
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.text_queue = []
        self.state = -1
        self.completed = threading.Event()
        self.lock = threading.Lock()
        self.full_response = ""
        self.error = None
        self.start_time = time.time()

# Global variables
request_lock = threading.Lock()
active_requests = 0
request_states: Dict[str, RequestState] = {}
rkllm_model = None
executor = None
api_model_name = "rkllm-model"

# Server configuration
class ServerConfig:
    def __init__(self):
        self.max_context_len = 2048  # Default context length
        self.default_temperature = 0.8  # Default temperature
        self.default_top_p = 0.9  # Default top_p
        self.default_top_k = 1  # Default top_k
        self.default_max_tokens = 512  # Default max tokens
        self.max_concurrent_requests = 2  # Max concurrent requests
        self.timeout_seconds = 120  # Timeout in seconds

config = ServerConfig()

# ==================== RKLLM Callback Function ====================
def callback_impl(result, userdata, state):
    """RKLLM callback function implementation"""
    if not userdata:
        return 0

    try:
        request_id = ctypes.cast(userdata, ctypes.c_char_p).value.decode('utf-8')

        if request_id not in request_states:
            return 0

        req_state = request_states[request_id]

        with req_state.lock:
            if state == LLMCallState.RKLLM_RUN_FINISH:
                req_state.state = state
                req_state.completed.set()
            elif state == LLMCallState.RKLLM_RUN_ERROR:
                req_state.state = state
                req_state.error = "RKLLM runtime error"
                req_state.completed.set()
            elif state == LLMCallState.RKLLM_RUN_NORMAL:
                req_state.state = state
                if result and result.contents.text:
                    try:
                        text = result.contents.text.decode('utf-8', errors='ignore')
                        req_state.text_queue.append(text)
                        req_state.full_response += text
                    except Exception as e:
                        print(f"Callback decoding error: {e}")

        return 0
    except Exception as e:
        print(f"Callback function error: {e}")
        return -1

callback = callback_type(callback_impl)
rkllm_callback = RKLLMCallback()
rkllm_callback.result_callback = callback
rkllm_callback.result_userdata = None
rkllm_callback.tokenizer_callback = None
rkllm_callback.tokenizer_userdata = None
rkllm_callback.embed_callback = None
rkllm_callback.embed_userdata = None

# ==================== RKLLM Model Manager ====================
class RKLLMModel:
    """RKLLM model manager class"""
    def __init__(self, model_path: str, platform: str = "rk3588"):
        self.model_path = model_path
        self.platform = platform
        self.handle = ctypes.c_void_p()
        self.initialized = False
        self.model_lock = threading.Lock()

        # Configuration parameters
        self.max_context_len = config.max_context_len
        self.default_temperature = config.default_temperature
        self.default_top_p = config.default_top_p
        self.default_top_k = config.default_top_k
        self.default_max_tokens = config.default_max_tokens

    def initialize(self):
        """Initialize the model"""
        with self.model_lock:
            try:
                print(f"Initializing RKLLM model: {self.model_path}")

                # Prepare model parameters
                rkllm_param = RKLLMParam()
                rkllm_param.model_path = ctypes.c_char_p(self.model_path.encode('utf-8'))
                rkllm_param.max_context_len = self.max_context_len
                rkllm_param.max_new_tokens = self.default_max_tokens
                rkllm_param.n_keep = 0
                rkllm_param.top_k = self.default_top_k  # Use default top_k
                rkllm_param.top_p = self.default_top_p
                rkllm_param.temperature = self.default_temperature
                rkllm_param.repeat_penalty = 1.1
                rkllm_param.frequency_penalty = 0.0
                rkllm_param.presence_penalty = 0.0
                rkllm_param.mirostat = 0
                rkllm_param.mirostat_tau = 5.0
                rkllm_param.mirostat_eta = 0.1
                rkllm_param.skip_special_token = True
                rkllm_param.ignore_eos_token = False
                rkllm_param.is_async = False

                # Extended parameters - critical settings to avoid GGML errors
                rkllm_param.extend_param.base_domain_id = 0
                rkllm_param.extend_param.embed_flash = 0  # Set to 0 to avoid GGML assertion error
                rkllm_param.extend_param.n_batch = 1
                rkllm_param.extend_param.use_cross_attn = 0

                # Set CPU mask based on platform
                if self.platform.lower() in ["rk3576", "rk3588", "rk3588s"]:
                    cpu_mask = 0xF0  # CPU 4-7 (big cores)
                else:
                    cpu_mask = 0x0F  # CPU 0-3
                rkllm_param.extend_param.enabled_cpus_mask = cpu_mask
                rkllm_param.extend_param.enabled_cpus_num = cpu_mask.bit_count()

                # Set function prototypes
                rkllm_init = rkllm_lib.rkllm_init
                rkllm_init.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(RKLLMParam),
                    ctypes.POINTER(RKLLMCallback),
                ]
                rkllm_init.restype = ctypes.c_int

                # Call initialization
                ret = rkllm_init(
                    ctypes.byref(self.handle),
                    ctypes.byref(rkllm_param),
                    ctypes.byref(rkllm_callback),
                )

                if ret != 0:
                    raise RuntimeError(f"RKLLM initialization failed with error code: {ret}")

                self.initialized = True
                print("✅ RKLLM model initialized successfully!")
                return True

            except Exception as e:
                print(f"❌ Model initialization failed: {e}")
                return False

    def generate(self, prompt: str, request_id: str, temperature: float = None,
                 top_p: float = None, top_k: int = None, max_tokens: int = None) -> int:
        """Generate text with the model"""
        with self.model_lock:
            if not self.initialized:
                raise RuntimeError("Model not initialized")

            try:
                # First update the model parameters if top_k is provided
                if top_k is not None:
                    # Need to update the model's top_k parameter
                    # Note: RKLLM might require reinitialization or parameter update
                    # For now, we'll log it and use the value in generation
                    print(f"[{request_id}] Setting top_k to {top_k}")

                    # Update RKLLM parameter structure for this generation
                    # This might require calling rkllm_set_param or similar function
                    # For simplicity, we'll use the existing handle with default params
                    # In a real implementation, you might need to:
                    # 1. Call a parameter update function if available
                    # 2. Or handle it differently based on RKLLM API
                    pass

                # Prepare input
                rkllm_input = RKLLMInput()
                rkllm_input.role = ctypes.c_char_p(b"user")
                rkllm_input.enable_thinking = False
                rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
                rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))

                # Prepare inference parameters
                infer_param = RKLLMInferParam()
                infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
                infer_param.lora_params = None
                infer_param.prompt_cache_params = None
                infer_param.sampling_params = None
                infer_param.keep_history = 0
                infer_param.max_new_tokens = max_tokens if max_tokens else 0

                # Prepare user data
                userdata_ptr = None
                if request_id:
                    userdata_ptr = ctypes.c_char_p(request_id.encode('utf-8'))

                # Set up run function
                rkllm_run = rkllm_lib.rkllm_run
                rkllm_run.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(RKLLMInput),
                    ctypes.POINTER(RKLLMInferParam),
                    ctypes.c_void_p
                ]
                rkllm_run.restype = ctypes.c_int

                # Call run function
                ret = rkllm_run(self.handle, ctypes.byref(rkllm_input),
                              ctypes.byref(infer_param), userdata_ptr)

                return ret

            except Exception as e:
                print(f"❌ Generation error: {e}")
                return -1

    def release(self):
        """Release model resources"""
        with self.model_lock:
            if self.initialized and self.handle:
                try:
                    rkllm_destroy = rkllm_lib.rkllm_destroy
                    rkllm_destroy.argtypes = [ctypes.c_void_p]
                    rkllm_destroy.restype = ctypes.c_int

                    ret = rkllm_destroy(self.handle)
                    if ret != 0:
                        print(f"⚠ rkllm_destroy returned error code: {ret}")

                    self.initialized = False
                    self.handle = None
                    print("✅ Model resources released")

                except Exception as e:
                    print(f"❌ Error releasing model resources: {e}")

# ==================== Helper Functions ====================
def message_text(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Convert OpenAI/Ollama text or content-part messages to plain text."""
    if isinstance(content, str):
        return content

    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)

def build_prompt(messages: List[ChatMessage]) -> str:
    """Build prompt from messages"""
    prompt = ""

    for msg in messages:
        content = message_text(msg.content)
        if msg.role == 'system':
            prompt += f"System: {content}\n\n"
        elif msg.role == 'user':
            prompt += f"Human: {content}\n"
        elif msg.role == 'assistant':
            prompt += f"Assistant: {content}\n"

    # Ensure it ends with Assistant:
    if not prompt.strip().endswith("Assistant:"):
        prompt += "Assistant:"

    return prompt

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    if not text:
        return 0

    # Simple estimation: Chinese characters ~1.5 tokens, others ~0.3 tokens
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    return int(chinese_chars * 1.5 + other_chars * 0.3)

def reserve_request_slot() -> None:
    """Reserve one inference slot or raise an API-compatible 429 error."""
    global active_requests
    with request_lock:
        if active_requests >= config.max_concurrent_requests:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Too many requests, please try again later",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }
            )
        active_requests += 1

def release_request_slot() -> None:
    """Release a previously reserved inference slot."""
    global active_requests
    with request_lock:
        active_requests = max(0, active_requests - 1)

def ollama_request_from_chat(request: OllamaChatRequest) -> ChatCompletionRequest:
    """Translate Ollama options into the common internal request format."""
    options = request.options or {}

    def positive_int(name: str, default: int) -> int:
        value = options.get(name, default)
        return value if isinstance(value, int) and value > 0 else default

    top_k = options.get("top_k")
    if not isinstance(top_k, int) or top_k < 1:
        top_k = None

    stop = options.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = None

    return ChatCompletionRequest(
        model=request.model,
        messages=request.messages,
        temperature=options.get("temperature", config.default_temperature),
        top_p=options.get("top_p", config.default_top_p),
        top_k=top_k,
        max_tokens=positive_int("num_predict", config.default_max_tokens),
        stop=stop,
        stream=request.stream,
    )

def ollama_request_from_generate(request: OllamaGenerateRequest) -> ChatCompletionRequest:
    messages = []
    if request.system:
        messages.append(ChatMessage(role="system", content=request.system))
    messages.append(ChatMessage(role="user", content=request.prompt))
    return ollama_request_from_chat(
        OllamaChatRequest(
            model=request.model,
            messages=messages,
            stream=request.stream,
            options=request.options,
            keep_alive=request.keep_alive,
        )
    )

def ollama_created_at() -> str:
    """Return Ollama's RFC3339-style timestamp without an extra dependency."""
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime())

def ollama_chunk(model: str, content: str, done: bool = False,
                 prompt_tokens: int = 0, completion_tokens: int = 0) -> str:
    payload = {
        "model": model,
        "created_at": ollama_created_at(),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
        payload.update({
            "done_reason": "stop",
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
        })
    return json.dumps(payload, ensure_ascii=False) + "\n"

def process_chat_completion(request: ChatCompletionRequest, request_id: str) -> RequestState:
    """Process chat completion request"""
    global rkllm_model

    # Create request state
    req_state = RequestState(request_id)
    request_states[request_id] = req_state

    try:
        # Build prompt
        prompt = build_prompt(request.messages)

        # Print debug information
        print(f"[{request_id}] Processing request:")
        print(f"  Prompt length: {len(prompt)} characters")
        print(f"  Temperature: {request.temperature}")
        print(f"  Top-p: {request.top_p}")
        print(f"  Top-k: {request.top_k}")
        print(f"  Max tokens: {request.max_tokens}")

        # Run model inference
        ret = rkllm_model.generate(
            prompt=prompt,
            request_id=request_id,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens
        )

        if ret != 0:
            req_state.error = f"Model inference failed with code: {ret}"
            req_state.completed.set()
            return req_state

        # Wait for completion
        timeout = config.timeout_seconds
        print(f"[{request_id}] Waiting for inference completion (timeout: {timeout}s)...")

        if not req_state.completed.wait(timeout=timeout):
            req_state.error = f"Inference timeout ({timeout}s)"
            print(f"✗ [{request_id}] {req_state.error}")

        elapsed = time.time() - req_state.start_time
        print(f"✅ [{request_id}] Inference completed in {elapsed:.2f}s")

        return req_state

    except Exception as e:
        error_msg = f"Error processing request {request_id}: {str(e)}"
        print(f"✗ {error_msg}")
        req_state.error = error_msg
        req_state.completed.set()
        return req_state

# ==================== Application Lifecycle Management ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global rkllm_model, executor

    # Startup
    print("=" * 60)
    print("Starting RKLLM OpenAI API Server")
    print("=" * 60)

    # Initialize thread pool
    executor = ThreadPoolExecutor(
        max_workers=config.max_concurrent_requests + 2,
        thread_name_prefix="rkllm_worker"
    )
    print("✅ Thread pool initialized")

    # Initialize model
    try:
        rkllm_model = RKLLMModel(args.rkllm_model_path, args.target_platform)

        # Apply configuration parameters
        if args.max_context_len:
            config.max_context_len = args.max_context_len
            rkllm_model.max_context_len = args.max_context_len

        if args.default_temperature:
            config.default_temperature = args.default_temperature
            rkllm_model.default_temperature = args.default_temperature

        if args.default_top_p:
            config.default_top_p = args.default_top_p
            rkllm_model.default_top_p = args.default_top_p

        if args.default_top_k:
            config.default_top_k = args.default_top_k
            rkllm_model.default_top_k = args.default_top_k

        if args.default_max_tokens:
            config.default_max_tokens = args.default_max_tokens
            rkllm_model.default_max_tokens = args.default_max_tokens

        if args.max_concurrent:
            config.max_concurrent_requests = args.max_concurrent

        rkllm_model.initialize()

        if not rkllm_model.initialized:
            raise RuntimeError("RKLLM model did not initialize")

        display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        print("=" * 60)
        print("✅ API is ready and listening")
        print(f"  OpenAI API:  http://{display_host}:{args.port}/v1")
        print(f"  Ollama API:  http://{display_host}:{args.port}/api")
        print(f"  API docs:    http://{display_host}:{args.port}/docs")
        print("  Terminal chat: enabled (type /help for commands)")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        print("Please check:")
        print("1. Model file exists and is accessible")
        print("2. RKLLM runtime is properly installed")
        print("3. OpenCL drivers are installed")
        raise

    yield

    # Shutdown
    print("\nShutting down server...")

    # Clean up request states
    request_states.clear()

    # Shutdown thread pool
    if executor:
        executor.shutdown(wait=False)
        print("✅ Thread pool shut down")

    # Release model
    if rkllm_model:
        rkllm_model.release()

# ==================== FastAPI Application ====================
app = FastAPI(
    title="RKLLM OpenAI API Server",
    version="1.0.0",
    description="OpenAI API compatible server for RKLLM models",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API Endpoints ====================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RKLLM OpenAI API Server",
        "status": "running",
        "model": api_model_name,
        "platform": args.target_platform,
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Server information",
            "GET /health": "Health check",
            "GET /v1/models": "List models",
            "POST /v1/chat/completions": "Chat completion"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if rkllm_model and rkllm_model.initialized else "unhealthy",
        "model_initialized": rkllm_model.initialized if rkllm_model else False,
        "active_requests": active_requests,
        "max_concurrent": config.max_concurrent_requests,
        "timestamp": int(time.time())
    }

@app.get("/v1/models", response_model=ModelsListResponse)
async def list_models():
    """List available models"""
    return ModelsListResponse(
        data=[
            ModelInfo(
                id=api_model_name,
                created=int(time.time()),
                owned_by="rkllm"
            )
        ]
    )

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion - Fully OpenAI API compatible"""
    reserve_request_slot()

    try:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        print(f"[{request_id}] New request: stream={request.stream}, messages={len(request.messages)}")
        if request.top_k is not None:
            print(f"[{request_id}] Top-k parameter: {request.top_k}")

        if request.stream:
            # Streaming response
            async def generate_stream():
                nonlocal request_id

                try:
                    # Submit task to thread pool
                    future = executor.submit(process_chat_completion, request, request_id)

                    # Send initial message - FIXED: use model_dump_json() instead of json()
                    initial_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(role="assistant"),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {initial_chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"

                    # Stream results
                    start_time = time.time()
                    last_activity = start_time

                    while True:
                        if request_id in request_states:
                            req_state = request_states[request_id]

                            with req_state.lock:
                                if req_state.text_queue:
                                    for text in req_state.text_queue:
                                        chunk = ChatCompletionStreamResponse(
                                            id=request_id,
                                            created=created,
                                            model=request.model,
                                            choices=[
                                                ChatCompletionStreamResponseChoice(
                                                    index=0,
                                                    delta=DeltaMessage(content=text),
                                                    finish_reason=None
                                                )
                                            ]
                                        )
                                        # FIXED: use model_dump_json() instead of json()
                                        yield f"data: {chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"
                                        last_activity = time.time()

                                    # Clear sent text
                                    req_state.text_queue.clear()

                            # Check if completed
                            if req_state.completed.is_set():
                                if req_state.error:
                                    error_data = {
                                        "error": {
                                            "message": req_state.error,
                                            "type": "server_error"
                                        }
                                    }
                                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                                break

                        # Check timeout
                        if time.time() - last_activity > 30:  # 30 seconds no activity
                            print(f"[{request_id}] Streaming response timeout")
                            break

                        # Brief wait
                        await asyncio.sleep(0.05)

                    # Send completion marker - FIXED: use model_dump_json() instead of json()
                    done_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=0,
                                delta=DeltaMessage(),
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {done_chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

                except Exception as e:
                    print(f"[{request_id}] Stream generation error: {e}")
                    error_data = {
                        "error": {
                            "message": str(e),
                            "type": "server_error"
                        }
                    }
                    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                finally:
                    # Clean up request state
                    if request_id in request_states:
                        del request_states[request_id]
                    release_request_slot()

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        else:
            # Non-streaming response
            def process_non_stream():
                nonlocal request_id

                try:
                    req_state = process_chat_completion(request, request_id)

                    if req_state.error:
                        raise HTTPException(status_code=500, detail=req_state.error)

                    # Estimate token usage
                    prompt_tokens = estimate_tokens(build_prompt(request.messages))
                    completion_tokens = estimate_tokens(req_state.full_response)

                    # Build response
                    response = ChatCompletionResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionResponseChoice(
                                index=0,
                                message=ChatMessage(
                                    role="assistant",
                                    content=req_state.full_response
                                ),
                                finish_reason="stop"
                            )
                        ],
                        usage=UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens
                        )
                    )

                    return response
                finally:
                    # Clean up request state
                    if request_id in request_states:
                        del request_states[request_id]

            # Execute in thread pool
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor, process_non_stream
                )
                return response
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Streaming requests release their slot when the generator exits.
        if not request.stream:
            release_request_slot()

async def generate_ollama_stream(request: ChatCompletionRequest, request_id: str):
    """Stream newline-delimited Ollama messages from the common RKLLM state."""
    try:
        executor.submit(process_chat_completion, request, request_id)
        last_activity = time.time()

        while True:
            req_state = request_states.get(request_id)
            if req_state:
                with req_state.lock:
                    pending = list(req_state.text_queue)
                    req_state.text_queue.clear()
                    completed = req_state.completed.is_set()
                    error = req_state.error
                    full_response = req_state.full_response

                for text in pending:
                    yield ollama_chunk(request.model, text)
                    last_activity = time.time()

                if completed:
                    if error:
                        yield json.dumps({"error": error}, ensure_ascii=False) + "\n"
                    else:
                        yield ollama_chunk(
                            request.model,
                            "",
                            done=True,
                            prompt_tokens=estimate_tokens(build_prompt(request.messages)),
                            completion_tokens=estimate_tokens(full_response),
                        )
                    break

            if time.time() - last_activity > config.timeout_seconds:
                yield json.dumps({"error": "Inference timeout"}, ensure_ascii=False) + "\n"
                break
            await asyncio.sleep(0.05)
    except Exception as error:
        yield json.dumps({"error": str(error)}, ensure_ascii=False) + "\n"
    finally:
        request_states.pop(request_id, None)
        release_request_slot()

@app.post("/api/chat")
async def ollama_chat(request: OllamaChatRequest):
    """Ollama-compatible chat endpoint backed by the RKLLM model."""
    internal_request = ollama_request_from_chat(request)
    reserve_request_slot()
    request_id = f"ollama-{uuid.uuid4().hex[:24]}"

    if request.stream:
        return StreamingResponse(
            generate_ollama_stream(internal_request, request_id),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        req_state = await asyncio.get_event_loop().run_in_executor(
            executor, process_chat_completion, internal_request, request_id
        )
        if req_state.error:
            raise HTTPException(status_code=500, detail=req_state.error)
        return {
            "model": request.model,
            "created_at": ollama_created_at(),
            "message": {"role": "assistant", "content": req_state.full_response},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": estimate_tokens(build_prompt(request.messages)),
            "eval_count": estimate_tokens(req_state.full_response),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        request_states.pop(request_id, None)
        release_request_slot()

@app.post("/api/generate")
async def ollama_generate(request: OllamaGenerateRequest):
    """Ollama-compatible single-prompt generation endpoint."""
    internal_request = ollama_request_from_generate(request)
    # /api/generate has the same response contract as /api/chat, but receives
    # one prompt instead of a messages array.
    chat_request = OllamaChatRequest(
        model=request.model,
        messages=internal_request.messages,
        stream=request.stream,
        options=request.options,
        keep_alive=request.keep_alive,
    )
    if request.stream:
        reserve_request_slot()
        request_id = f"ollama-{uuid.uuid4().hex[:24]}"

        async def generate_stream():
            async for line in generate_ollama_stream(internal_request, request_id):
                payload = json.loads(line)
                if "message" in payload:
                    payload["response"] = payload["message"].get("content", "")
                    payload.pop("message", None)
                yield json.dumps(payload, ensure_ascii=False) + "\n"

        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )

    response = await ollama_chat(chat_request)

    # Ollama's generate response uses "response" instead of "message".
    payload = response
    payload["response"] = payload["message"].pop("content")
    payload.pop("message", None)
    return payload

@app.get("/api/tags")
async def ollama_tags():
    """List the local model using Ollama's model discovery shape."""
    model_name = api_model_name
    return {
        "models": [{
            "name": model_name,
            "model": model_name,
            "modified_at": ollama_created_at(),
            "size": 0,
            "digest": "rkllm",
            "details": {"family": "rkllm", "parameter_size": "device"},
        }]
    }

def run_interactive_chat(base_url: str, model: str) -> None:
    """Run a persistent, history-aware terminal chat for local demonstrations."""
    try:
        import requests
    except ImportError:
        print("⚠ Terminal chat is unavailable because the 'requests' package is missing.")
        print("  Install it with: pip install requests")
        return

    messages: List[Dict[str, str]] = []
    print("\n💬 Chat demo ready")
    print("   This is one continuous conversation: previous turns are sent with each request.")
    print("   Commands: /reset starts a new conversation, /help shows commands, /exit quits.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Leaving chat")
            return

        if not user_text:
            continue
        if user_text in ("/exit", "/quit"):
            print("👋 Leaving chat")
            return
        if user_text == "/help":
            print("Commands: /reset, /help, /exit")
            continue
        if user_text == "/reset":
            messages.clear()
            print("↻ Conversation reset")
            continue

        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": config.default_temperature,
            "top_p": config.default_top_p,
            "max_tokens": config.default_max_tokens,
        }

        answer_parts = []
        print("Assistant: ", end="", flush=True)
        try:
            with requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=(10, config.timeout_seconds + 30),
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line[5:].strip() if raw_line.startswith("data:") else raw_line
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    error = chunk.get("error")
                    if error:
                        raise RuntimeError(error.get("message", str(error)))
                    choices = chunk.get("choices", [])
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if content:
                        answer_parts.append(content)
                        print(content, end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": "".join(answer_parts)})
        except Exception as error:
            # Do not leave a failed user turn in the persistent conversation.
            messages.pop()
            print(f"\n✗ Chat request failed: {error}")

# ==================== Main Program ====================
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="RKLLM OpenAI and Ollama API Compatible Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rkllm_server.py --rkllm_model_path ../model/model.rkllm --target_platform rk3588
  python rkllm_server.py --rkllm_model_path ../model/model.rkllm --target_platform rk3588 \\
                         --port 8001 --max_concurrent 2 --default_temperature 0.7 --default_top_k 50
        """
    )

    # Required arguments
    parser.add_argument('--rkllm_model_path', type=str, required=True,
                       help='Path to RKLLM model file (absolute path)')
    parser.add_argument('--target_platform', type=str, required=True,
                       choices=['rk3588', 'rk3576', 'rk3588s'],
                       help='Target platform: rk3588, rk3576, rk3588s')

    # Server parameters
    parser.add_argument('--port', type=int, default=8001,
                       help='Server port (default: 8001)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host (default: 0.0.0.0)')
    parser.add_argument('--model_name', type=str, default='rkllm-model',
                       help='Model name returned by the OpenAI/Ollama APIs (default: rkllm-model)')
    parser.add_argument('--api_format', type=str, default=os.environ.get('API_FORMAT', 'both').lower(),
                       choices=['openai', 'ollama', 'both'],
                       help='Compatibility option retained for the container interface')
    parser.add_argument('--no_chat', action='store_true',
                       help='Start the API server without opening the terminal chat')

    # Model parameters
    parser.add_argument('--max_context_len', type=int, default=2048,
                       help='Maximum context length (default: 2048)')
    parser.add_argument('--default_temperature', type=float, default=0.8,
                       help='Default temperature parameter (default: 0.8)')
    parser.add_argument('--default_top_p', type=float, default=0.9,
                       help='Default top_p parameter (default: 0.9)')
    parser.add_argument('--default_top_k', type=int, default=1,
                       help='Default top_k parameter (default: 1, range: 1-100)')
    parser.add_argument('--default_max_tokens', type=int, default=512,
                       help='Default maximum tokens to generate (default: 512)')

    # Performance parameters
    parser.add_argument('--max_concurrent', type=int, default=2,
                       help='Maximum concurrent requests (default: 2)')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Request timeout in seconds (default: 120)')

    # Debug parameters
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    args = parser.parse_args()

    # Validate model file
    if not os.path.exists(args.rkllm_model_path):
        print(f"❌ Error: Model file not found: {args.rkllm_model_path}")
        sys.exit(1)

    # Validate top_k range
    if args.default_top_k < 1 or args.default_top_k > 100:
        print(f"⚠ Warning: default_top_k should be between 1 and 100, got {args.default_top_k}")
        args.default_top_k = max(1, min(100, args.default_top_k))
        print(f"  Adjusted to: {args.default_top_k}")

    # Convert to absolute path
    args.rkllm_model_path = os.path.abspath(args.rkllm_model_path)

    # Apply configuration
    api_model_name = os.environ.get('MODEL_ID', args.model_name)
    config.max_context_len = args.max_context_len
    config.default_temperature = args.default_temperature
    config.default_top_p = args.default_top_p
    config.default_top_k = args.default_top_k
    config.default_max_tokens = args.default_max_tokens
    config.max_concurrent_requests = args.max_concurrent
    config.timeout_seconds = args.timeout

    # Print configuration
    print("=" * 60)
    print("Configuration:")
    print(f"  Model: {args.rkllm_model_path}")
    print(f"  Platform: {args.target_platform}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Max context length: {config.max_context_len}")
    print(f"  Default temperature: {config.default_temperature}")
    print(f"  Default Top-p: {config.default_top_p}")
    print(f"  Default Top-k: {config.default_top_k}")
    print(f"  Default max tokens: {config.default_max_tokens}")
    print(f"  Max concurrent requests: {config.max_concurrent_requests}")
    print(f"  Request timeout: {config.timeout_seconds}s")
    print("=" * 60)

    # Start server. Running Uvicorn in a background thread lets this process
    # keep the API available while the main thread owns the terminal chat.
    server = None
    server_thread = None
    try:
        uvicorn_config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info" if not args.debug else "debug",
            access_log=True,
            timeout_keep_alive=30,
            server_header=False,
        )
        server = uvicorn.Server(uvicorn_config)
        # Signal handlers can only be installed by the main thread. The main
        # thread is intentionally reserved for input(), so disable Uvicorn's
        # signal installation and handle Ctrl-C around the chat loop instead.
        server.install_signal_handlers = lambda: None
        server_thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
        server_thread.start()

        deadline = time.time() + max(30, config.timeout_seconds)
        while not server.started and server_thread.is_alive() and time.time() < deadline:
            time.sleep(0.1)

        if not server.started:
            raise RuntimeError("API server did not finish starting")

        display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        if not args.no_chat:
            run_interactive_chat(f"http://{display_host}:{args.port}", args.model_name)
        else:
            print("API server is running. Press Ctrl-C to stop it.")
            while server_thread.is_alive():
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Server interrupted by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)
