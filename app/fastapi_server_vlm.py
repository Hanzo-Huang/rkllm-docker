#!/usr/bin/env python3
"""
OpenAI-compatible RKLLM Vision Language Model server.

This server uses Rockchip's official runtime APIs directly:

* librknnrt.so runs the vision encoder and produces image embeddings.
* librkllmrt.so runs the multimodal RKLLM model.

The ctypes structures below match the RKLLM v1.3.0 runtime; no
project-specific runtime wrapper is required.
"""

import argparse
import asyncio
import base64
import ctypes
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class OutputDelta:
    """One protocol-visible piece of generated output."""
    channel: str
    text: str


class ThinkingStreamParser:
    """Separate reasoning from the answer across arbitrary callback chunks."""

    START_MARKERS = (
        "<thinking>",
        "<think>",
        "Thinking Process:",
        "Okay, so I need",
        "Okay, I need",
        "Let me think",
        "Let's think",
        "The user wants",
        "I need to",
        "Let's analyze",
        "We need to",
    )
    UNMARKED_START_MARKERS = (
        "Okay, so I need",
        "Okay, I need",
        "Let me think",
        "Let's think",
        "The user wants",
        "I need to",
        "Let's analyze",
        "We need to",
    )
    END_MARKERS = (
        "</thinking>",
        "</think>",
        "Final Answer:",
        "Final answer:",
        "*Drafting the response:*",
        "Drafting the response:",
    )

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.in_thinking = False
        self._buffer = ""

    @staticmethod
    def _marker_at(text: str, start: int, markers: tuple[str, ...]) -> Optional[str]:
        matches = [marker for marker in markers if text.startswith(marker, start)]
        return max(matches, key=len) if matches else None

    @staticmethod
    def _partial_suffix(text: str, markers: tuple[str, ...]) -> str:
        for size in range(min(len(text), max(map(len, markers))), 0, -1):
            suffix = text[-size:]
            if any(marker.startswith(suffix) for marker in markers):
                return suffix
        return ""

    def feed(self, text: str) -> List[OutputDelta]:
        if not text:
            return []
        self._buffer += text
        deltas: List[OutputDelta] = []
        while self._buffer:
            markers = self.END_MARKERS if self.in_thinking else self.START_MARKERS + self.END_MARKERS
            found = None
            for index in range(len(self._buffer)):
                marker = self._marker_at(self._buffer, index, markers)
                if marker:
                    found = (index, marker)
                    break
            if found:
                index, marker = found
                orphan_end = not self.in_thinking and marker in self.END_MARKERS
                unmarked_start = (
                    not self.in_thinking and marker in self.UNMARKED_START_MARKERS
                )
                channel = (
                    "reasoning_content"
                    if self.in_thinking or orphan_end or unmarked_start
                    else "content"
                )
                value = self._buffer[:index]
                if unmarked_start:
                    value = marker + value
                if value and (channel == "content" or self.enabled):
                    deltas.append(OutputDelta(channel, value))
                self._buffer = self._buffer[index + len(marker):]
                self.in_thinking = (
                    False
                    if orphan_end
                    else True
                    if unmarked_start
                    else not self.in_thinking
                )
                continue
            markers_suffix = self._partial_suffix(self._buffer, markers)
            value = self._buffer[:-len(markers_suffix)] if markers_suffix else self._buffer
            channel = "reasoning_content" if self.in_thinking else "content"
            if value and (channel == "content" or self.enabled):
                deltas.append(OutputDelta(channel, value))
            self._buffer = markers_suffix
            break
        return deltas

    def finish(self) -> List[OutputDelta]:
        if not self._buffer:
            return []
        channel = "reasoning_content" if self.in_thinking else "content"
        value = self._buffer
        self._buffer = ""
        if channel == "reasoning_content" and not self.enabled:
            return []
        return [OutputDelta(channel, value)]


def _thinking_enabled(
    enable_thinking: Optional[bool] = None,
    thinking: Optional[Union[bool, Dict[str, Any]]] = None,
    reasoning_effort: Optional[str] = None,
) -> bool:
    if enable_thinking is not None:
        return bool(enable_thinking)
    if isinstance(thinking, bool):
        return thinking
    if isinstance(thinking, dict):
        return str(thinking.get("type", "enabled")).lower() not in {
            "disabled", "disable", "off", "none", "false"
        }
    if reasoning_effort is not None:
        return str(reasoning_effort).lower() not in {"none", "off", "disabled", "false"}
    return False


def model_supports_reasoning(model_name: str) -> bool:
    name = str(model_name).lower()
    return any(family in name for family in ("qwen3", "qwen-3", "deepseek-r1", "deepseek_r1"))


def model_supports_thinking(model_name: str) -> bool:
    """Return whether RKLLM can switch thinking on/off for this model family."""
    name = str(model_name).lower()
    return any(family in name for family in ("qwen3", "qwen-3"))


def runtime_max_tokens(model_name: str, requested: int, thinking: bool) -> int:
    """Reserve hidden generation room for reasoning-only model artifacts."""
    if model_supports_reasoning(model_name):
        return min(4096, requested + max(512, requested * 2))
    return requested


LOG_LEVELS = ("critical", "error", "warning", "info", "debug")


def normalize_log_level(value: str) -> str:
    """Return a logging level accepted by both Python logging and Uvicorn."""
    level = str(value).strip().lower()
    if level == "warn":
        level = "warning"
    if level not in LOG_LEVELS:
        valid_levels = ", ".join(LOG_LEVELS)
        raise ValueError(f"invalid log level {value!r}; use one of: {valid_levels}")
    return level


try:
    initial_log_level = normalize_log_level(os.environ.get("LOG_LEVEL", "info"))
except ValueError:
    initial_log_level = "info"

logging.basicConfig(
    level=getattr(logging, initial_log_level.upper()),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rkllm")


def preload_libraries() -> None:
    """Load Rockchip runtimes globally so their dependent symbols resolve."""
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:/usr/lib:" + os.environ.get(
        "LD_LIBRARY_PATH", ""
    )
    for library in ("librknnrt.so", "/usr/lib/librkllmrt.so"):
        try:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            logger.info("Loaded %s", library)
        except OSError as error:
            logger.warning("Could not preload %s: %s", library, error)


preload_libraries()


# ==================== OpenAI request/response models ====================
class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    model: str = "rkllm-vision"
    messages: List[Message]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(1, ge=1, le=100)
    n: Optional[int] = Field(1, ge=1, le=10)
    stream: Optional[bool] = False
    max_tokens: Optional[int] = Field(512, ge=1, le=4096)
    max_completion_tokens: Optional[int] = Field(None, ge=1, le=4096)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None
    max_context_len: Optional[int] = Field(2048, ge=512, le=8192)
    rknn_core_num: Optional[int] = Field(3, ge=1, le=4)
    # Common OpenAI-compatible client controls.  Cherry Studio may send any
    # of these when its reasoning/thinking control is enabled.
    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    thinking: Optional[Union[bool, Dict[str, Any]]] = None
    repeat_penalty: Optional[float] = Field(1.1, ge=0.0, le=2.0)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str = ""
    reasoning_content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: AssistantMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = "fp_rkllm_vision"


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = "fp_rkllm_vision"


# ==================== Official RKNN v2 C API structures ====================
class RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RknnInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class RknnOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class RKNNImageEncoder:
    """Direct wrapper around the official RKNN image encoder API."""

    RKNN_QUERY_IN_OUT_NUM = 0
    RKNN_QUERY_INPUT_ATTR = 1
    RKNN_QUERY_OUTPUT_ATTR = 2
    RKNN_TENSOR_NCHW = 0
    RKNN_TENSOR_NHWC = 1
    RKNN_TENSOR_UINT8 = 3

    def __init__(self, model_path: str, core_num: int):
        self.model_path = model_path
        self.lib = ctypes.CDLL("/usr/lib/librknnrt.so", mode=ctypes.RTLD_GLOBAL)
        self.ctx = ctypes.c_uint64(0)
        self.lock = threading.Lock()
        self._setup_signatures()

        with open(model_path, "rb") as model_file:
            model_data = model_file.read()
        self.model_data = ctypes.create_string_buffer(model_data)
        ret = self.lib.rknn_init(
            ctypes.byref(self.ctx),
            ctypes.cast(self.model_data, ctypes.c_void_p),
            len(model_data),
            0,
            None,
        )
        if ret != 0:
            raise RuntimeError(f"rknn_init failed with code {ret}")

        if hasattr(self.lib, "rknn_set_core_mask"):
            core_mask = {1: 1, 2: 3, 3: 7}.get(core_num, 0)
            self.lib.rknn_set_core_mask(self.ctx, core_mask)

        self.input_attr = self._query_attr(self.RKNN_QUERY_INPUT_ATTR)
        self.output_attrs = [
            self._query_attr(self.RKNN_QUERY_OUTPUT_ATTR, index)
            for index in range(self.io_num.n_output)
        ]

        input_dims = list(self.input_attr.dims[: self.input_attr.n_dims])
        if self.input_attr.fmt == self.RKNN_TENSOR_NCHW:
            self.channels, self.height, self.width = input_dims[-3:]
        else:
            self.height, self.width, self.channels = input_dims[-3:]

        output_dims = list(self.output_attrs[0].dims[: self.output_attrs[0].n_dims])
        self.image_tokens, self.embed_size = self._find_embedding_shape(output_dims)
        logger.info(
            "RKNN encoder ready: %sx%s, %s image tokens, embedding size %s, %s outputs",
            self.width,
            self.height,
            self.image_tokens,
            self.embed_size,
            self.io_num.n_output,
        )

    def _setup_signatures(self) -> None:
        self.lib.rknn_init.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.lib.rknn_init.restype = ctypes.c_int
        self.lib.rknn_destroy.argtypes = [ctypes.c_uint64]
        self.lib.rknn_destroy.restype = ctypes.c_int
        self.lib.rknn_query.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.lib.rknn_query.restype = ctypes.c_int
        self.lib.rknn_inputs_set.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(RknnInput),
        ]
        self.lib.rknn_inputs_set.restype = ctypes.c_int
        self.lib.rknn_run.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
        self.lib.rknn_run.restype = ctypes.c_int
        self.lib.rknn_outputs_get.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(RknnOutput),
            ctypes.c_void_p,
        ]
        self.lib.rknn_outputs_get.restype = ctypes.c_int
        self.lib.rknn_outputs_release.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(RknnOutput),
        ]
        self.lib.rknn_outputs_release.restype = ctypes.c_int
        if hasattr(self.lib, "rknn_set_core_mask"):
            self.lib.rknn_set_core_mask.argtypes = [ctypes.c_uint64, ctypes.c_int]
            self.lib.rknn_set_core_mask.restype = ctypes.c_int

    def _query_attr(self, query: int, index: int = 0) -> RknnTensorAttr:
        attr = RknnTensorAttr()
        attr.index = index
        ret = self.lib.rknn_query(
            self.ctx, query, ctypes.byref(attr), ctypes.sizeof(attr)
        )
        if ret != 0:
            raise RuntimeError(f"rknn_query failed with code {ret}")
        if query == self.RKNN_QUERY_IN_OUT_NUM:
            return attr
        return attr

    @property
    def io_num(self) -> RknnInputOutputNum:
        if not hasattr(self, "_io_num"):
            self._io_num = RknnInputOutputNum()
            ret = self.lib.rknn_query(
                self.ctx,
                self.RKNN_QUERY_IN_OUT_NUM,
                ctypes.byref(self._io_num),
                ctypes.sizeof(self._io_num),
            )
            if ret != 0:
                raise RuntimeError(f"rknn_query I/O count failed with code {ret}")
        return self._io_num

    @staticmethod
    def _find_embedding_shape(dims: List[int]) -> Tuple[int, int]:
        for index, dimension in enumerate(dims[:-1]):
            if dimension > 1 and dims[index + 1] > 1:
                return dimension, dims[index + 1]
        raise RuntimeError(f"Could not infer image embedding shape from {dims}")

    def encode(self, image_bytes: bytes) -> np.ndarray:
        with self.lock:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            size = max(image.width, image.height)
            square = Image.new("RGB", (size, size), (127, 127, 127))
            square.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
            image = square.resize((self.width, self.height), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.uint8).copy()

            if self.input_attr.fmt == self.RKNN_TENSOR_NCHW:
                pixels = np.transpose(pixels, (2, 0, 1)).copy()

            input_info = RknnInput()
            input_info.index = 0
            input_info.buf = pixels.ctypes.data_as(ctypes.c_void_p)
            input_info.size = pixels.nbytes
            input_info.pass_through = 0
            input_info.type = self.RKNN_TENSOR_UINT8
            input_info.fmt = self.input_attr.fmt

            ret = self.lib.rknn_inputs_set(self.ctx, 1, ctypes.byref(input_info))
            if ret != 0:
                raise RuntimeError(f"rknn_inputs_set failed with code {ret}")
            ret = self.lib.rknn_run(self.ctx, None)
            if ret != 0:
                raise RuntimeError(f"rknn_run failed with code {ret}")

            outputs = (RknnOutput * self.io_num.n_output)()
            for index in range(self.io_num.n_output):
                outputs[index].want_float = 1

            ret = self.lib.rknn_outputs_get(self.ctx, self.io_num.n_output, outputs, None)
            if ret != 0:
                raise RuntimeError(f"rknn_outputs_get failed with code {ret}")

            try:
                output_arrays = []
                for index in range(self.io_num.n_output):
                    count = outputs[index].size // ctypes.sizeof(ctypes.c_float)
                    pointer = ctypes.cast(outputs[index].buf, ctypes.POINTER(ctypes.c_float))
                    output_arrays.append(np.ctypeslib.as_array(pointer, shape=(count,)).copy())

                if len(output_arrays) == 1:
                    return output_arrays[0].astype(np.float32, copy=False)

                # Match Rockchip's official multi-output interleaving logic.
                result = np.empty(
                    self.image_tokens * self.io_num.n_output * self.embed_size,
                    dtype=np.float32,
                )
                for token in range(self.image_tokens):
                    for output_index, output in enumerate(output_arrays):
                        source_start = token * self.embed_size
                        result_start = (
                            token * self.io_num.n_output * self.embed_size
                            + output_index * self.embed_size
                        )
                        result[result_start : result_start + self.embed_size] = output[
                            source_start : source_start + self.embed_size
                        ]
                return result
            finally:
                self.lib.rknn_outputs_release(self.ctx, self.io_num.n_output, outputs)

    def close(self) -> None:
        if self.ctx.value:
            self.lib.rknn_destroy(self.ctx)
            self.ctx.value = 0


# ==================== Official RKLLM v1.3.0 C API structures ====================
#
# Keep these definitions in sync with rkllm.h. v1.3.0 changed the ABI from
# v1.2.3: RKLLMParam gained ignore_eos_token and no longer contains the image
# marker strings; those strings now belong to the multimodal image input.
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
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMSamplingParam(ctypes.Structure):
    """Per-request sampling override introduced by RKLLM v1.3.0."""
    _fields_ = [
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int32)]


LLMResultCallbackType = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(RKLLMResult),
    ctypes.c_void_p,
    ctypes.c_int,
)


class RKLLMCallback(ctypes.Structure):
    _fields_ = [
        ("result_callback", LLMResultCallbackType),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", ctypes.c_void_p),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", ctypes.c_void_p),
        ("embed_userdata", ctypes.c_void_p),
    ]


class InferenceState:
    def __init__(self, thinking: bool = False):
        self.text_queue: List[OutputDelta] = []
        self.full_response = ""
        self.reasoning_response = ""
        self.finish_reason = "stop"
        self.error: Optional[str] = None
        self.completed = threading.Event()
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.parser = ThinkingStreamParser(thinking)

    def append_runtime_text(self, text: str) -> None:
        for marker in ("<｜end▁of▁sentence｜>", "<|endoftext|>", "<|im_end|>"):
            text = text.replace(marker, "")
        for delta in self.parser.feed(text):
            self.text_queue.append(delta)
            if delta.channel == "reasoning_content":
                self.reasoning_response += delta.text
            else:
                self.full_response += delta.text

    def finish_output(self) -> None:
        if self.parser.in_thinking:
            self.finish_reason = "length"
        for delta in self.parser.finish():
            self.text_queue.append(delta)
            if delta.channel == "reasoning_content":
                self.reasoning_response += delta.text
            else:
                self.full_response += delta.text


request_states: Dict[str, InferenceState] = {}
request_lock = threading.Lock()
active_requests = 0
executor: Optional[ThreadPoolExecutor] = None
runtime = None


class RKLLMRuntime:
    """Direct wrapper around the official RKLLM v1.3.0 runtime."""

    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_MULTIMODAL = 3
    RKLLM_INFER_GENERATE = 0
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

    def __init__(self, model_path: str, platform: str, config: "ServerConfig"):
        self.lib = ctypes.CDLL("/usr/lib/librkllmrt.so", mode=ctypes.RTLD_GLOBAL)
        self.handle = ctypes.c_void_p()
        self.lock = threading.Lock()
        self.config = config
        self.platform = platform.lower()
        self._setup_signatures()

        self._callback = LLMResultCallbackType(self._callback_impl)
        self._callback_config = RKLLMCallback()
        self._callback_config.result_callback = self._callback
        self._callback_config.result_userdata = None
        self._callback_config.tokenizer_callback = None
        self._callback_config.tokenizer_userdata = None
        self._callback_config.embed_callback = None
        self._callback_config.embed_userdata = None

        params = RKLLMParam()
        params.model_path = model_path.encode()
        params.max_context_len = config.max_context_len
        params.max_new_tokens = config.default_max_tokens
        params.top_k = config.default_top_k
        params.n_keep = 0
        params.top_p = config.default_top_p
        params.temperature = config.default_temperature
        params.repeat_penalty = 1.1
        params.frequency_penalty = 0.0
        params.presence_penalty = 0.0
        params.mirostat = 0
        params.mirostat_tau = 5.0
        params.mirostat_eta = 0.1
        # Keep <think>/</think> markers visible to the protocol parser.
        params.skip_special_token = False
        params.ignore_eos_token = False
        params.is_async = False
        params.extend_param.base_domain_id = (
            1 if self.platform in {"rk3576", "rk3588", "rk3588s"} else 0
        )
        params.extend_param.embed_flash = 0
        params.extend_param.enabled_cpus_num = 4
        params.extend_param.enabled_cpus_mask = 0xF0
        params.extend_param.n_batch = 1
        params.extend_param.use_cross_attn = 0

        ret = self.lib.rkllm_init(
            ctypes.byref(self.handle),
            ctypes.byref(params),
            ctypes.byref(self._callback_config),
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed with code {ret}")
        logger.info("RKLLM runtime initialized with official librkllmrt.so")

    def _setup_signatures(self) -> None:
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(RKLLMParam),
            ctypes.POINTER(RKLLMCallback),
        ]
        self.lib.rkllm_init.restype = ctypes.c_int
        self.lib.rkllm_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self.lib.rkllm_run.restype = ctypes.c_int
        self.lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_destroy.restype = ctypes.c_int

    def _callback_impl(self, result_ptr, userdata, state_code):
        if not userdata:
            return 0
        try:
            request_id = ctypes.cast(userdata, ctypes.c_char_p).value.decode()
            state = request_states.get(request_id)
            if state is None:
                return 0
            with state.lock:
                if state_code == self.RKLLM_RUN_NORMAL and result_ptr and result_ptr.contents.text:
                    text = result_ptr.contents.text.decode("utf-8", errors="ignore")
                    state.append_runtime_text(text)
                elif state_code == self.RKLLM_RUN_ERROR:
                    state.error = "RKLLM runtime error"
                    state.completed.set()
                elif state_code == self.RKLLM_RUN_FINISH:
                    state.finish_output()
                    state.completed.set()
            return 0
        except Exception as error:
            logger.exception("RKLLM callback failed: %s", error)
            return -1

    def run(
        self,
        request_id: str,
        prompt: str,
        image_embeddings: Optional[np.ndarray],
        image_width: int = 0,
        image_height: int = 0,
        enable_thinking: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        max_tokens: Optional[int] = None,
    ) -> InferenceState:
        state = request_states[request_id]
        prompt_buffer = ctypes.create_string_buffer(prompt.encode("utf-8"))
        request_id_buffer = ctypes.create_string_buffer(request_id.encode("utf-8"))
        input_data = RKLLMInput()
        input_data.role = b"user"
        input_data.enable_thinking = enable_thinking

        if image_embeddings is None:
            input_data.input_type = self.RKLLM_INPUT_PROMPT
            input_data.input_data.prompt_input = ctypes.cast(prompt_buffer, ctypes.c_char_p)
        else:
            image_embeddings = np.ascontiguousarray(image_embeddings, dtype=np.float32)
            multimodal = RKLLMMultiModalInput()
            multimodal.prompt = ctypes.cast(prompt_buffer, ctypes.c_char_p)
            multimodal.image.image_embed = image_embeddings.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )
            multimodal.image.n_image_tokens = runtime.encoder.image_tokens
            multimodal.image.n_image = 1
            image_start = ctypes.create_string_buffer(config.img_start.encode("utf-8"))
            image_end = ctypes.create_string_buffer(config.img_end.encode("utf-8"))
            image_content = ctypes.create_string_buffer(config.img_content.encode("utf-8"))
            multimodal.image.image_start = ctypes.cast(image_start, ctypes.c_char_p)
            multimodal.image.image_end = ctypes.cast(image_end, ctypes.c_char_p)
            multimodal.image.image_content = ctypes.cast(image_content, ctypes.c_char_p)
            multimodal.image.image_width = image_width
            multimodal.image.image_height = image_height
            input_data.input_type = self.RKLLM_INPUT_MULTIMODAL
            input_data.input_data.multimodal_input = multimodal

        infer_params = RKLLMInferParam()
        infer_params.mode = self.RKLLM_INFER_GENERATE
        infer_params.lora_params = None
        infer_params.prompt_cache_params = None
        sampling_params = RKLLMSamplingParam()
        sampling_params.top_k = top_k or self.config.default_top_k
        sampling_params.top_p = top_p if top_p is not None else self.config.default_top_p
        sampling_params.temperature = (
            temperature if temperature is not None else self.config.default_temperature
        )
        sampling_params.repeat_penalty = repeat_penalty
        sampling_params.frequency_penalty = frequency_penalty
        sampling_params.presence_penalty = presence_penalty
        sampling_params.mirostat = 0
        sampling_params.mirostat_tau = 5.0
        sampling_params.mirostat_eta = 0.1
        infer_params.sampling_params = ctypes.pointer(sampling_params)
        infer_params.keep_history = 0
        infer_params.max_new_tokens = max_tokens or 0

        with self.lock:
            ret = self.lib.rkllm_run(
                self.handle,
                ctypes.byref(input_data),
                ctypes.byref(infer_params),
                ctypes.cast(request_id_buffer, ctypes.c_void_p),
            )
        if ret != 0 and not state.error:
            state.error = f"rkllm_run failed with code {ret}"
        state.completed.set()
        return state

    def close(self) -> None:
        if self.handle:
            self.lib.rkllm_destroy(self.handle)
            self.handle = None


class ServerConfig:
    def __init__(self):
        self.model_name = os.environ.get("API_MODEL_NAME") or "rkllm-vision"
        self.encoder_model_path = ""
        self.llm_model_path = ""
        self.platform = "rk3588"
        self.max_context_len = 2048
        self.default_temperature = 0.7
        self.default_top_p = 1.0
        self.default_top_k = 1
        self.default_max_tokens = 512
        self.max_concurrent_requests = 1
        self.timeout_seconds = 300
        self.host = "0.0.0.0"
        self.port = 8001
        self.rknn_core_num = 3
        self.img_start = "<|vision_start|>"
        self.img_end = "<|vision_end|>"
        self.img_content = "<|image_pad|>"


config = ServerConfig()


def platform_npu_core_limit(platform: str) -> int:
    """Return the available RKNN NPU core count for a supported platform."""
    if platform == "rk3576":
        return 2
    if platform in {"rk3588", "rk3588s"}:
        return 3
    return 3


def reserve_request_slot() -> None:
    global active_requests
    with request_lock:
        if active_requests >= config.max_concurrent_requests:
            raise HTTPException(status_code=429, detail="Too many requests")
        active_requests += 1


def release_request_slot() -> None:
    global active_requests
    with request_lock:
        active_requests = max(0, active_requests - 1)


def load_image(url: str) -> bytes:
    if url.startswith("data:"):
        try:
            return base64.b64decode(url.split(",", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError("Invalid base64 image data") from error
    if url.startswith(("http://", "https://")):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    raise ValueError("image_url must be an HTTP(S) URL or a data URL")


def message_parts(message: Message) -> Tuple[str, Optional[str]]:
    if isinstance(message.content, str):
        return message.content, None
    text_parts = []
    image_url = None
    for part in message.content:
        if part.type == "text" and part.text:
            text_parts.append(part.text)
        elif part.type == "image_url" and part.image_url:
            if image_url is not None:
                raise ValueError("Only one image is supported per request")
            image_url = part.image_url.url
    return "".join(text_parts), image_url


def build_prompt(messages: List[Message]) -> Tuple[str, Optional[str]]:
    prompt_parts = []
    image_url = None
    for message in messages:
        text, message_image = message_parts(message)
        if message_image:
            if image_url is not None:
                raise ValueError("Only one image is supported per request")
            image_url = message_image
            text = "<image>" + text
        if message.role == "system":
            prompt_parts.append(f"System: {text}\n")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {text}\n")
        else:
            prompt_parts.append(f"Human: {text}\n")
    prompt = "".join(prompt_parts)
    if not prompt.rstrip().endswith("Assistant:"):
        prompt += "Assistant:"
    return prompt, image_url


def estimate_tokens(text: str) -> int:
    chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return max(1, int(chinese * 1.5 + (len(text) - chinese) * 0.3)) if text else 0


def thinking_enabled(request: ChatCompletionRequest) -> bool:
    """Normalize thinking controls used by OpenAI-compatible clients."""
    enabled = _thinking_enabled(
        request.enable_thinking, request.thinking, request.reasoning_effort
    )
    if (
        request.enable_thinking is None
        and request.thinking is None
        and request.reasoning_effort is None
        and model_supports_reasoning(config.llm_model_path)
    ):
        enabled = True
    return enabled


def execute_inference(
    request_id: str,
    prompt: str,
    image_url: Optional[str],
    enable_thinking: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    max_tokens: Optional[int] = None,
) -> InferenceState:
    state = request_states[request_id]
    started = time.time()
    try:
        image_embeddings = None
        image_width = image_height = 0
        if image_url:
            logger.info("[%s] loading image", request_id)
            image_bytes = load_image(image_url)
            image_embeddings = runtime.encoder.encode(image_bytes)
            image_width = runtime.encoder.width
            image_height = runtime.encoder.height
            logger.info(
                "[%s] image encoded: bytes=%s size=%sx%s tokens=%s",
                request_id,
                len(image_bytes),
                image_width,
                image_height,
                runtime.encoder.image_tokens,
            )
        result = runtime.llm.run(
            request_id,
            prompt,
            image_embeddings,
            image_width,
            image_height,
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
        )
        if result.error:
            logger.error("[%s] request failed: %s", request_id, result.error)
        else:
            logger.info("[%s] request completed in %.2fs", request_id, time.time() - started)
        return result
    except Exception as error:
        state.error = str(error)
        state.completed.set()
        logger.exception("[%s] request failed", request_id)
        return state


def openai_chunk(
    request_id: str,
    created: int,
    model: str,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> str:
    delta_data: Dict[str, str] = {}
    if role is not None:
        delta_data["role"] = role
    if content is not None:
        delta_data["content"] = content
    if reasoning_content is not None:
        delta_data["reasoning_content"] = reasoning_content
    chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(**delta_data),
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json(exclude_unset=True, ensure_ascii=False)}\n\n"


async def stream_completion(request: ChatCompletionRequest, request_id: str, created: int,
                           prompt: str, image_url: Optional[str]):
    try:
        logger.info(
            "[%s] request started: stream=true messages=%s image=%s prompt_chars=%s max_tokens=%s",
            request_id,
            len(request.messages),
            bool(image_url),
            len(prompt),
            request.max_tokens,
        )
        executor.submit(
            execute_inference,
            request_id,
            prompt,
            image_url,
            thinking_enabled(request),
            request.temperature,
            request.top_p,
            request.top_k,
            request.frequency_penalty or 0.0,
            request.presence_penalty or 0.0,
            request.repeat_penalty or 1.1,
            runtime_max_tokens(
                config.llm_model_path,
                request.max_completion_tokens or request.max_tokens,
                enable_thinking,
            ),
        )
        yield openai_chunk(request_id, created, request.model, role="assistant")
        last_activity = time.time()
        while True:
            state = request_states.get(request_id)
            if state:
                with state.lock:
                    pending = list(state.text_queue)
                    state.text_queue.clear()
                    completed = state.completed.is_set()
                    error = state.error
                for delta in pending:
                    yield openai_chunk(
                        request_id,
                        created,
                        request.model,
                        content=delta.text if delta.channel == "content" else None,
                        reasoning_content=(
                            delta.text if delta.channel == "reasoning_content" else None
                        ),
                    )
                    last_activity = time.time()
                if completed:
                    if error:
                        yield f"data: {json.dumps({'error': {'message': error}})}\n\n"
                    else:
                        yield openai_chunk(
                            request_id,
                            created,
                            request.model,
                            finish_reason=state.finish_reason,
                        )
                    yield "data: [DONE]\n\n"
                    break
            if time.time() - last_activity > config.timeout_seconds:
                timeout_error = f"Inference timeout ({config.timeout_seconds}s)"
                logger.error("[%s] streaming response timeout", request_id)
                yield f"data: {json.dumps({'error': {'message': timeout_error}})}\n\n"
                yield "data: [DONE]\n\n"
                break
            await asyncio.sleep(0.05)
    except Exception as error:
        logger.exception("[%s] stream generation failed", request_id)
        yield f"data: {json.dumps({'error': {'message': str(error)}})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        request_states.pop(request_id, None)
        release_request_slot()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime, executor
    executor = ThreadPoolExecutor(
        max_workers=config.max_concurrent_requests + 1,
        thread_name_prefix="rkllm_vlm_worker",
    )
    try:
        runtime = type("OfficialVLMRuntime", (), {})()
        encoder_core_num = config.rknn_core_num
        core_limit = platform_npu_core_limit(config.platform)
        if encoder_core_num > core_limit:
            logger.warning(
                "%s exposes only %s RKNN cores; clamping rknn_core_num=%s to %s",
                config.platform,
                core_limit,
                encoder_core_num,
                core_limit,
            )
            encoder_core_num = core_limit
        runtime.encoder = RKNNImageEncoder(config.encoder_model_path, encoder_core_num)
        runtime.llm = RKLLMRuntime(config.llm_model_path, config.platform, config)
        logger.info(
            "API ready: model=%s platform=%s vision_cores=%s",
            config.model_name,
            config.platform,
            encoder_core_num,
        )
        # Show the bind address in startup logs.  For Docker this is normally
        # 0.0.0.0, which makes it clear the service listens on all interfaces.
        display_host = config.host
        logger.info("OpenAI API: http://%s:%s/v1", display_host, config.port)
        logger.info("API docs: http://%s:%s/docs", display_host, config.port)
        yield
    finally:
        logger.info("Shutting down VLM server")
        request_states.clear()
        if executor:
            executor.shutdown(wait=False)
        if runtime:
            llm = getattr(runtime, "llm", None)
            encoder = getattr(runtime, "encoder", None)
            if llm:
                llm.close()
            if encoder:
                encoder.close()
        runtime = None


app = FastAPI(
    title="RKLLM Vision OpenAI API Server",
    version="2.0.0",
    description="OpenAI-compatible multimodal API using official RKNN and RKLLM runtimes",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "RKLLM Vision OpenAI API Server",
        "status": "running",
        "runtime": "official librknnrt.so + librkllmrt.so",
        "endpoints": {
            "GET /health": "Health check",
            "GET /v1/models": "List models",
            "POST /v1/chat/completions": "OpenAI-compatible text and image chat",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if runtime and runtime.llm.handle else "unhealthy",
        "runtime": "official",
        "active_requests": active_requests,
        "max_concurrent": config.max_concurrent_requests,
        "timestamp": int(time.time()),
    }


@app.get("/v1/models")
async def list_models():
    reasoning_supported = model_supports_reasoning(config.llm_model_path)
    thinking_supported = model_supports_thinking(config.llm_model_path)
    return {
        "object": "list",
        "data": [{
            "id": config.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rockchip",
            # These are optional vendor metadata fields. Standard OpenAI
            # clients ignore unknown fields; clients such as Cherry Studio
            # can use them when capability discovery is supported.
            "capabilities": (
                ["vision"]
                + (["reasoning"] if reasoning_supported else [])
                + (["thinking"] if thinking_supported else [])
            ),
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
            "reasoning": {
                "supported": reasoning_supported,
                "thinking_control": thinking_supported,
            },
        }],
    }


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
)
async def create_chat_completion(request: ChatCompletionRequest):
    if runtime is None:
        raise HTTPException(status_code=503, detail="VLM runtime is not initialized")
    try:
        prompt, image_url = build_prompt(request.messages)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    reserve_request_slot()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    request_states[request_id] = InferenceState(thinking=thinking_enabled(request))
    logger.info(
        "[%s] request accepted: stream=%s messages=%s image=%s model=%s",
        request_id,
        request.stream,
        len(request.messages),
        bool(image_url),
        request.model,
    )

    if request.stream:
        return StreamingResponse(
            stream_completion(request, request_id, created, prompt, image_url),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        state = await asyncio.get_event_loop().run_in_executor(
            executor,
            execute_inference,
            request_id,
            prompt,
            image_url,
            thinking_enabled(request),
            request.temperature,
            request.top_p,
            request.top_k,
            request.frequency_penalty or 0.0,
            request.presence_penalty or 0.0,
            request.repeat_penalty or 1.1,
            runtime_max_tokens(
                config.llm_model_path,
                request.max_completion_tokens or request.max_tokens,
                thinking_enabled(request),
            ),
        )
        if state.error:
            raise HTTPException(status_code=500, detail=state.error)
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(state.full_response + state.reasoning_response)
        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=AssistantMessage(
                        content=state.full_response,
                        **(
                            {"reasoning_content": state.reasoning_response}
                            if state.reasoning_response
                            else {}
                        ),
                    ),
                    finish_reason=state.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("[%s] non-stream request failed", request_id)
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        request_states.pop(request_id, None)
        release_request_slot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official-runtime RKLLM Vision OpenAI API server")
    parser.add_argument("--encoder_model", required=True, help="Vision encoder .rknn path")
    parser.add_argument("--llm_model", required=True, help="Multimodal language model .rkllm path")
    parser.add_argument("--model_name", default=os.environ.get("API_MODEL_NAME") or "rkllm-vision")
    parser.add_argument("--target_platform", choices=["rk3576", "rk3588", "rk3588s", "rk3562", "rv1126b"], default="rk3588")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max_context_len", type=int, default=2048)
    parser.add_argument("--default_temperature", type=float, default=0.7)
    parser.add_argument("--default_top_p", type=float, default=1.0)
    parser.add_argument("--default_top_k", type=int, default=1)
    parser.add_argument("--default_max_tokens", type=int, default=512)
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--rknn_core_num", type=int, default=None)
    parser.add_argument("--img_start", default="<|vision_start|>")
    parser.add_argument("--img_end", default="<|vision_end|>")
    parser.add_argument("--img_content", default="<|image_pad|>")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--log_level",
        default=initial_log_level,
        choices=LOG_LEVELS,
        help="Log level (default: LOG_LEVEL or info)",
    )
    args = parser.parse_args()

    effective_log_level = "debug" if args.debug else normalize_log_level(args.log_level)
    numeric_log_level = getattr(logging, effective_log_level.upper())
    logging.getLogger().setLevel(numeric_log_level)
    logger.setLevel(numeric_log_level)

    for path, label in ((args.encoder_model, "encoder model"), (args.llm_model, "LLM model")):
        if not os.path.exists(path):
            logger.error("%s not found: %s", label.capitalize(), path)
            sys.exit(1)
    config.encoder_model_path = os.path.abspath(args.encoder_model)
    config.llm_model_path = os.path.abspath(args.llm_model)
    config.model_name = args.model_name
    config.platform = args.target_platform
    config.max_context_len = args.max_context_len
    config.default_temperature = args.default_temperature
    config.default_top_p = args.default_top_p
    config.default_top_k = args.default_top_k
    config.default_max_tokens = args.default_max_tokens
    config.max_concurrent_requests = args.max_concurrent
    config.timeout_seconds = args.timeout
    config.host = args.host
    config.port = args.port
    config.rknn_core_num = (
        args.rknn_core_num
        if args.rknn_core_num is not None
        else platform_npu_core_limit(args.target_platform)
    )
    config.img_start = args.img_start
    config.img_end = args.img_end
    config.img_content = args.img_content
    logger.info(
        "Configuration: vision_model=%s llm_model=%s platform=%s host=%s "
        "port=%s api_model=%s vision_cores=%s context=%s temperature=%s "
        "top_p=%s top_k=%s max_tokens=%s timeout=%ss log_level=%s",
        config.encoder_model_path,
        config.llm_model_path,
        config.platform,
        args.host,
        args.port,
        config.model_name,
        config.rknn_core_num,
        config.max_context_len,
        config.default_temperature,
        config.default_top_p,
        config.default_top_k,
        config.default_max_tokens,
        config.timeout_seconds,
        effective_log_level,
    )

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=effective_log_level,
            access_log=True,
            timeout_keep_alive=config.timeout_seconds,
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
