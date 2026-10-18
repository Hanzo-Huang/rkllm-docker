#!/usr/bin/env python3
"""Smoke-test the installed ARM64 runtime and both servers without loading a model."""
import ctypes
import importlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/app')


def main():
    llm = importlib.import_module('fastapi_server_llm')
    vlm = importlib.import_module('fastapi_server_vlm')
    if llm.rkllm_lib is None:
        raise RuntimeError('The bundled RKLLM library could not be loaded')
    if llm.RKLLMInput is not vlm.RKLLMInput:
        raise RuntimeError('LLM and VLM bindings differ')
    ctypes.CDLL('/usr/lib/librknnrt.so')
    manifest = json.loads(Path('/usr/share/doc/rkllm/rkllm-manifest.json').read_text())
    if manifest['version'] != os.environ['RKLLM_RUNTIME_VERSION']:
        raise RuntimeError('Image version does not match its bundled runtime manifest')
    default_params = llm.rkllm_lib.rkllm_createDefaultParam
    default_params.argtypes = []
    default_params.restype = llm.RKLLMParam
    params = default_params()
    if params.extend_param.n_batch != 1:
        raise RuntimeError('Unexpected native default parameters; check the RKLLM ABI')
    print(json.dumps({'runtime': manifest['version'],
                      'input_bytes': ctypes.sizeof(llm.RKLLMInput),
                      'native_default_batch': params.extend_param.n_batch,
                      'servers_imported': ['llm', 'vlm'],
                      'inference_tested': False}))


if __name__ == '__main__':
    main()
