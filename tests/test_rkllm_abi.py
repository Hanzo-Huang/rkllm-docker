"""Compare Python bindings with C++ layouts from the actual vendored SDK header.

Run: python -m unittest discover -s tests -v
Requires a C++ compiler, but no model, Rockchip hardware, or Python ML packages.
"""
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import rkllm_types as abi


class NativeLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++') or shutil.which('g++')
        if not compiler:
            raise RuntimeError('A C++ compiler is required to verify the RKLLM ABI')
        cls.types = {name: value for name, value in vars(abi).items()
                     if isinstance(value, type) and issubclass(value, (ctypes.Structure, ctypes.Union))}
        source = '''#include <cstddef>
#include <iostream>
#include "rkllm.h"
using RKLLMImageInput = decltype(RKLLMMultiModalInput::image);
using RKLLMVideoInput = decltype(RKLLMMultiModalInput::video);
using RKLLMAudioInput = decltype(RKLLMMultiModalInput::audio);
// The header exposes this union anonymously. Use its native member types.
union RKLLMInputUnion {
    decltype(RKLLMInput::prompt_input) prompt_input;
    decltype(RKLLMInput::embed_input) embed_input;
    decltype(RKLLMInput::token_input) token_input;
    decltype(RKLLMInput::multimodal_input) multimodal_input;
};
int main() {
'''
        for name, typ in cls.types.items():
            source += f'std::cout << "{name} " << sizeof({name}) << " " << alignof({name}) << "\\n";\n'
            for field, _ in typ._fields_:
                native_field = 'prompt_input' if name == 'RKLLMInput' and field == 'input_data' else field
                size = 'sizeof(RKLLMInputUnion)' if name == 'RKLLMInput' and field == 'input_data' else f'sizeof((({name}*)nullptr)->{native_field})'
                source += f'std::cout << "{name}.{field} " << offsetof({name}, {native_field}) << " " << {size} << "\\n";\n'
        source += '}\n'
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        probe = directory / 'probe.cpp'
        probe.write_text(source)
        executable = directory / 'probe'
        subprocess.run([compiler, '-std=c++11', '-I', str(ROOT / 'runtime/include'),
                        str(probe), '-o', str(executable)], check=True, capture_output=True, text=True)
        output = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout
        cls.native = {parts[0]: tuple(map(int, parts[1:]))
                      for line in output.splitlines() if (parts := line.split())}

    def test_struct_sizes_alignments_and_every_field(self):
        for name, typ in self.types.items():
            with self.subTest(structure=name):
                self.assertEqual((ctypes.sizeof(typ), ctypes.alignment(typ)), self.native[name])
            for field, field_type in typ._fields_:
                with self.subTest(structure=name, field=field):
                    self.assertEqual((getattr(typ, field).offset, ctypes.sizeof(field_type)),
                                     self.native[f'{name}.{field}'])

    def test_131_input_contains_zero_initialized_audio_tail(self):
        value = abi.RKLLMInput()
        value.input_type = 0
        value.input_data.prompt_input = b'Hello'
        # SDK 1.3.1 input is 208 bytes on LP64; old 1.3.0 allocated only 160.
        self.assertEqual(ctypes.sizeof(abi.RKLLMInput), 208)
        self.assertEqual(ctypes.sizeof(abi.RKLLMMultiModalInput), 192)
        audio = value.input_data.multimodal_input.audio
        self.assertEqual(ctypes.string_at(ctypes.addressof(audio), ctypes.sizeof(audio)),
                         bytes(ctypes.sizeof(audio)))
        self.assertFalse(audio.audio_embed)
        self.assertEqual(audio.n_audio_tokens, 0)
        self.assertEqual(audio.n_audio, 0)


class RuntimePackagingTests(unittest.TestCase):
    def test_bundled_runtime_matches_release_manifest(self):
        manifest = json.loads((ROOT / 'runtime/rkllm-manifest.json').read_text())
        self.assertEqual(manifest['version'], '1.3.1')
        self.assertEqual(manifest['tag'], 'release-v1.3.1')
        self.assertRegex(manifest['commit'], r'^[a-f0-9]{40}$')
        for filename, expected in manifest['sha256'].items():
            with self.subTest(file=filename):
                self.assertEqual(hashlib.sha256((ROOT / 'runtime' / filename).read_bytes()).hexdigest(), expected)
        binary = (ROOT / 'runtime/lib/librkllmrt.so').read_bytes()
        self.assertEqual(binary[:6], b'\x7fELF\x02\x01')  # ELF64, little endian
        self.assertEqual(struct.unpack_from('<H', binary, 18)[0], 183)  # AArch64

    def test_image_version_and_shared_module_packaging(self):
        dockerfile = (ROOT / 'docker/Dockerfile').read_text()
        workflow = (ROOT / '.github/workflows/build-env-image.yml').read_text()
        self.assertEqual(re.search(r'ARG RKLLM_RUNTIME_VERSION=(\S+)', dockerfile)[1], '1.3.1')
        self.assertEqual(re.search(r'RKLLM_RUNTIME_VERSION: (\S+)', workflow)[1], '1.3.1')
        self.assertIn('COPY app/rkllm_types.py /app/rkllm_types.py', dockerfile)

    def test_both_servers_use_the_shared_native_types(self):
        for filename in ('fastapi_server_llm.py', 'fastapi_server_vlm.py'):
            tree = ast.parse((ROOT / 'app' / filename).read_text())
            self.assertFalse(any(isinstance(node, ast.ClassDef) and node.name.startswith('RKLLM')
                                 and node.name in vars(abi) for node in tree.body))
            binding_import = next(node for node in tree.body if isinstance(node, ast.If)
                                  and isinstance(node.test, ast.Name) and node.test.id == '__package__')
            namespace = {'__package__': 'app'}
            exec(compile(ast.Module(body=[binding_import], type_ignores=[]), filename, 'exec'), namespace)
            for name in ('RKLLMInput', 'RKLLMAudioInput', 'RKLLMResult', 'RKLLMCallback'):
                self.assertIs(namespace[name], getattr(abi, name))


if __name__ == '__main__':
    unittest.main()
