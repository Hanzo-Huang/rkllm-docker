import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from typing import List, Optional, Tuple

from app.vlm_profiles import resolve_profile, format_gemma4_chat


class VisionProfileTests(unittest.TestCase):
    def test_gemma4_markers_and_renamed_override(self):
        profile=resolve_profile('/models/gemma-4-E2B-it_RK3576_w8a8.rkllm')
        self.assertEqual((profile.family,profile.image_start,profile.image_end,profile.image_content),
                         ('gemma4','<|image>','<image|>','<|image|>'))
        self.assertEqual(resolve_profile('renamed.rkllm','gemma4'),profile)
        self.assertEqual(resolve_profile('Qwen3.5-2B.rkllm').image_content,'<|image_pad|>')

    def test_gemma_prompt_has_one_canonical_chat_template(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'app/fastapi_server_vlm.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_prompt')
        config=SimpleNamespace(model_family='gemma4')
        namespace={'List':List,'Optional':Optional,'Tuple':Tuple,'Message':object,'config':config,
                   'format_gemma4_chat':format_gemma4_chat,
                   'message_parts':lambda message:(message.text,message.image)}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'build_prompt','exec'),namespace)
        message=SimpleNamespace(role='user',text='Describe this picture.',image='data:image/png;base64,...')
        prompt,url=namespace['build_prompt']([message])
        self.assertEqual(prompt,'<bos><|turn>user\n<image>Describe this picture.<turn|>\n<|turn>model\n')
        self.assertEqual(url,message.image)
        config.model_family='qwen'
        self.assertEqual(namespace['build_prompt']([message])[0],'Human: <image>Describe this picture.\nAssistant:')

    def test_gemma_system_and_history_roles(self):
        self.assertEqual(format_gemma4_chat([('system','Be brief.'),('user','Hi'),('assistant','Hello'),('user','Again')]),
                         '<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello<turn|>\n<|turn>user\nAgain<turn|>\n<|turn>model\n')
