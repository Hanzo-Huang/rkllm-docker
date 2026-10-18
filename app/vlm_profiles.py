"""Model-specific vision markers for the native RKLLM chat template."""
from dataclasses import dataclass


@dataclass(frozen=True)
class VLMProfile:
    family: str
    image_start: str
    image_end: str
    image_content: str


def resolve_profile(model_path: str, family: str = 'auto') -> VLMProfile:
    if family == 'auto':
        name = model_path.lower().replace('_', '-')
        family = 'gemma4' if 'gemma-4' in name or 'gemma4' in name else 'qwen'
    if family == 'gemma4':
        return VLMProfile('gemma4', '<|image>', '<image|>', '<|image|>')
    if family == 'qwen':
        return VLMProfile('qwen', '<|vision_start|>', '<|vision_end|>', '<|image_pad|>')
    raise ValueError(f'Unsupported VLM model family: {family}')


def format_gemma4_chat(turns) -> str:
    """Canonical multimodal chat with thinking disabled and explicit history."""
    prompt = '<bos>'
    for role, text in turns:
        role = 'model' if role == 'assistant' else role
        prompt += f'<|turn>{role}\n{text.strip()}<turn|>\n'
    return prompt + '<|turn>model\n'
