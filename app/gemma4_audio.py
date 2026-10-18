"""Gemma4 WAV decoding and log-mel features matching Rockchip's audio demo."""

import io
import math
import wave
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def mel_filters():
    frequencies = np.arange(257, dtype=np.float32) * (16000 / 512)
    points = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 130) / 2595) - 1)
    down = (frequencies[:, None] - points[:-2]) / np.diff(points)[:-1]
    up = (points[2:] - frequencies[:, None]) / np.diff(points)[1:]
    return np.maximum(0, np.minimum(down, up)).astype(np.float32)


def decode_wav(data: bytes):
    """Decode and resample supported WAV audio to mono 16 kHz.

    Accept PCM16 WAV (mono/stereo, 8–48 kHz). Longer inputs must be split by
    the caller; do not silently truncate speech beyond the fixed export window.
    """
    try:
        with wave.open(io.BytesIO(data), 'rb') as audio:
            rate, channels, width = audio.getframerate(), audio.getnchannels(), audio.getsampwidth()
            frames = audio.getnframes()
            if width != 2 or channels not in (1, 2) or not 8000 <= rate <= 48000:
                raise ValueError('Audio requires PCM16 WAV, mono/stereo, 8–48 kHz')
            if frames == 0 or frames / rate > 7.56:
                raise ValueError('Audio must contain between 0 and 7.56 seconds of speech')
            samples = np.frombuffer(audio.readframes(frames), dtype='<i2').astype(np.float32)
            if samples.size != frames * channels:
                raise ValueError('WAV audio is truncated')
    except (wave.Error, EOFError) as error:
        raise ValueError('Invalid PCM WAV audio') from error
    samples = samples.reshape(-1, channels).mean(axis=1) / 32768
    if rate != 16000:
        from scipy.signal import resample_poly
        factor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // factor, rate // factor).astype(np.float32)
    if len(samples) <= 160:
        raise ValueError("Audio must be longer than 10 milliseconds")
    return samples


def audio_features(data: bytes):
    """Return fixed encoder features, frame mask, and valid audio-token count."""
    samples = decode_wav(data)
    valid = min(756, max(0, (len(samples) - 160 + 159) // 160))
    if valid < 1:
        raise ValueError('Audio is too short to encode')
    # Semi-causal padding, periodic Hann, magnitude FFT, HTK mel, natural log.
    padded = np.pad(samples, (160, max(0, (755 * 160 + 320) - 160 - len(samples))))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 320)[::160][:756]
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(320) / 320)).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(windows * hann, n=512)).astype(np.float32)
    features = np.log(magnitude @ mel_filters() + 1e-3).astype(np.float32)
    features[valid:] = 0
    mask = (np.arange(756) < valid).astype(np.float32)
    return np.ascontiguousarray(features[None]), mask[None], (valid + 3) // 4
