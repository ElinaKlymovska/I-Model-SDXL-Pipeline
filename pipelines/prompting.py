from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prompts:
    base_positive: str = (
        "detailed, masterpiece, hyper-realistic, soft studio lighting, sharp eyes, perfect skin texture,"
        " clean background, high dynamic range, editorial photo"
    )
    base_negative: str = (
        "lowres, blurry, out of focus, artifacts, extra fingers, deformed, text, watermark, harsh shadows,"
        " oversaturated, poorly drawn, duplicate, worst quality"
    )


def build_positive(style_hint: str, extra: str = "") -> str:
    parts = [Prompts.base_positive]
    if style_hint:
        parts.append(style_hint)
    if extra:
        parts.append(extra)
    return ", ".join([p for p in parts if p])


def build_negative(extra: str = "") -> str:
    if extra:
        return f"{Prompts.base_negative}, {extra}"
    return Prompts.base_negative


