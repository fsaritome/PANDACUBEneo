"""OpenAI-compatible vision client for the local LLM fallback (§5.8).

Talks to whatever OpenAI-compatible endpoint is already running on `ai01`
(vLLM or Ollama) — no cloud API, per the on-prem deployment constraint.
"""
from __future__ import annotations

import base64
import io

import requests
from PIL import Image

from patent_ocr.config import FallbackConfig


def _image_to_base64(image) -> str:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def transcribe_region(region_image, config: FallbackConfig) -> str:
    """Ask the local vision LLM to transcribe a low-confidence region image.

    Returns plain text only — this is a *second-pass fallback*, never trusted
    as bbox-accurate (per §3.1, vision LLMs don't produce reliable boxes).
    """
    b64 = _image_to_base64(region_image)
    payload = {
        "model": config.model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the text in this image exactly as written, "
                            "preserving line breaks. Do not translate, summarize, "
                            "or correct spelling/OCR-looking errors — reproduce "
                            "exactly what is printed, including any obvious typos."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {config.api_key}"}
    resp = requests.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=config.timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
