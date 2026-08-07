"""Vision-LLM fallback for low-confidence regions (§5.8).

The LLM's only sanctioned role is a second-pass check on flagged regions —
never the primary path, never silently trusted over the primary engine (§3.1:
OCR errors are visibly wrong, LLM errors are plausibly wrong — the worse
failure mode in a legal/IP corpus).

Default behavior is (b): route the page to human review with the LLM
transcription attached as an aid, keep the primary engine's best-effort
text/boxes in the actual text layer. Option (a) — using the LLM's text to
*replace* the primary engine's word text while keeping its boxes — is only
attempted when `apply_as_text_layer` is enabled AND the LLM's token count
matches the primary engine's word count exactly (a cheap, conservative safety
check); any mismatch always falls back to (b), regardless of the config flag.
"""
from __future__ import annotations

from dataclasses import dataclass

from patent_ocr.config import FallbackConfig
from patent_ocr.fallback.llm_client import transcribe_region
from patent_ocr.layout.types import Region
from patent_ocr.ocr.base import Word


@dataclass
class FallbackResult:
    fired: bool
    llm_text: str | None
    applied_as_text_layer: bool
    flagged_for_review: bool
    reason: str


def run_fallback(region_image, region: Region, config: FallbackConfig) -> FallbackResult:
    if not config.enabled:
        return FallbackResult(False, None, False, False, "fallback disabled")

    try:
        llm_text = transcribe_region(region_image, config)
    except Exception as exc:  # noqa: BLE001 - fallback must never crash the pipeline
        return FallbackResult(True, None, False, True, f"LLM call failed: {exc}")

    if not config.apply_as_text_layer:
        return FallbackResult(True, llm_text, False, True, "low confidence; routed to human review")

    llm_tokens = llm_text.split()
    if len(llm_tokens) != len(region.words):
        return FallbackResult(
            True, llm_text, False, True,
            f"LLM token count ({len(llm_tokens)}) != primary word count ({len(region.words)}); "
            "unsafe to graft LLM text onto primary boxes, routed to human review",
        )

    # Safe to substitute text 1:1 onto the primary engine's existing geometry.
    corrected_words = [
        Word(text=tok, bbox=w.bbox, confidence=w.confidence, engine=f"{w.engine}+llm")
        for tok, w in zip(llm_tokens, region.words)
    ]
    region.words = corrected_words
    return FallbackResult(True, llm_text, True, False, "LLM correction applied 1:1 onto primary boxes")
