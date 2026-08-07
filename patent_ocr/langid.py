"""Per-page language detection (§5.5). EPO documents mix DE/EN/FR within one
document, so detection happens per page, not just once per file, and the
result is fed back into the OCR engine as a language hint."""
from __future__ import annotations

import numpy as np
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

# Deterministic results across runs (langdetect is otherwise seeded from clock time).
DetectorFactory.seed = 0

_ISO_TO_TESSERACT = {"en": "eng", "de": "deu", "fr": "fra"}


def detect_languages_from_text(
    text: str, known_languages: list[str], min_probability: float = 0.15
) -> list[str]:
    """Return tesseract-style 3-letter codes for languages detected in `text`,
    above `min_probability`. Results are constrained to `known_languages` (the
    tesseract codes this deployment is actually configured/installed for) —
    `langdetect` can report any of ~55 ISO codes, and blindly passing an
    unsupported/uninstalled one to tesseract crashes the OCR call outright.
    Falls back to `known_languages` (or ['eng']) if detection fails, the text
    is too short, or no candidate matches a known language — never returns an
    empty list, since something must hint the OCR engine."""
    fallback = known_languages or ["eng"]
    text = text.strip()
    if len(text) < 8:
        return fallback
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return fallback
    codes = [
        _ISO_TO_TESSERACT.get(c.lang, c.lang)
        for c in candidates
        if c.prob >= min_probability
    ]
    codes = [c for c in codes if c in known_languages]
    return codes or fallback


def detect_languages_for_image(region_image, quick_engine, known_languages: list[str]) -> list[str]:
    """Run a cheap OCR pass (all configured languages loaded at once) purely to
    obtain a text sample for language ID, then detect language(s) from that
    sample. The *real* recognition pass is re-run afterward with the resulting
    hint for better per-language accuracy."""
    sample_words = quick_engine.recognize(region_image, lang_hint=known_languages)
    sample_text = " ".join(w.text for w in sample_words[:200])
    return detect_languages_from_text(sample_text, known_languages)
