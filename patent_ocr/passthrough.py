"""Text-native passthrough detection (§5.4).

Distinguishes "has extractable, sane text" (modern bulk USPTO/EPO XML-derived
PDFs) from "has a text layer at all" (e.g. a prior garbage OCR pass, or
control-character soup from a bad CID font without a ToUnicode map). Only the
former should skip OCR entirely.
"""
from __future__ import annotations

import re
import string
from pathlib import Path

from pypdf import PdfReader

_PLAUSIBLE_CHARS = set(string.ascii_letters + string.digits + string.whitespace + string.punctuation)
_WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-.,;:()/]*$")

# Minimum extractable characters per page for a page to even be considered.
_MIN_CHARS_PER_PAGE = 40
# Minimum fraction of characters that must be "plausible" (normal ASCII-range text).
_MIN_PLAUSIBLE_CHAR_RATIO = 0.92
# Minimum fraction of whitespace-split tokens that look like real words/numbers.
_MIN_PLAUSIBLE_WORD_RATIO = 0.75


def _page_is_sane_text(text: str) -> bool:
    text = text.strip()
    if len(text) < _MIN_CHARS_PER_PAGE:
        return False
    plausible_chars = sum(1 for c in text if c in _PLAUSIBLE_CHARS)
    if plausible_chars / len(text) < _MIN_PLAUSIBLE_CHAR_RATIO:
        return False
    tokens = text.split()
    if not tokens:
        return False
    plausible_tokens = sum(1 for t in tokens if _WORD_RE.match(t))
    return (plausible_tokens / len(tokens)) >= _MIN_PLAUSIBLE_WORD_RATIO


def analyze_text_native(pdf_path: str | Path) -> tuple[bool, list[bool]]:
    """Return (fully_text_native, per_page_sane_flags).

    fully_text_native is True only if every page has sane, extractable text —
    conservative on purpose: a single scanned page in an otherwise text-native
    file still needs the OCR pipeline to run (OCRmyPDF's own --skip-text then
    protects the already-good pages at page granularity).
    """
    reader = PdfReader(str(pdf_path))
    flags: list[bool] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        flags.append(_page_is_sane_text(text))
    fully_native = bool(flags) and all(flags)
    return fully_native, flags
