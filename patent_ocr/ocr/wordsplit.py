"""Approximate word-level boxes from coarser OCR output.

Detection-based engines (PP-OCR, Surya) return text-*line* granularity and
VLM engines (PaddleOCR-VL) return whole-*block* granularity — neither gives
word boxes. Distributing tokens proportionally by character width across a
line is a close approximation, because the line's true pixel extent was
measured by a detector rather than guessed. Doing the same across a
multi-line block is not: every word would inherit the block's full vertical
extent. So `split_block_into_words` first divides the block into per-line
strips and only then defers to `split_line_into_words`.
"""
from __future__ import annotations

from patent_ocr.ocr.base import BBox, Word


def split_line_into_words(text: str, bbox: BBox, confidence: float, engine: str) -> list[Word]:
    """Split one text line's box into per-token boxes by character width."""
    tokens = text.split()
    if not tokens:
        return []
    x0, y0, x1, y1 = bbox
    span = x1 - x0
    if span <= 0 or len(tokens) == 1:
        return [Word(text=" ".join(tokens), bbox=bbox, confidence=confidence, engine=engine)]

    # Inter-word spaces occupy real width on the page; excluding them from the
    # denominator makes every box drift progressively right across the line.
    total = sum(len(t) for t in tokens) + (len(tokens) - 1)
    words: list[Word] = []
    cursor = 0
    for token in tokens:
        wx0 = x0 + round(span * cursor / total)
        wx1 = x0 + round(span * (cursor + len(token)) / total)
        words.append(
            Word(
                text=token,
                bbox=(int(wx0), int(y0), int(max(wx1, wx0 + 1)), int(y1)),
                confidence=confidence,
                engine=engine,
            )
        )
        cursor += len(token) + 1
    return words


def split_block_into_words(text: str, bbox: BBox, confidence: float, engine: str) -> list[Word]:
    """Split a multi-line block's box into per-token boxes.

    Assumes uniform line height across the block, which is only an estimate —
    but it keeps each word inside its own line's horizontal band instead of
    stretching it over the block's entire height.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    words: list[Word] = []
    for index, line in enumerate(lines):
        line_y0 = y0 + round(height * index / len(lines))
        line_y1 = y0 + round(height * (index + 1) / len(lines))
        words.extend(
            split_line_into_words(line, (x0, int(line_y0), x1, int(line_y1)), confidence, engine)
        )
    return words
