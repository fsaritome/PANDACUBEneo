"""Empirical detection of patent margin line numbers.

Position alone - "a narrow cluster in the left margin" - is not a reliable
signal. Measured on real documents it classified only ~19% of pages, flipped
to detecting nothing when the same page was rasterized at 350dpi instead of
300, and misread the vertical sidebar publication number "EP 1 439 083 A3" as
line numbering.

Line numbering has a much stronger, testable signature: a column of purely
numeric tokens sharing a near-constant edge, strictly ascending, separated by
a constant step. This module validates that arithmetic instead of trusting
geometry, so a run either satisfies the sequence or it does not.
"""
from __future__ import annotations

from dataclasses import dataclass

from patent_ocr.ocr.base import Word


@dataclass
class LineNumberRun:
    words: list[Word]
    step: int

    @property
    def values(self) -> list[int]:
        return [int(w.text) for w in self.words]


def _edge_spread(words: list[Word], index: int) -> int:
    edges = [w.bbox[index] for w in words]
    return max(edges) - min(edges)


def _cluster_by_edge(words: list[Word], index: int, tolerance: int) -> list[Word]:
    """Largest group of words whose chosen edge falls within `tolerance`."""
    best: list[Word] = []
    for anchor in words:
        anchor_edge = anchor.bbox[index]
        group = [w for w in words if abs(w.bbox[index] - anchor_edge) <= tolerance]
        if len(group) > len(best):
            best = group
    return best


def _constant_step(values: list[int]) -> int | None:
    """The common step of a strictly ascending run, tolerating skipped entries.

    Line numbers that the OCR missed leave a gap, so a diff of 2*step is
    still consistent with the sequence - but an irregular diff is not.
    """
    diffs = [b - a for a, b in zip(values, values[1:])]
    if not diffs or any(d <= 0 for d in diffs):
        return None
    step = min(diffs)
    if step <= 0 or any(d % step for d in diffs):
        return None
    return step


def detect_line_numbers(
    words: list[Word],
    page_width: int,
    max_width_fraction: float = 0.12,
    min_run: int = 3,
    edge_tolerance_fraction: float = 0.015,
) -> LineNumberRun | None:
    """Find the margin line-number run on a page, or None if there isn't one.

    `words` may be every word on the page: detection does not depend on the
    layout model having produced a region, which is what made the previous
    positional approach resolution-fragile.
    """
    if not words or page_width <= 0:
        return None
    limit = max_width_fraction * page_width
    candidates = [
        w for w in words
        if w.bbox[2] <= limit and w.text.strip().isdigit() and len(w.text.strip()) <= 4
    ]
    if len(candidates) < min_run:
        return None

    tolerance = max(4, int(edge_tolerance_fraction * page_width))
    # Right edge is the stable anchor when numbers are right-aligned against
    # the text block ('5' and '10' share x1 but differ ~20px in x0); left edge
    # is stable when they are left-aligned. Use whichever actually holds.
    edge = 2 if _edge_spread(candidates, 2) <= _edge_spread(candidates, 0) else 0
    aligned = _cluster_by_edge(candidates, edge, tolerance)
    if len(aligned) < min_run:
        return None

    ordered = sorted(aligned, key=lambda w: w.bbox[1])
    values = [int(w.text.strip()) for w in ordered]
    step = _constant_step(values)
    if step is None:
        return None
    return LineNumberRun(words=ordered, step=step)


def looks_like_line_numbers(words: list[Word], page_width: int, **kwargs) -> bool:
    return detect_line_numbers(words, page_width, **kwargs) is not None
