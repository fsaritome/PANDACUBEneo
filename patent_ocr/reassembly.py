"""Reassembly (§5.6/§4): remap region-local word boxes back to page coordinates
and reconcile primary/secondary engine output per region. Region *order* (the
reading-order fix from the segmenter) is preserved as-is — this stage only
fills in `.words`, it never reorders regions.
"""
from __future__ import annotations

from patent_ocr.config import EngineConfig
from patent_ocr.layout.types import Region
from patent_ocr.ocr.base import Word


def remap_words_to_page(region: Region, words: list[Word]) -> list[Word]:
    """Offset region-local word boxes by the region's page-coordinate origin."""
    ox, oy = region.bbox[0], region.bbox[1]
    return [
        Word(
            text=w.text,
            bbox=(w.bbox[0] + ox, w.bbox[1] + oy, w.bbox[2] + ox, w.bbox[3] + oy),
            confidence=w.confidence,
            engine=w.engine,
        )
        for w in words
    ]


def reconcile_region(
    region: Region,
    primary_words: list[Word],
    secondary_words: list[Word] | None,
    strategy: str,
) -> list[Word]:
    """Combine primary/secondary engine output for one region.

    Reconciliation is done at region granularity (whichever engine has higher
    mean confidence for the whole region wins), not per-word alignment — the
    per-word alignment approach is exactly the failure mode called out in
    §3.1 for vision-LLM/bbox reconciliation, and it's no more reliable between
    two OCR engines with different tokenization than it is for an LLM.
    """
    if not secondary_words or strategy == "single":
        return primary_words

    def mean_conf(words: list[Word]) -> float:
        return sum(w.confidence for w in words) / len(words) if words else 0.0

    if mean_conf(secondary_words) > mean_conf(primary_words):
        return secondary_words
    return primary_words


def assemble_page(regions: list[Region]) -> list[Region]:
    """Return regions sorted by their fixed reading-order index, ready for hOCR."""
    return sorted(regions, key=lambda r: r.order_index)
