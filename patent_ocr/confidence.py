"""Confidence scoring (§5.7, §5.11): drives both the fallback trigger and QC log."""
from __future__ import annotations

from patent_ocr.config import EngineConfig
from patent_ocr.layout.types import Region


def region_confidence_summary(region: Region) -> dict:
    if not region.words:
        return {"mean": 0.0, "min": 0.0, "n_words": 0, "low_fraction": 1.0}
    confidences = [w.confidence for w in region.words]
    return {
        "mean": sum(confidences) / len(confidences),
        "min": min(confidences),
        "n_words": len(confidences),
        "low_fraction": 0.0,  # filled in by is_low_confidence's caller with a threshold
    }


def is_low_confidence(region: Region, config: EngineConfig) -> bool:
    """A region is low-confidence if enough of its words fall under the word
    threshold — this is what triggers secondary-engine or LLM fallback."""
    if not region.words:
        return True
    low = sum(1 for w in region.words if w.confidence < config.low_confidence_word_threshold)
    return (low / len(region.words)) >= config.low_confidence_region_fraction
