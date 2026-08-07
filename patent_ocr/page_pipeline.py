"""Per-page pipeline (§4): layout segmentation -> OCR engine layer -> confidence
scoring -> optional LLM fallback -> reassembly -> hOCR. This is the piece that
makes the pipeline more than "just run OCR" (§3.2): region order is decided
*before* recognition, not inferred from raw OCR output afterward.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from patent_ocr.config import Config
from patent_ocr.confidence import is_low_confidence, region_confidence_summary
from patent_ocr.fallback.llm_fallback import run_fallback
from patent_ocr.hocr import build_hocr
from patent_ocr.langid import detect_languages_for_image
from patent_ocr.layout.segmenter import segment_page
from patent_ocr.layout.types import Region, RegionKind
from patent_ocr.ocr.registry import build_engine
from patent_ocr.reassembly import assemble_page, reconcile_region, remap_words_to_page

# Engines are expensive to instantiate (esp. GPU-backed ones); reuse within a worker process.
_ENGINE_CACHE: dict[str, object] = {}


def _get_engine(name: str, **kwargs):
    if name not in _ENGINE_CACHE:
        _ENGINE_CACHE[name] = build_engine(name, **kwargs)
    return _ENGINE_CACHE[name]


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


class PageResult:
    def __init__(self, hocr_xml: str, text: str, qc: dict, regions_for_render: list[Region]):
        self.hocr_xml = hocr_xml
        self.text = text
        self.qc = qc
        # Ordered regions with page-coordinate words, for direct PDF text-layer
        # rendering (bypasses hOCR entirely — see pdf_text_layer.py).
        self.regions_for_render = regions_for_render


def process_page_image(image_path, config: Config) -> PageResult:
    pil_image = Image.open(image_path).convert("RGB")
    image = np.array(pil_image)
    h, w = image.shape[:2]

    layout_type, regions = segment_page(image, config.layout)

    primary_engine = _get_engine(config.engine.primary, **config.engine.engine_options.get(config.engine.primary, {}))
    secondary_engine = (
        _get_engine(config.engine.secondary, **config.engine.engine_options.get(config.engine.secondary, {}))
        if config.engine.secondary
        else None
    )

    # Per-page language ID (§5.5): sample the largest column-ish region, hint every region with it.
    sample_region = max(regions, key=lambda r: r.width * r.height)
    languages = detect_languages_for_image(_crop(image, sample_region.bbox), primary_engine, config.languages)

    region_qc: list[dict] = []
    for region in regions:
        if region.kind == RegionKind.FIGURE:
            # Underlying image is never touched (R6); whether a text layer gets
            # attached here is a config decision deferred per §5.9 — skip OCR for now.
            region_qc.append({"kind": region.kind.value, "bbox": region.bbox, "skipped": True})
            continue

        crop = _crop(image, region.bbox)
        primary_words = remap_words_to_page(region, primary_engine.recognize(crop, languages))
        engines_used = [config.engine.primary]

        secondary_words = None
        run_secondary = secondary_engine is not None and (
            config.engine.strategy == "always_parallel"
            or (
                config.engine.strategy == "low_confidence_only"
                and is_low_confidence(_region_with(region, primary_words), config.engine)
            )
        )
        if run_secondary:
            secondary_words = remap_words_to_page(region, secondary_engine.recognize(crop, languages))
            engines_used.append(config.engine.secondary)

        region.words = reconcile_region(region, primary_words, secondary_words, config.engine.strategy)
        winner = region.words[0].engine if region.words else None

        fallback_info = {"fired": False}
        if is_low_confidence(region, config.engine) and config.fallback.enabled:
            result = run_fallback(crop, region, config.fallback)
            fallback_info = {
                "fired": result.fired,
                "applied_as_text_layer": result.applied_as_text_layer,
                "flagged_for_review": result.flagged_for_review,
                "reason": result.reason,
            }

        region_qc.append(
            {
                "kind": region.kind.value,
                "bbox": region.bbox,
                "engines_used": engines_used,
                "winner": winner,
                "confidence": region_confidence_summary(region),
                "fallback": fallback_info,
            }
        )

    ordered_regions = assemble_page(regions)
    hocr_xml = build_hocr(w, h, ordered_regions)
    text = "\n".join(
        " ".join(word.text for word in region.words) for region in ordered_regions if region.words
    )

    flagged = any(rq.get("fallback", {}).get("flagged_for_review") for rq in region_qc)
    fallback_fired = any(rq.get("fallback", {}).get("fired") for rq in region_qc)
    qc = {
        "layout_type": layout_type.value,
        "languages": languages,
        "regions": region_qc,
        "flagged": flagged,
        "fallback_fired": fallback_fired,
    }
    return PageResult(hocr_xml, text, qc, ordered_regions)


def _region_with(region: Region, words) -> Region:
    """Shallow copy helper so a low-confidence check can run before commit to `region.words`."""
    probe = Region(kind=region.kind, bbox=region.bbox, order_index=region.order_index, column_index=region.column_index)
    probe.words = words
    return probe
