"""Per-page pipeline (§4): layout segmentation -> OCR engine layer -> confidence
scoring -> optional LLM fallback -> reassembly -> hOCR. This is the piece that
makes the pipeline more than "just run OCR" (§3.2): region order is decided
*before* recognition, not inferred from raw OCR output afterward.
"""
from __future__ import annotations

import math
import threading

import numpy as np
from PIL import Image

from patent_ocr.config import Config
from patent_ocr.confidence import is_low_confidence, region_confidence_summary
from patent_ocr.fallback.llm_fallback import run_fallback
from patent_ocr.hocr import build_hocr
from patent_ocr.langid import detect_languages_for_image, detect_languages_from_text
from patent_ocr.layout.segmenter import segment_page
from patent_ocr.layout.types import Region, RegionKind
from patent_ocr.ocr.registry import build_engine
from patent_ocr.reassembly import assemble_page, reconcile_region, remap_words_to_page

# Engines are expensive to instantiate (esp. GPU-backed ones); reuse within a worker process.
_ENGINE_CACHE: dict[str, object] = {}
# Concurrent worker threads (GPU mode uses ThreadPoolExecutor, see pipeline.py)
# racing on the check-then-create above would each build their own duplicate,
# GPU-memory-heavy engine instance before the first one lands in the cache.
_ENGINE_CACHE_LOCK = threading.Lock()


def _get_engine(name: str, **kwargs):
    if name not in _ENGINE_CACHE:
        with _ENGINE_CACHE_LOCK:
            if name not in _ENGINE_CACHE:
                # num_gpus is a pipeline-level hint consumed by the worker pool; strip it before
                # passing kwargs to the engine constructor which doesn't know about it.
                kwargs.pop("num_gpus", None)
                _ENGINE_CACHE[name] = build_engine(name, **kwargs)
    return _ENGINE_CACHE[name]


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def _assign_words_to_regions(regions: list[Region], words: list) -> dict[int, list]:
    """Distribute page-coordinate OCR words (from a full-page engine call)
    into whichever region's bbox contains each word's center point, falling
    back to max bbox-overlap for words landing in a gap between regions
    (rounding at region edges, thin separator strips, etc). Keyed by
    `id(region)` since Region isn't hashable/comparable by value.
    """
    by_region: dict[int, list] = {id(r): [] for r in regions}
    for word in words:
        wx0, wy0, wx1, wy1 = word.bbox
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        target = None
        for region in regions:
            rx0, ry0, rx1, ry1 = region.bbox
            if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                target = region
                break
        if target is None:
            best_overlap = 0
            for region in regions:
                rx0, ry0, rx1, ry1 = region.bbox
                ox = max(0, min(wx1, rx1) - max(wx0, rx0))
                oy = max(0, min(wy1, ry1) - max(wy0, ry0))
                overlap = ox * oy
                if overlap > best_overlap:
                    best_overlap = overlap
                    target = region
        if target is not None:
            by_region[id(target)].append(word)
    return by_region


class PageResult:
    def __init__(self, hocr_xml: str, text: str, qc: dict, regions_for_render: list[Region]):
        self.hocr_xml = hocr_xml
        self.text = text
        self.qc = qc
        # Ordered regions with page-coordinate words, for direct PDF text-layer
        # rendering (bypasses hOCR entirely — see pdf_text_layer.py).
        self.regions_for_render = regions_for_render


def _prepare_page_image(pil_image: Image.Image, config: Config) -> np.ndarray:
    max_mp = config.preprocess.max_page_megapixels
    w, h = pil_image.size
    pixels = w * h
    max_pixels = int(max_mp * 1_000_000)
    if max_pixels > 0 and pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(pil_image)


def process_page_image(image_path, config: Config) -> PageResult:
    pil_image = Image.open(image_path).convert("RGB")
    image = _prepare_page_image(pil_image, config)
    h, w = image.shape[:2]

    layout_type, regions = segment_page(image, config.layout)

    primary_engine = _get_engine(config.engine.primary, **config.engine.engine_options.get(config.engine.primary, {}))
    secondary_engine = (
        _get_engine(config.engine.secondary, **config.engine.engine_options.get(config.engine.secondary, {}))
        if config.engine.secondary
        else None
    )

    # Collect all non-FIGURE regions up front; how OCR is actually invoked
    # depends on whether the primary engine wants a whole page or per-region
    # crops (see OCREngine.operates_on_full_page).
    ocr_regions = [r for r in regions if r.kind != RegionKind.FIGURE]
    full_page_primary = getattr(primary_engine, "operates_on_full_page", False)

    if full_page_primary:
        # One call for the whole page; words already come back in page
        # coordinates, so no per-region crop/remap for the primary engine.
        page_words = primary_engine.recognize(image)
        sample_text = " ".join(w.text for w in page_words[:200])
        languages = detect_languages_from_text(sample_text, config.languages)
        primary_words_by_region = _assign_words_to_regions(ocr_regions, page_words)
    else:
        # Per-page language ID (§5.5): sample the largest column-ish region, hint every region with it.
        sample_region = max(regions, key=lambda r: r.width * r.height)
        languages = detect_languages_for_image(_crop(image, sample_region.bbox), primary_engine, config.languages)

        # Collect crops for all non-FIGURE regions and run one batched GPU call.
        ocr_crops = [_crop(image, r.bbox) for r in ocr_regions]
        recognize_batch = getattr(primary_engine, "recognize_batch", None)
        if recognize_batch and ocr_crops:
            primary_batch = recognize_batch(ocr_crops, languages)
        else:
            primary_batch = [primary_engine.recognize(c, languages) for c in ocr_crops]
        primary_words_by_region = {
            id(region): remap_words_to_page(region, raw)
            for region, raw in zip(ocr_regions, primary_batch)
        }

    secondary_batch: list | None = None
    if secondary_engine is not None and ocr_regions:
        secondary_crops = [_crop(image, r.bbox) for r in ocr_regions]
        sec_batch_fn = getattr(secondary_engine, "recognize_batch", None)
        if sec_batch_fn:
            secondary_batch = sec_batch_fn(secondary_crops, languages)
        else:
            secondary_batch = [secondary_engine.recognize(c, languages) for c in secondary_crops]
    secondary_words_by_region = (
        {id(region): remap_words_to_page(region, raw) for region, raw in zip(ocr_regions, secondary_batch)}
        if secondary_batch is not None
        else {}
    )

    region_qc: list[dict] = []
    for region in regions:
        if region.kind == RegionKind.FIGURE:
            region_qc.append({"kind": region.kind.value, "bbox": region.bbox, "skipped": True})
            continue

        primary_words = primary_words_by_region.get(id(region), [])
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
            secondary_words = secondary_words_by_region.get(id(region), [])
            engines_used.append(config.engine.secondary)

        region.words = reconcile_region(region, primary_words, secondary_words, config.engine.strategy)
        winner = region.words[0].engine if region.words else None

        fallback_info = {"fired": False}
        if is_low_confidence(region, config.engine) and config.fallback.enabled:
            result = run_fallback(_crop(image, region.bbox), region, config.fallback)
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
