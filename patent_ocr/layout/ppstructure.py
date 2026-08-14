"""PP-StructureV3 layout backend (§5.6 alternative to the heuristic segmenter).

Replaces the OpenCV projection-profile heuristics with PaddleOCR's trained
layout model (PP-DocLayout), which also yields reading order, semantic region
labels, and table/formula regions. Layout and OCR come from a *single* call —
`parse_page` returns both regions and page-coordinate words — because running
PP-StructureV3 twice (once for layout, once for text) would double the cost.

Known gap this module compensates for: the layout model does **not** emit a
region for patent margin line-numbers (verified on a real EP claims page — 0
of 23 detected regions covered the margin band, though the OCR layer read all
seven numbers correctly). Any recognized line left unclaimed by every layout
region is therefore recovered into a synthetic region rather than dropped.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from patent_ocr.config import Config
from patent_ocr.layout.line_numbers import detect_line_numbers
from patent_ocr.layout.types import LayoutType, Region, RegionKind
from patent_ocr.ocr.base import Word

log = logging.getLogger(__name__)

_ENGINE_NAME = "ppstructure"

# PP-DocLayout labels -> our RegionKind.
_LABEL_KINDS = {
    "text": RegionKind.COLUMN,
    "paragraph_title": RegionKind.TITLE,
    "doc_title": RegionKind.TITLE,
    "abstract": RegionKind.COLUMN,
    "content": RegionKind.COLUMN,
    "header": RegionKind.OTHER,
    "footer": RegionKind.OTHER,
    "number": RegionKind.OTHER,
    "footnote": RegionKind.COLUMN,
    "formula": RegionKind.FORMULA,
    "algorithm": RegionKind.FORMULA,
    "table": RegionKind.TABLE,
    "image": RegionKind.FIGURE,
    "figure": RegionKind.FIGURE,
    "chart": RegionKind.FIGURE,
    "figure_title": RegionKind.TITLE,
    "table_title": RegionKind.TITLE,
    "seal": RegionKind.OTHER,
}

_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline(config: Config):
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from paddleocr import PPStructureV3

                opts = dict(config.layout.ppstructure_options or {})
                use_gpu = opts.pop("use_gpu", True)
                gpu_id = opts.pop("gpu_id", 0)
                opts.setdefault("device", f"gpu:{gpu_id}" if use_gpu else "cpu")
                # Both distort page geometry and would offset every box against
                # the untouched image we sandwich under (see paddleocr_engine).
                opts.setdefault("use_doc_orientation_classify", False)
                opts.setdefault("use_doc_unwarping", False)
                opts.setdefault("use_table_recognition", True)
                try:
                    _pipeline = PPStructureV3(**opts)
                except TypeError:
                    # return_word_box isn't a documented PPStructureV3 arg; drop it
                    # and fall back to line-level boxes rather than failing outright.
                    opts.pop("return_word_box", None)
                    _pipeline = PPStructureV3(**opts)
    return _pipeline


def _words_from_ocr(ocr: dict) -> list[Word]:
    """Page-coordinate words, preferring the recognizer's measured token boxes."""
    texts = ocr.get("rec_texts") or []
    scores = ocr.get("rec_scores") or []
    boxes = ocr.get("rec_boxes")
    tokens = ocr.get("text_word")
    token_boxes = ocr.get("text_word_boxes")

    words: list[Word] = []
    for i, line in enumerate(texts):
        conf = float(scores[i]) * 100.0 if i < len(scores) else 0.0
        if tokens is not None and token_boxes is not None and i < len(token_boxes):
            for token, box in zip(tokens[i], token_boxes[i]):
                text = str(token).strip()
                if not text:
                    continue
                x0, y0, x1, y1 = box
                words.append(Word(text, (int(x0), int(y0), int(x1), int(y1)), conf, _ENGINE_NAME))
            continue
        if boxes is not None and i < len(boxes):
            x0, y0, x1, y1 = boxes[i]
            words.append(Word(line, (int(x0), int(y0), int(x1), int(y1)), conf, _ENGINE_NAME))
    return words


def _contains(bbox, word: Word) -> bool:
    x0, y0, x1, y1 = bbox
    wx0, wy0, wx1, wy1 = word.bbox
    cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _trailing_region(unclaimed: list[Word]) -> list[Region]:
    """Bucket words no layout region claimed, appended after the real content.

    Line numbers are handled separately by `_extract_line_numbers`, so what
    remains here is typically figure callout numbers bleeding out of a drawing
    region. They go to the BACK: front-loading them once put
    '17 18 12 9 ALM ON 16 SEL COM' ahead of the document's opening text.
    """
    return [_region_around(unclaimed, RegionKind.OTHER)] if unclaimed else []


def _region_around(words: list[Word], kind: RegionKind) -> Region:
    region = Region(
        kind=kind,
        bbox=(
            min(w.bbox[0] for w in words),
            min(w.bbox[1] for w in words),
            max(w.bbox[2] for w in words),
            max(w.bbox[3] for w in words),
        ),
        order_index=0,
    )
    region.words = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    return region


def _attach_table_html(regions: list[Region], res) -> None:
    """Give each TABLE region the recognized HTML for the table it covers.

    `table_res_list` entries are matched to layout regions by bbox overlap
    rather than by index, since the two lists come from different models and
    are not guaranteed to correspond positionally.
    """
    tables = res.get("table_res_list") or []
    targets = [r for r in regions if r.kind == RegionKind.TABLE]
    if not tables or not targets:
        return
    for table in tables:
        if not hasattr(table, "get"):
            continue
        html = table.get("pred_html") or table.get("html") or ""
        if not html:
            continue
        box = table.get("table_box") or table.get("bbox")
        best, best_overlap = None, 0
        if box is not None and len(list(box)) >= 4:
            tx0, ty0, tx1, ty1 = (int(v) for v in list(box)[:4])
            for region in targets:
                rx0, ry0, rx1, ry1 = region.bbox
                ox = max(0, min(tx1, rx1) - max(tx0, rx0))
                oy = max(0, min(ty1, ry1) - max(ty0, ry0))
                if ox * oy > best_overlap:
                    best, best_overlap = region, ox * oy
        if best is None:
            best = next((r for r in targets if r.html is None), None)
        if best is not None:
            best.html = html


def _extract_line_numbers(regions: list[Region], words: list[Word], page_width: int,
                          config: Config):
    """Pull a validated line-number run out of wherever it ended up.

    Scans every word on the page rather than only unclaimed ones. PP-DocLayout
    emits no region for patent line numbers, but its text regions frequently
    extend far enough left to swallow them, and a stray number can be claimed
    by its own small region - so restricting the search to unclaimed words
    missed them on 2 of 4 pages of the same document, and missed the trailing
    '35' even on a page where the rest of the run was found.
    """
    run = detect_line_numbers(
        words, page_width, max_width_fraction=config.layout.margin_max_width_fraction
    )
    if run is None:
        return None, set()
    run_ids = {id(w) for w in run.words}
    for region in regions:
        if region.words:
            region.words = [w for w in region.words if id(w) not in run_ids]
    return _region_around(run.words, RegionKind.MARGIN_NUMBERS), run_ids


def parse_page(
    image: np.ndarray, config: Config, words: list[Word] | None = None
) -> tuple[LayoutType, list[Region], list[Word]]:
    """Run layout (+ OCR) in one pass. Returns (layout_type, ordered regions, all words).

    `words` lets the caller supply page-coordinate words from the configured OCR
    engine instead of PP-StructureV3's own text. That is the only way to get
    word-level boxes here: PPStructureV3 rejects `return_word_box` outright, so
    its built-in `overall_ocr_res` is always line-level.
    """
    pipeline = _get_pipeline(config)
    results = list(pipeline.predict(image))
    if not results:
        h, w = image.shape[:2]
        return LayoutType.OTHER, [Region(RegionKind.FULL_PAGE, (0, 0, w, h), 0)], []

    res = results[0]
    if words is None:
        words = _words_from_ocr(res.get("overall_ocr_res") or {})

    # parsing_res_list is already in the model's reading order.
    blocks = res.get("parsing_res_list") or []
    regions: list[Region] = []
    for block in blocks:
        label = str(getattr(block, "label", "") or "")
        bbox = tuple(int(v) for v in getattr(block, "bbox", (0, 0, 0, 0)))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        regions.append(
            Region(kind=_LABEL_KINDS.get(label, RegionKind.OTHER), bbox=bbox, order_index=len(regions))
        )

    claimed: set[int] = set()
    for region in regions:
        # Figures hold artwork, not prose; their words are recovered separately.
        if region.kind == RegionKind.FIGURE:
            continue
        for idx, word in enumerate(words):
            if idx not in claimed and _contains(region.bbox, word):
                region.words.append(word)
                claimed.add(idx)
    for region in regions:
        region.words.sort(key=lambda w: (w.bbox[1], w.bbox[0]))

    # Line numbers are extracted page-wide, before anything else claims them:
    # they may be unclaimed, swallowed by a text region, or sitting in a tiny
    # region of their own, and only the arithmetic test distinguishes them.
    margin_region, run_ids = _extract_line_numbers(regions, words, image.shape[1], config)

    unclaimed = [
        w for i, w in enumerate(words) if i not in claimed and id(w) not in run_ids
    ]
    trailing = _trailing_region(unclaimed)
    leading = [margin_region] if margin_region is not None else []
    if unclaimed or margin_region is not None:
        log.info(
            "ppstructure: %d line-number(s) extracted, %d unclaimed word(s) appended",
            len(run_ids), len(unclaimed),
        )

    ordered = leading + [r for r in regions if r.words or r.kind == RegionKind.FIGURE] + trailing
    for i, region in enumerate(ordered):
        region.order_index = i

    kinds = {r.kind for r in ordered}
    if RegionKind.MARGIN_NUMBERS in kinds:
        layout_type = LayoutType.TWO_COLUMN_MARGIN
    elif len(ordered) <= 1:
        layout_type = LayoutType.SINGLE_COLUMN
    else:
        layout_type = LayoutType.OTHER
    _attach_table_html(ordered, res)
    return layout_type, ordered, words
