from patent_ocr.config import Config
from patent_ocr.layout.ppstructure import (
    _contains,
    _extract_line_numbers,
    _trailing_region,
    _words_from_ocr,
)
from patent_ocr.layout.types import Region, RegionKind
from patent_ocr.ocr.base import Word


def _cfg() -> Config:
    return Config()


def test_measured_token_boxes_are_preferred_over_line_boxes():
    ocr = {
        "rec_texts": ["EP 23 705"],
        "rec_scores": [0.96],
        "rec_boxes": [[0, 0, 300, 50]],
        "text_word": [["EP", " ", "23", " ", "705"]],
        "text_word_boxes": [[[10, 0, 40, 50], [40, 0, 50, 50], [50, 0, 80, 50],
                             [80, 0, 90, 50], [90, 0, 130, 50]]],
    }
    words = _words_from_ocr(ocr)
    # Bare-space tokens carry their own boxes and must not become words.
    assert [w.text for w in words] == ["EP", "23", "705"]
    assert words[0].bbox == (10, 0, 40, 50)
    assert words[2].bbox == (90, 0, 130, 50)


def test_falls_back_to_line_boxes_without_word_data():
    ocr = {
        "rec_texts": ["a whole line"],
        "rec_scores": [0.8],
        "rec_boxes": [[5, 5, 200, 40]],
    }
    words = _words_from_ocr(ocr)
    assert len(words) == 1
    assert words[0].bbox == (5, 5, 200, 40)
    assert words[0].confidence == 80.0


def test_contains_uses_word_centre_point():
    word = Word("x", (90, 10, 130, 30), 90.0, "e")
    assert _contains((0, 0, 120, 100), word)  # centre 110 is inside
    assert not _contains((0, 0, 100, 100), word)


def test_unclaimed_words_are_appended_not_front_loaded():
    """Regression: figure callout numbers used to be front-loaded, so drawing
    labels opened the document text ahead of the real content."""
    callouts = [Word("17", (900, 400, 940, 430), 99.0, "e")]
    trailing = _trailing_region(callouts)
    assert len(trailing) == 1
    assert trailing[0].kind == RegionKind.OTHER
    assert [w.text for w in trailing[0].words] == ["17"]


def test_nothing_unclaimed_yields_no_region():
    assert _trailing_region([]) == []


def test_line_numbers_are_pulled_out_of_whatever_region_absorbed_them():
    """PP-DocLayout emits no line-number region, and its text regions often
    extend far enough left to swallow them - so extraction must scan the whole
    page, not just words no region claimed."""
    numbers = [Word(str(n), (250, 400 + i * 300, 275, 455 + i * 300), 99.0, "e")
               for i, n in enumerate((5, 10, 15, 20))]
    body = Word("spinal", (400, 400, 600, 455), 99.0, "e")

    text_region = Region(kind=RegionKind.COLUMN, bbox=(200, 0, 2000, 2000), order_index=0)
    text_region.words = numbers + [body]

    margin, run_ids = _extract_line_numbers([text_region], numbers + [body], 2480, Config())

    assert margin is not None
    assert margin.kind == RegionKind.MARGIN_NUMBERS
    assert [w.text for w in margin.words] == ["5", "10", "15", "20"]
    # The absorbing region keeps its prose and loses only the numbers.
    assert [w.text for w in text_region.words] == ["spinal"]
    assert len(run_ids) == 4


def test_extraction_leaves_regions_alone_when_there_is_no_run():
    body = [Word("spinal", (400, 400, 600, 455), 99.0, "e")]
    region = Region(kind=RegionKind.COLUMN, bbox=(200, 0, 2000, 2000), order_index=0)
    region.words = list(body)
    margin, run_ids = _extract_line_numbers([region], body, 2480, Config())
    assert margin is None
    assert run_ids == set()
    assert [w.text for w in region.words] == ["spinal"]
