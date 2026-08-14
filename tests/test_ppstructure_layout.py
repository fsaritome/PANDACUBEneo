from patent_ocr.config import Config
from patent_ocr.layout.ppstructure import _contains, _recover_unclaimed, _words_from_ocr
from patent_ocr.layout.types import RegionKind
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


def test_left_margin_words_lead_and_others_trail():
    """Regression: figure callout numbers used to be front-loaded alongside
    margin numbers, so drawing labels opened the document text."""
    margin = [Word("5", (30, 100, 60, 130), 99.0, "e")]
    callouts = [Word("17", (900, 400, 940, 430), 99.0, "e")]
    leading, trailing = _recover_unclaimed(margin + callouts, 2000, _cfg())

    assert len(leading) == 1 and leading[0].kind == RegionKind.MARGIN_NUMBERS
    assert [w.text for w in leading[0].words] == ["5"]
    assert len(trailing) == 1 and trailing[0].kind == RegionKind.OTHER
    assert [w.text for w in trailing[0].words] == ["17"]


def test_no_leading_region_when_nothing_is_in_the_margin():
    callouts = [Word("17", (900, 400, 940, 430), 99.0, "e")]
    leading, trailing = _recover_unclaimed(callouts, 2000, _cfg())
    assert leading == []
    assert len(trailing) == 1


def test_nothing_unclaimed_yields_nothing():
    assert _recover_unclaimed([], 2000, _cfg()) == ([], [])
