import pytest

from patent_ocr.layout.line_numbers import detect_line_numbers, looks_like_line_numbers
from patent_ocr.ocr.base import Word

PAGE_W = 2480


def _num(text: str, x0: int, y: int, x1: int) -> Word:
    return Word(text, (x0, y, x1, y + 55), 99.0, "e")


def _epo_margin() -> list[Word]:
    """Real geometry from a 300dpi EP claims page: right-aligned, so x1 is
    stable (~270-277) while x0 varies ~20px between '5' and '10'."""
    return [
        _num("5", 263, 466, 273),
        _num("10", 243, 837, 270),
        _num("15", 242, 1209, 277),
        _num("20", 241, 1581, 274),
        _num("25", 241, 1953, 273),
        _num("30", 242, 2389, 273),
        _num("35", 241, 2781, 273),
    ]


def test_detects_a_real_epo_margin_run():
    run = detect_line_numbers(_epo_margin(), PAGE_W)
    assert run is not None
    assert run.values == [5, 10, 15, 20, 25, 30, 35]
    assert run.step == 5


def test_tolerates_a_missing_number():
    """A skipped detection leaves a 2*step gap, still consistent with the run."""
    words = [w for w in _epo_margin() if w.text != "20"]
    run = detect_line_numbers(words, PAGE_W)
    assert run is not None
    assert run.step == 5
    assert 20 not in run.values


def test_every_line_numbering_is_detected():
    words = [_num(str(n), 250, 200 + n * 60, 275) for n in range(1, 9)]
    run = detect_line_numbers(words, PAGE_W)
    assert run is not None and run.step == 1


def test_rejects_the_sidebar_publication_number():
    """Regression: 'EP 1 439 083 A3' printed vertically in the margin was
    previously classified as line numbering by position alone."""
    words = [
        _num("EP", 250, 400, 275),
        _num("1", 250, 500, 275),
        _num("439", 250, 600, 275),
        _num("083", 250, 700, 275),
        _num("A3", 250, 800, 275),
    ]
    assert detect_line_numbers(words, PAGE_W) is None


def test_rejects_a_descending_or_irregular_sequence():
    assert detect_line_numbers(
        [_num("30", 250, 400, 275), _num("20", 250, 500, 275), _num("25", 250, 600, 275)],
        PAGE_W) is None
    assert detect_line_numbers(
        [_num("5", 250, 400, 275), _num("12", 250, 500, 275), _num("14", 250, 600, 275)],
        PAGE_W) is None


def test_rejects_numbers_scattered_across_the_page():
    """Figure callouts are numeric but not vertically aligned."""
    words = [
        _num("5", 250, 400, 275),
        _num("10", 900, 500, 940),
        _num("15", 1600, 600, 1650),
    ]
    assert detect_line_numbers(words, PAGE_W) is None


def test_ignores_numbers_outside_the_margin_band():
    words = [_num(str(n * 5), 900, 200 + n * 60, 950) for n in range(1, 6)]
    assert detect_line_numbers(words, PAGE_W) is None


def test_requires_a_minimum_run_length():
    words = [_num("5", 263, 466, 273), _num("10", 243, 837, 270)]
    assert detect_line_numbers(words, PAGE_W) is None


def test_handles_empty_and_degenerate_input():
    assert detect_line_numbers([], PAGE_W) is None
    assert detect_line_numbers(_epo_margin(), 0) is None


def test_looks_like_helper_agrees():
    assert looks_like_line_numbers(_epo_margin(), PAGE_W) is True
    assert looks_like_line_numbers([], PAGE_W) is False
