from patent_ocr.ocr.wordsplit import split_block_into_words, split_line_into_words


def test_words_stay_inside_the_line_box():
    words = split_line_into_words("the quick brown fox", (100, 50, 300, 70), 90.0, "e")
    assert [w.text for w in words] == ["the", "quick", "brown", "fox"]
    assert words[0].bbox[0] == 100
    # The old implementation excluded spaces from the denominator while advancing
    # the cursor past them, so the final box overran the right edge.
    assert words[-1].bbox[2] <= 300


def test_word_boxes_are_ordered_and_non_overlapping():
    words = split_line_into_words("alpha beta gamma delta", (0, 0, 400, 20), 90.0, "e")
    for left, right in zip(words, words[1:]):
        assert left.bbox[2] <= right.bbox[0]


def test_line_vertical_extent_is_preserved():
    words = split_line_into_words("one two", (10, 30, 90, 45), 90.0, "e")
    assert all(w.bbox[1] == 30 and w.bbox[3] == 45 for w in words)


def test_single_token_keeps_the_measured_box_exactly():
    words = split_line_into_words("solo", (5, 5, 55, 25), 88.0, "e")
    assert len(words) == 1
    assert words[0].bbox == (5, 5, 55, 25)


def test_block_splits_into_per_line_bands():
    block = "first line here\nsecond line here\nthird line here"
    words = split_block_into_words(block, (0, 0, 300, 90), 95.0, "e")
    bands = sorted({(w.bbox[1], w.bbox[3]) for w in words})
    # Three distinct vertical bands, not every word spanning the whole block.
    assert bands == [(0, 30), (30, 60), (60, 90)]


def test_block_words_do_not_span_full_block_height():
    words = split_block_into_words("a b\nc d", (0, 0, 100, 100), 95.0, "e")
    assert all(w.bbox[3] - w.bbox[1] < 100 for w in words)


def test_blank_input_yields_no_words():
    assert split_line_into_words("   ", (0, 0, 10, 10), 90.0, "e") == []
    assert split_block_into_words("\n\n", (0, 0, 10, 10), 90.0, "e") == []
