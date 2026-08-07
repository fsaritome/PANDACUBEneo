from patent_ocr.ledger import Ledger


def test_needs_processing_new_file(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    assert ledger.needs_processing("/in/a.pdf", "hash1") is True


def test_needs_processing_false_after_done(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.enqueue("/in/a.pdf", "hash1")
    ledger.mark_processing("/in/a.pdf")
    ledger.mark_done(
        "/in/a.pdf", "/out/a.pdf",
        engines_used=["tesseract"], confidence_summary={"mean": 90.0},
        languages=["eng"], layout_type="single_column", fallback_fired=False,
    )
    assert ledger.needs_processing("/in/a.pdf", "hash1") is False


def test_needs_processing_true_after_hash_change(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.enqueue("/in/a.pdf", "hash1")
    ledger.mark_done(
        "/in/a.pdf", "/out/a.pdf",
        engines_used=[], confidence_summary={}, languages=[],
        layout_type=None, fallback_fired=False,
    )
    assert ledger.needs_processing("/in/a.pdf", "hash2") is True


def test_needs_processing_true_after_failure(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.enqueue("/in/a.pdf", "hash1")
    ledger.mark_failed("/in/a.pdf", "boom")
    assert ledger.needs_processing("/in/a.pdf", "hash1") is True


def test_flagged_report_only_returns_flagged(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.enqueue("/in/a.pdf", "hash1")
    ledger.mark_done(
        "/in/a.pdf", "/out/a.pdf",
        engines_used=[], confidence_summary={}, languages=[],
        layout_type=None, fallback_fired=True, flagged=True, flag_reason="low confidence",
    )
    ledger.enqueue("/in/b.pdf", "hash2")
    ledger.mark_done(
        "/in/b.pdf", "/out/b.pdf",
        engines_used=[], confidence_summary={}, languages=[],
        layout_type=None, fallback_fired=False,
    )
    flagged = ledger.flagged_report()
    assert [r.input_path for r in flagged] == ["/in/a.pdf"]
