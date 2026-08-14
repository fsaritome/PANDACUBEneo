from patent_ocr.ledger import Ledger


def _ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "ledger.sqlite3")


def test_reconcile_clears_rows_left_as_processing(tmp_path):
    """A SIGKILL skips process_file's except/finally, stranding rows as
    'processing' so the dashboard shows them as live work forever."""
    ledger = _ledger(tmp_path)
    ledger.enqueue("/in/a.pdf", "hash-a")
    ledger.mark_processing("/in/a.pdf")
    assert ledger.status_counts().get("processing") == 1

    assert ledger.reconcile_interrupted() == 1

    assert ledger.status_counts().get("processing") is None
    rec = ledger.get("/in/a.pdf")
    assert rec.status == "failed"
    assert "interrupted" in rec.error


def test_reconcile_leaves_done_and_queued_alone(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.enqueue("/in/done.pdf", "h1")
    ledger.mark_done(
        "/in/done.pdf",
        "/out/done.pdf",
        engines_used=[],
        confidence_summary={},
        languages=[],
        layout_type=None,
        fallback_fired=False,
    )
    ledger.enqueue("/in/queued.pdf", "h2")

    assert ledger.reconcile_interrupted() == 0
    assert ledger.get("/in/done.pdf").status == "done"
    assert ledger.get("/in/queued.pdf").status == "queued"


def test_reconcile_is_idempotent(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.enqueue("/in/a.pdf", "hash-a")
    ledger.mark_processing("/in/a.pdf")

    assert ledger.reconcile_interrupted() == 1
    assert ledger.reconcile_interrupted() == 0
