from pathlib import Path

from patent_ocr.qc import staging_dir


def test_staging_dirs_differ_per_output_file():
    """GPU mode runs files concurrently in threads sharing os.environ, so a
    global PATENT_OCR_QC_DIR let one file's pages land in another's sidecar
    and silently dropped the loser's DOCX."""
    work = Path("/work")
    a = staging_dir("/out/alpha.pdf", work)
    b = staging_dir("/out/beta.pdf", work)
    assert a != b
    assert a.parent == work and b.parent == work


def test_staging_dir_is_stable_for_the_same_output():
    work = Path("/work")
    assert staging_dir("/out/alpha.pdf", work) == staging_dir("/out/alpha.pdf", work)


def test_staging_dir_name_is_filesystem_safe():
    name = staging_dir("/out/a b/w\u00e4rme (1).pdf", Path("/work")).name
    assert name.isalnum() and len(name) == 16
