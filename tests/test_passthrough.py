from reportlab.pdfgen import canvas

from patent_ocr.passthrough import analyze_text_native


def test_native_pdf_with_sane_text_is_passthrough(tmp_path):
    pdf_path = tmp_path / "native.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(300, 150))
    c.drawString(20, 100, "This is a native text patent claim one, describing a widget.")
    c.save()

    fully_native, flags = analyze_text_native(pdf_path)
    assert fully_native is True
    assert flags == [True]


def test_image_only_pdf_is_not_passthrough(tmp_path):
    from PIL import Image
    from reportlab.lib.utils import ImageReader

    img = Image.new("RGB", (200, 100), "white")
    pdf_path = tmp_path / "scanned.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(200, 100))
    c.drawImage(ImageReader(img), 0, 0, width=200, height=100)
    c.save()

    fully_native, flags = analyze_text_native(pdf_path)
    assert fully_native is False
    assert flags == [False]
