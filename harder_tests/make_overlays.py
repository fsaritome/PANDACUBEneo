"""Render every page with its region and word boxes drawn, for visual review."""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from patent_ocr.config import load_config
from patent_ocr.page_pipeline import process_page_image

root = Path("/home/install/patent_ocr/harder_test")
src = root / "input_backup/claims_test.pdf"
cfg = load_config(root / "prod_like.yaml")

KIND_COLOR = {
    "margin_numbers": (0, 130, 255),
    "column": (220, 0, 0),
    "other": (150, 0, 200),
    "figure": (0, 160, 0),
    "formula": (255, 140, 0),
    "full_page": (100, 100, 100),
}

for page in range(1, 5):
    png = root / f"pg{page}.png"
    subprocess.run(
        ["gs", "-q", "-sDEVICE=png16m", "-r300", f"-dFirstPage={page}",
         f"-dLastPage={page}", "-dNOPAUSE", "-dBATCH", f"-sOutputFile={png}", str(src)],
        check=True,
    )
    result = process_page_image(png, cfg)

    img = Image.open(png).convert("RGB")
    draw = ImageDraw.Draw(img)
    n_words = 0
    for region in result.regions_for_render:
        color = KIND_COLOR.get(region.kind.value, (120, 120, 120))
        draw.rectangle(region.bbox, outline=color, width=7)
        for w in region.words:
            draw.rectangle(w.bbox, outline=color, width=2)
            n_words += 1

    out = root / f"overlay_page{page}.png"
    img.thumbnail((1600, 1600))
    img.save(out)
    kinds = {}
    for r in result.regions_for_render:
        kinds[r.kind.value] = kinds.get(r.kind.value, 0) + 1
    print(f"page {page}: {len(result.regions_for_render):2} regions {kinds}  words={n_words}  "
          f"layout={result.qc['layout_type']}")
    png.unlink(missing_ok=True)
