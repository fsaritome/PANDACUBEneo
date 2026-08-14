"""Did ppstructure actually lose margin line-numbers, or just not isolate them?"""
import subprocess
import sys
from pathlib import Path

from patent_ocr.config import load_config
from patent_ocr.page_pipeline import process_page_image

pdf = Path(sys.argv[1])
png = Path("/tmp/margin_check.png")
subprocess.run(
    ["gs", "-q", "-sDEVICE=png16m", "-r300", "-dFirstPage=1", "-dLastPage=1",
     "-dNOPAUSE", "-dBATCH", f"-sOutputFile={png}", str(pdf)],
    check=True,
)

for name, cfgpath in (
    ("heuristic", "/home/install/patent_ocr/harder_test/config_heuristic.yaml"),
    ("ppstructure", "/home/install/patent_ocr/harder_test/config_ppstructure.yaml"),
):
    cfg = load_config(cfgpath)
    res = process_page_image(png, cfg)
    print(f"\n{'='*70}\n### {name}")
    for r in res.regions_for_render[:6]:
        preview = " ".join(w.text for w in r.words)[:70]
        print(f"  [{r.order_index}] {r.kind.value:16} n={len(r.words):4} {preview!r}")
    print(f"  --- first 220 chars of reading-order text ---")
    print("  " + repr(res.text[:220]))
    nums = [w.text for r in res.regions_for_render for w in r.words
            if w.text.strip() in {"5", "10", "15", "20", "25", "30", "35"}]
    print(f"  line-number tokens present: {nums}")
