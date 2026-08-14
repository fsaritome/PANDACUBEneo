"""A/B the two layout backends across a corpus sample and report aggregate metrics.

Reports the things that actually indicate breakage at scale: pages yielding no
text, dropped margin line-numbers, region counts, confidence distribution and
per-page latency.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from patent_ocr.config import load_config
from patent_ocr.page_pipeline import process_page_image

WORK = Path("/tmp/ab_pages")


def rasterize(pdf: Path, page: int, dpi: int = 300) -> Path | None:
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"{pdf.stem[:40]}_{page}.png"
    r = subprocess.run(
        ["gs", "-q", "-sDEVICE=png16m", f"-r{dpi}", f"-dFirstPage={page}",
         f"-dLastPage={page}", "-dNOPAUSE", "-dBATCH", f"-sOutputFile={out}", str(pdf)],
        capture_output=True,
    )
    return out if r.returncode == 0 and out.exists() else None


def run(cfg, png: Path) -> dict:
    t0 = time.time()
    res = process_page_image(png, cfg)
    elapsed = time.time() - t0
    words = [w for r in res.regions_for_render for w in r.words]
    confs = [w.confidence for w in words]
    kinds = [r.kind.value for r in res.regions_for_render]
    return {
        "seconds": elapsed,
        "words": len(words),
        "chars": len(res.text.strip()),
        "regions": len(res.regions_for_render),
        "kinds": kinds,
        "has_margin": "margin_numbers" in kinds,
        "mean_conf": sum(confs) / len(confs) if confs else 0.0,
        "layout": res.qc["layout_type"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.corpus)
    pdfs = sorted(p for p in root.rglob("*") if p.suffix.lower() == ".pdf")
    random.Random(args.seed).shuffle(pdfs)
    pdfs = pdfs[: args.n]
    print(f"sampling {len(pdfs)} pdfs from {args.corpus}\n")

    heur = load_config("/home/install/patent_ocr/harder_test/config_heuristic.yaml")
    ppst = load_config("/home/install/patent_ocr/harder_test/config_ppstructure.yaml")

    agg: dict[str, list[dict]] = {"heuristic": [], "ppstructure": []}
    failures: list[str] = []

    for i, pdf in enumerate(pdfs, 1):
        png = rasterize(pdf, 1)
        if png is None:
            failures.append(f"{pdf.name}: rasterize failed")
            continue
        row = {"file": pdf.name}
        for name, cfg in (("heuristic", heur), ("ppstructure", ppst)):
            try:
                row[name] = run(cfg, png)
                agg[name].append(row[name])
            except Exception as exc:
                failures.append(f"{pdf.name} [{name}]: {type(exc).__name__}: {exc}")
        h, p = row.get("heuristic"), row.get("ppstructure")
        if h and p:
            print(f"{i:3}. {pdf.name[:44]:44} "
                  f"heur:{h['chars']:6}ch/{h['regions']:2}r/{h['seconds']:5.2f}s "
                  f"pps:{p['chars']:6}ch/{p['regions']:2}r/{p['seconds']:5.2f}s "
                  f"{'MARGIN-LOST' if h['has_margin'] and not p['has_margin'] else ''}")
        png.unlink(missing_ok=True)

    print("\n" + "=" * 78)
    for name, rows in agg.items():
        if not rows:
            continue
        n = len(rows)
        empty = sum(1 for r in rows if r["chars"] < 50)
        print(f"\n### {name}  (n={n})")
        print(f"  empty/near-empty pages : {empty}  ({100*empty/n:.0f}%)")
        print(f"  mean chars/page        : {sum(r['chars'] for r in rows)/n:.0f}")
        print(f"  mean words/page        : {sum(r['words'] for r in rows)/n:.0f}")
        print(f"  mean regions/page      : {sum(r['regions'] for r in rows)/n:.1f}")
        print(f"  mean confidence        : {sum(r['mean_conf'] for r in rows)/n:.1f}")
        print(f"  mean seconds/page      : {sum(r['seconds'] for r in rows)/n:.2f}")
        print(f"  pages with margin nums : {sum(1 for r in rows if r['has_margin'])}")

    h, p = agg["heuristic"], agg["ppstructure"]
    if h and p:
        lost = sum(1 for a, b in zip(h, p) if a["has_margin"] and not b["has_margin"])
        gained = sum(1 for a, b in zip(h, p) if b["chars"] > a["chars"] * 1.05)
        worse = sum(1 for a, b in zip(h, p) if b["chars"] < a["chars"] * 0.95)
        print(f"\n### head-to-head")
        print(f"  ppstructure extracted MORE text : {gained}/{len(h)}")
        print(f"  ppstructure extracted LESS text : {worse}/{len(h)}")
        print(f"  margin numbers lost by ppstruct : {lost}/{len(h)}")

    if failures:
        print(f"\n### failures ({len(failures)})")
        for f in failures[:20]:
            print("  ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
