"""Regenerate every result into a temporary folder and compare SHA-256 with results/.

    SECOM_DIR=... python scripts/repro_check.py [--out repro_report.json]

Used to check that Windows and Linux produce byte-identical outputs. Exit code
1 if any JSON/Markdown file differs. PNGs are compared byte-wise and, if the
bytes differ, pixel-wise (different zlib builds compress identical pixels
differently).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fabpc import cli, report, secom  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same_pixels(a: Path, b: Path) -> bool:
    import numpy as np
    from PIL import Image
    x = np.asarray(Image.open(a))
    y = np.asarray(Image.open(b))
    return x.shape == y.shape and bool((x == y).all())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="optional JSON report path")
    a = ap.parse_args()
    committed = ROOT / "results"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        have_data = secom.data_available()
        if have_data:
            cli.cmd_secom(results=tmp)
        cli.cmd_doe(results=tmp)
        cli.cmd_r2r(results=tmp)
        cli.cmd_fmea(results=tmp)
        cli.cmd_arl(results=tmp)
        report.main(results=tmp, readme=tmp / "no_readme.md", docs=())
        rows = []
        for p in sorted(committed.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(committed).as_posix()
            q = tmp / rel
            if not q.exists():
                rows.append({"file": rel, "status": "not regenerated (needs SECOM data)" if not have_data else "missing"})
                continue
            if sha(p) == sha(q):
                status = "identical"
            elif rel.endswith(".png") and same_pixels(p, q):
                status = "pixel-identical"   # PNG bytes differ only through the zlib build
            else:
                status = "DIFFERENT"
            rows.append({"file": rel, "status": status, "sha256": sha(q)})
    text_bad = [r for r in rows if r["status"] == "DIFFERENT" and not r["file"].endswith(".png")]
    png_bad = [r for r in rows if r["status"] == "DIFFERENT" and r["file"].endswith(".png")]
    summary = {"platform": platform.platform(), "python": platform.python_version(),
               "secom_data": have_data, "n_files": len(rows),
               "n_identical": sum(r["status"] == "identical" for r in rows),
               "text_files_different": [r["file"] for r in text_bad],
               "png_files_different": [r["file"] for r in png_bad], "files": rows}
    for r in rows:
        print(f"{r['status']:>10}  {r['file']}")
    n_pix = sum(r["status"] == "pixel-identical" for r in rows)
    summary["n_pixel_identical_png"] = n_pix
    print(f"{summary['n_identical']}/{summary['n_files']} byte-identical, {n_pix} more pixel-identical, "
          f"on {summary['platform']} (Python {summary['python']})")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 1 if text_bad else 0


if __name__ == "__main__":
    sys.exit(main())
