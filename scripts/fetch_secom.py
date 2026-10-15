"""Download the SECOM dataset from the UCI ML Repository and verify SHA-256.

    python scripts/fetch_secom.py --dest data/secom
    export SECOM_DIR=data/secom   (PowerShell: $env:SECOM_DIR = "data/secom")

Data: McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C54305 - licensed CC BY 4.0.
The raw files are intentionally not committed to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SHA256 = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/secom")
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in (*SHA256, "secom.names"):
            (dest / name).write_bytes(zf.read(name))
    ok = True
    for name, digest in SHA256.items():
        got = hashlib.sha256((dest / name).read_bytes()).hexdigest()
        status = "OK" if got == digest else "MISMATCH"
        ok &= got == digest
        print(f"{name}: {status} {got}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
