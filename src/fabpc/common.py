"""Shared helpers: deterministic rounding and JSON output.

Results are rounded to 10 significant digits before they are written so that
the files are byte-identical across platforms (Windows / Linux) even when the
last floating-point bits differ.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SIG_DIGITS = 10


def round_sig(x: float, sig: int = SIG_DIGITS) -> float | None:
    """Round a float to ``sig`` significant digits; NaN/inf become None."""
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    r = float(f"{x:.{sig}g}")
    if r == 0.0:  # normalise -0.0
        return 0.0
    return r


def clean(obj: Any, sig: int = SIG_DIGITS) -> Any:
    """Recursively convert numpy types and round floats for JSON output."""
    if isinstance(obj, dict):
        return {str(k): clean(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, sig) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v, sig) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round_sig(float(obj), sig)
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"cannot serialise {type(obj)!r}")


def write_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` as rounded, key-sorted JSON with LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return path


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def fmt(x: float | None, nd: int = 3) -> str:
    """Format a number for markdown tables (fixed decimals)."""
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"
