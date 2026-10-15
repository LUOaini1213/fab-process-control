"""SECOM loading, integrity check, time-ordered split and leakage-safe preprocessing.

Data: McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C54305 (CC BY 4.0).

The raw files are NOT part of this repository. Point ``SECOM_DIR`` at the
folder that contains ``secom.data`` and ``secom_labels.data`` (see
``scripts/fetch_secom.py``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# SHA-256 of the files inside https://archive.ics.uci.edu/static/public/179/secom.zip
# (downloaded 2026-10-02).
SHA256 = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
}

# Time-ordered split boundaries (fractions of rows; rows are sorted by time).
FIT_END = 0.50   # rows [0, 50%)   : model fitting / Phase I estimation
CAL_END = 0.65   # rows [50%, 65%) : calibration (limits, thresholds, C choice)
#                  rows [65%, 100%): held-out test / Phase II monitoring

MAX_MISSING_FRAC = 0.20     # sensors with more missing values in the fit period are dropped
NEAR_CONSTANT_SHARE = 0.95  # sensor dropped if its most frequent value covers >= 95% of fit rows


def default_dir() -> Path:
    return Path(os.environ.get("SECOM_DIR", "data/secom"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_available(data_dir: str | Path | None = None) -> bool:
    d = Path(data_dir) if data_dir is not None else default_dir()
    return all((d / name).is_file() for name in SHA256)


@dataclass
class Secom:
    X: np.ndarray            # (n, p) float, NaN = missing
    y: np.ndarray            # (n,) int, 1 = fail, 0 = pass
    t: pd.DatetimeIndex      # (n,) timestamps
    sensors: list[str]       # sensor names "S000".."S589" (column order of the raw file)


def load(data_dir: str | Path | None = None, verify: bool = True) -> Secom:
    d = Path(data_dir) if data_dir is not None else default_dir()
    if verify:
        for name, digest in SHA256.items():
            got = sha256_file(d / name)
            if got != digest:
                raise ValueError(f"SHA-256 mismatch for {name}: {got}")
    X = pd.read_csv(d / "secom.data", sep=r"\s+", header=None).to_numpy(dtype=float)
    lab = pd.read_csv(d / "secom_labels.data", sep=r"\s+", header=None, names=["y", "ts"])
    y = (lab["y"].to_numpy() == 1).astype(int)
    t = pd.DatetimeIndex(pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S"))
    if len(X) != len(y):
        raise ValueError("row count mismatch between data and labels")
    if not t.is_monotonic_increasing:
        raise ValueError("SECOM rows are expected to be time-ordered")
    sensors = [f"S{j:03d}" for j in range(X.shape[1])]
    return Secom(X=X, y=y, t=t, sensors=sensors)


def split_indices(n: int, fit_end: float = FIT_END, cal_end: float = CAL_END,
                  test_end: float = 1.0) -> dict[str, np.ndarray]:
    """Contiguous, time-ordered index blocks: fit = [0, fit_end), cal = [fit_end, cal_end),
    test = [cal_end, test_end) as fractions of the rows."""
    if not 0.0 < fit_end < cal_end < test_end <= 1.0:
        raise ValueError("need 0 < fit_end < cal_end < test_end <= 1")
    i = int(round(fit_end * n))
    j = int(round(cal_end * n))
    k = int(round(test_end * n))
    return {"fit": np.arange(0, i), "cal": np.arange(i, j), "test": np.arange(j, k)}


def most_frequent_share(col: np.ndarray) -> float:
    v = col[~np.isnan(col)]
    if v.size == 0:
        return 1.0
    _, counts = np.unique(v, return_counts=True)
    return float(counts.max() / v.size)


def select_sensors(X_fit: np.ndarray,
                   max_missing: float = MAX_MISSING_FRAC,
                   near_constant: float = NEAR_CONSTANT_SHARE) -> np.ndarray:
    """Boolean mask of usable sensors, decided on the fit period only."""
    miss = np.isnan(X_fit).mean(axis=0)
    keep = miss <= max_missing
    for j in np.where(keep)[0]:
        col = X_fit[:, j]
        v = col[~np.isnan(col)]
        if v.size < 2 or np.nanstd(v) == 0.0 or most_frequent_share(col) >= near_constant:
            keep[j] = False
    return keep


@dataclass
class Standardizer:
    """Median imputation + z-scoring, parameters estimated on reference rows only."""

    keep: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X_ref: np.ndarray, keep: np.ndarray) -> "Standardizer":
        Z = X_ref[:, keep]
        med = np.nanmedian(Z, axis=0)
        Zi = np.where(np.isnan(Z), med, Z)
        mean = Zi.mean(axis=0)
        std = Zi.std(axis=0, ddof=1)
        std = np.where(std > 0, std, 1.0)
        return cls(keep=keep, median=med, mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X[:, self.keep]
        Zi = np.where(np.isnan(Z), self.median, Z)
        return (Zi - self.mean) / self.std
