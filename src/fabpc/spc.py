"""Univariate SPC: Shewhart X-bar/R and individuals/MR charts, EWMA, tabular CUSUM,
Western Electric (WECO) run rules and capability indices.

Formulas and constants follow the NIST/SEMATECH e-Handbook of Statistical
Methods, sections 6.3.2 - 6.3.2.4 and 6.1.6 (see docs/sources.md).
"""

from __future__ import annotations

import numpy as np

# Factors for X-bar and R charts, NIST e-Handbook 6.3.2.1 table: n -> (A2, D3, D4)
XBAR_R_FACTORS: dict[int, tuple[float, float, float]] = {
    2: (1.880, 0.0, 3.267),
    3: (1.023, 0.0, 2.575),
    4: (0.729, 0.0, 2.282),
    5: (0.577, 0.0, 2.115),
    6: (0.483, 0.0, 2.004),
    7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864),
    9: (0.337, 0.184, 1.816),
    10: (0.308, 0.223, 1.777),
}

# d2 for n = 2 (NIST e-Handbook 6.3.2.2: "1.128 is the value of d2 for n = 2")
D2_N2 = 1.128


def xbar_r_limits(subgroups: np.ndarray) -> dict[str, float]:
    """Phase I limits for X-bar and R charts from an (m, n) array of subgroups."""
    sg = np.asarray(subgroups, dtype=float)
    if sg.ndim != 2:
        raise ValueError("subgroups must be a 2-D array (m subgroups x n)")
    m, n = sg.shape
    if n not in XBAR_R_FACTORS:
        raise ValueError(f"subgroup size {n} not tabulated (2..10)")
    a2, d3, d4 = XBAR_R_FACTORS[n]
    xbar = sg.mean(axis=1)
    r = sg.max(axis=1) - sg.min(axis=1)
    xbb = float(xbar.mean())
    rbar = float(r.mean())
    return {
        "n": n,
        "m": m,
        "xbar_center": xbb,
        "xbar_ucl": xbb + a2 * rbar,
        "xbar_lcl": xbb - a2 * rbar,
        "r_center": rbar,
        "r_ucl": d4 * rbar,
        "r_lcl": d3 * rbar,
    }


def moving_range(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.abs(np.diff(x))


def individuals_limits(x: np.ndarray) -> dict[str, float]:
    """Individuals (I) and moving-range (MR) chart limits (NIST 6.3.2.2)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two observations")
    mr = moving_range(x)
    mrbar = float(mr.mean())
    center = float(x.mean())
    sigma = mrbar / D2_N2
    _, d3, d4 = XBAR_R_FACTORS[2]
    return {
        "center": center,
        "sigma": sigma,
        "ucl": center + 3.0 * sigma,
        "lcl": center - 3.0 * sigma,
        "mr_center": mrbar,
        "mr_ucl": d4 * mrbar,
        "mr_lcl": d3 * mrbar,
    }


def phase1_individuals(x: np.ndarray, max_iter: int = 10) -> tuple[dict[str, float], np.ndarray]:
    """Phase I: estimate I-MR limits, drop points beyond the 3-sigma limits and
    re-estimate until no point is excluded (or ``max_iter`` passes).

    Returns the final limits and the boolean mask of points retained.
    """
    x = np.asarray(x, dtype=float)
    keep = np.ones(x.size, dtype=bool)
    lim = individuals_limits(x)
    for _ in range(max_iter):
        out = (x > lim["ucl"]) | (x < lim["lcl"])
        new_keep = keep & ~out
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
        lim = individuals_limits(x[keep])
    return lim, keep


def ewma(x: np.ndarray, lam: float, start: float) -> np.ndarray:
    """EWMA_t = lam * Y_t + (1 - lam) * EWMA_{t-1}, EWMA_0 = start (NIST 6.3.2.4).

    Returns EWMA_1..EWMA_n (length n, EWMA_0 excluded).
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError("lambda must be in (0, 1]")
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    prev = float(start)
    for i, v in enumerate(x):
        prev = lam * v + (1.0 - lam) * prev
        out[i] = prev
    return out


def ewma_limits(center: float, s: float, lam: float, k: float = 3.0) -> dict[str, float]:
    """Asymptotic EWMA limits: center +/- k * s * sqrt(lam / (2 - lam)) (NIST 6.3.2.4)."""
    s_ewma = s * np.sqrt(lam / (2.0 - lam))
    return {"center": center, "ucl": center + k * s_ewma, "lcl": center - k * s_ewma, "s_ewma": float(s_ewma)}


def tabular_cusum(x: np.ndarray, mu0: float, k: float, h: float) -> dict[str, np.ndarray]:
    """Tabular CUSUM (NIST 6.3.2.3):
    S_hi(i) = max(0, S_hi(i-1) + x_i - mu0 - k)
    S_lo(i) = max(0, S_lo(i-1) + mu0 - k - x_i), S(0) = 0; signal when S > h.
    ``k`` and ``h`` are in the units of ``x``.
    """
    x = np.asarray(x, dtype=float)
    hi = np.empty_like(x)
    lo = np.empty_like(x)
    sh = sl = 0.0
    for i, v in enumerate(x):
        sh = max(0.0, sh + v - mu0 - k)
        sl = max(0.0, sl + mu0 - k - v)
        hi[i] = sh
        lo[i] = sl
    signal = (hi > h) | (lo > h)
    return {"s_hi": hi, "s_lo": lo, "signal": signal}


def cusum_run_lengths(k: float, h: float, n_rep: int, seed: int, shift: float = 0.0,
                      two_sided: bool = True, max_len: int = 1_000_000) -> np.ndarray:
    """Simulated run lengths of the tabular CUSUM above on N(shift, 1) data
    (k, h in sigma units, S(0) = 0; signal when S > h, as in ``tabular_cusum``).

    ``two_sided=False`` uses the upper statistic S_hi only. All replications are
    advanced together; the run length is the index of the first signal.
    """
    rng = np.random.default_rng(seed)
    sh = np.zeros(n_rep)
    sl = np.zeros(n_rep)
    rl = np.zeros(n_rep, dtype=np.int64)
    active = np.arange(n_rep)
    t = 0
    while active.size and t < max_len:
        t += 1
        x = rng.standard_normal(active.size) + shift
        sh[active] = np.maximum(0.0, sh[active] + x - k)
        sl[active] = np.maximum(0.0, sl[active] - x - k)
        sig = sh[active] > h
        if two_sided:
            sig |= sl[active] > h
        rl[active[sig]] = t
        active = active[~sig]
    if active.size:
        raise RuntimeError("some replications did not signal within max_len")
    return rl


def _window_count(flags: np.ndarray, w: int) -> np.ndarray:
    """count[i] = number of True in flags[i-w+1 .. i] (only for i >= w-1; else 0)."""
    f = flags.astype(int)
    c = np.concatenate([[0], np.cumsum(f)])
    out = np.zeros(f.size, dtype=int)
    if f.size >= w:
        out[w - 1:] = c[w:] - c[:-w]
    return out


def weco_rules(x: np.ndarray, center: float, sigma: float) -> dict[str, np.ndarray]:
    """Western Electric rules as listed in NIST e-Handbook 6.3.2.

    Each entry is a boolean array; True at index i means the rule fires on the
    window that ends at point i.

    rule1: any point beyond +/-3 sigma
    rule2: 2 of the last 3 points beyond +2 sigma (or all on the -2 sigma side)
    rule3: 4 of the last 5 points beyond +1 sigma (or all on the -1 sigma side)
    rule4: 8 consecutive points on the same side of the center line
    trend6: 6 points in a row steadily increasing or decreasing
    alt14: 14 points in a row alternating up and down
    """
    x = np.asarray(x, dtype=float)
    z = (x - center) / sigma
    out: dict[str, np.ndarray] = {}
    out["rule1"] = np.abs(z) > 3.0
    out["rule2"] = (_window_count(z > 2.0, 3) >= 2) | (_window_count(z < -2.0, 3) >= 2)
    out["rule3"] = (_window_count(z > 1.0, 5) >= 4) | (_window_count(z < -1.0, 5) >= 4)
    out["rule4"] = (_window_count(z > 0.0, 8) == 8) | (_window_count(z < 0.0, 8) == 8)
    d = np.diff(x)
    up = np.concatenate([[False], d > 0])
    down = np.concatenate([[False], d < 0])
    # 6 points steadily increasing = 5 consecutive increases ending at i
    out["trend6"] = (_window_count(up, 5) == 5) | (_window_count(down, 5) == 5)
    # 14 points alternating = 13 consecutive differences alternating in sign
    sgn = np.sign(d)
    alt = np.concatenate([[False, False], (sgn[1:] * sgn[:-1]) < 0])
    out["alt14"] = _window_count(alt, 12) == 12
    return out


def capability(x: np.ndarray, lsl: float, usl: float) -> dict[str, float]:
    """Sample Cp and Cpk (NIST e-Handbook 6.1.6), s with ddof = 1."""
    x = np.asarray(x, dtype=float)
    if usl <= lsl:
        raise ValueError("usl must exceed lsl")
    m = float(x.mean())
    s = float(x.std(ddof=1))
    cp = (usl - lsl) / (6.0 * s)
    cpk = min((usl - m) / (3.0 * s), (m - lsl) / (3.0 * s))
    return {"mean": m, "s": s, "cp": cp, "cpk": cpk,
            "frac_out_of_spec": float(np.mean((x < lsl) | (x > usl)))}
