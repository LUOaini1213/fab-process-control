"""Run-to-run (R2R) controllers for a single-input, single-output process.

EWMA with intercept updating (EWMA-I), as written in Wang & He (2008), eqs. (3)-(4):
    u[n+1] = (T - a[n]) / b
    a[n]   = w * (y[n] - b * u[n]) + (1 - w) * a[n-1]
Stability (Ingolfsson & Sachs 1993, as restated by Wang & He 2008, Table I):
    stable iff 0 < w * beta / b < 2.
With a deterministic intercept drift n*delta, the steady-state bias is
    y - T = b * delta / (beta * w)        (Wang & He 2008, Sec. III-B).

Double EWMA (d-EWMA), Chen & Guo (2001), eqs. (6) and (12):
    a[i] = w1 * (Y_i - b X_i) + (1 - w1) * (a[i-1] + p[i-1])
    p[i] = w2 * (Y_i - b X_i - a[i-1]) + (1 - w2) * p[i-1]
    X[i+1] = (T - (a[i] + p[i])) / b
(the paper writes beta for the gain and assumes it known; here b is the
model gain so plant-model mismatch can be studied.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulator import LinearDriftProcess


@dataclass(frozen=True)
class ControllerSpec:
    kind: str            # "none" | "ewma" | "dewma"
    w1: float = 0.0      # EWMA weight (lambda) for the intercept
    w2: float = 0.0      # d-EWMA drift weight
    label: str = ""


def simulate(proc: LinearDriftProcess, ctrl: ControllerSpec, target: float, b: float,
             n_runs: int, eps: np.ndarray, a0: float = 0.0) -> dict[str, np.ndarray]:
    """Closed-loop simulation. ``eps`` holds the noise for runs 1..n_runs.

    ``a0`` is the initial intercept estimate (the off-line model intercept).
    With kind == "none" the recipe stays at (T - a0) / b for every run.
    """
    eps = np.asarray(eps, dtype=float)
    if eps.size != n_runs:
        raise ValueError("eps must have n_runs entries")
    if ctrl.kind not in ("none", "ewma", "dewma"):
        raise ValueError(f"unknown controller {ctrl.kind!r}")
    y = np.empty(n_runs)
    u = np.empty(n_runs)
    a = a0
    p = 0.0
    u_next = (target - a0) / b
    for i in range(n_runs):
        n = i + 1
        u[i] = u_next
        y[i] = proc.output(n, u[i], eps[i])
        resid = y[i] - b * u[i]
        if ctrl.kind == "ewma":
            a = ctrl.w1 * resid + (1.0 - ctrl.w1) * a
            u_next = (target - a) / b
        elif ctrl.kind == "dewma":
            a_prev = a
            a = ctrl.w1 * resid + (1.0 - ctrl.w1) * (a_prev + p)
            p = ctrl.w2 * (resid - a_prev) + (1.0 - ctrl.w2) * p
            u_next = (target - (a + p)) / b
        # kind == "none": recipe unchanged
    return {"y": y, "u": u}


def ewma_stable(w: float, beta: float, b: float) -> bool:
    """Ingolfsson & Sachs stability condition 0 < w*beta/b < 2."""
    g = w * beta / b
    return 0.0 < g < 2.0


def ewma_drift_bias(w: float, beta: float, b: float, delta: float) -> float:
    """Asymptotic offset y - T of EWMA-I under intercept drift (Wang & He 2008)."""
    return b * delta / (beta * w)


def summarize(y: np.ndarray, target: float, lsl: float, usl: float, burn_in: int) -> dict[str, float]:
    yy = np.asarray(y, dtype=float)[burn_in:]
    e = yy - target
    return {
        "mean_offset": float(e.mean()),
        "std": float(yy.std(ddof=1)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "pct_in_spec": float(100.0 * np.mean((yy >= lsl) & (yy <= usl))),
    }
