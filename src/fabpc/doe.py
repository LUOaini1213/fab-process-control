"""Design of experiments: two-level full / fractional factorials, effect estimates,
ANOVA via orthogonal contrasts and OLS, rotatable central composite designs and
the stationary point of a fitted quadratic.

References (docs/sources.md): NIST/SEMATECH e-Handbook 5.3.3.3-5.3.3.4 (factorial
and fractional designs; effect = mean(high) - mean(low); 2^(4-1) with 4 = 123),
5.3.3.6.1 (CCD, alpha = (2^k)^(1/4) for rotatability), 5.5.3.1.4 and its
Technical Appendix 5C (x* = -1/2 B^-1 b).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


def full_factorial(k: int) -> np.ndarray:
    """2^k design in standard (Yates) order, coded -1/+1; factor 1 changes fastest."""
    if k < 1:
        raise ValueError("k >= 1")
    n = 2 ** k
    D = np.empty((n, k))
    for j in range(k):
        D[:, j] = np.where((np.arange(n) // (2 ** j)) % 2 == 0, -1.0, 1.0)
    return D


def fractional_2_4_1() -> np.ndarray:
    """2^(4-1) resolution IV design with generator 4 = 123 (defining relation I = 1234)."""
    base = full_factorial(3)
    return np.column_stack([base, base[:, 0] * base[:, 1] * base[:, 2]])


def term_columns(D: np.ndarray, max_order: int | None = None) -> tuple[list[str], np.ndarray]:
    """All interaction columns up to ``max_order`` (names like '1', '12', '123')."""
    D = np.asarray(D, dtype=float)
    k = D.shape[1]
    max_order = k if max_order is None else max_order
    names: list[str] = []
    cols: list[np.ndarray] = []
    for order in range(1, max_order + 1):
        for combo in combinations(range(k), order):
            names.append("".join(str(c + 1) for c in combo))
            cols.append(np.prod(D[:, list(combo)], axis=1))
    return names, np.column_stack(cols)


def effects(D: np.ndarray, y: np.ndarray, max_order: int | None = None) -> dict[str, float]:
    """Effect of each term = mean(y | column = +1) - mean(y | column = -1)."""
    y = np.asarray(y, dtype=float)
    names, C = term_columns(D, max_order)
    out = {}
    for name, c in zip(names, C.T):
        out[name] = float(y[c > 0].mean() - y[c < 0].mean())
    return out


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Ordinary least squares with classical standard errors and t-tests."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    df = n - p
    if df <= 0:
        raise ValueError("no residual degrees of freedom")
    s2 = sse / df
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pval = 2.0 * stats.t.sf(np.abs(t), df)
    sst = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": t, "p": pval, "sse": sse, "df": df, "s2": s2,
            "r2": 1.0 - sse / sst if sst > 0 else float("nan"), "resid": resid}


def factorial_anova(D: np.ndarray, y: np.ndarray, max_order: int | None = None,
                    extra: dict[str, np.ndarray] | None = None) -> dict[str, dict[str, float]]:
    """ANOVA for a two-level design (optionally with centre points / extra columns).

    Model: intercept + all factorial terms up to ``max_order`` (+ ``extra`` columns,
    e.g. a centre-point curvature indicator). Each term's sum of squares is its
    extra sum of squares (SSE without the term - SSE with it); F = SS / MSE on
    (1, df_error). For an orthogonal two-level design this equals the classical
    contrast sum of squares.
    """
    y = np.asarray(y, dtype=float)
    names, C = term_columns(D, max_order)
    cols = [np.ones(len(y))] + list(C.T)
    labels = ["intercept"] + names
    if extra:
        for k_, v in extra.items():
            cols.append(np.asarray(v, dtype=float))
            labels.append(k_)
    X = np.column_stack(cols)
    full = fit_ols(X, y)
    mse = full["s2"]
    out: dict[str, dict[str, float]] = {}
    for j, lab in enumerate(labels):
        if lab == "intercept":
            continue
        Xr = np.delete(X, j, axis=1)
        red = fit_ols(Xr, y) if Xr.shape[0] - Xr.shape[1] > 0 else None
        ss = (red["sse"] - full["sse"]) if red is not None else float("nan")
        f = ss / mse
        out[lab] = {"coef": float(full["beta"][j]), "ss": float(ss), "f": float(f),
                    "p": float(stats.f.sf(f, 1, full["df"]))}
    out["_error"] = {"ss": float(full["sse"]), "df": float(full["df"]), "ms": float(mse), "r2": float(full["r2"])}
    return out


def rotatable_alpha(n_factorial_runs: int) -> float:
    """alpha = (number of factorial runs)^(1/4) (NIST 5.3.3.6.1)."""
    return float(n_factorial_runs) ** 0.25


def ccd(k: int, n_center: int, alpha: float | None = None) -> np.ndarray:
    """Circumscribed central composite design: 2^k cube + 2k axial + n_center centre runs."""
    cube = full_factorial(k)
    a = rotatable_alpha(2 ** k) if alpha is None else alpha
    axial = np.zeros((2 * k, k))
    for j in range(k):
        axial[2 * j, j] = -a
        axial[2 * j + 1, j] = a
    centre = np.zeros((n_center, k))
    return np.vstack([cube, axial, centre])


def quad_model_matrix(X: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Columns: 1, x_i, x_i^2, x_i x_j (i < j)."""
    X = np.asarray(X, dtype=float)
    k = X.shape[1]
    names = ["1"] + [f"x{i+1}" for i in range(k)] + [f"x{i+1}^2" for i in range(k)]
    cols = [np.ones(len(X))] + [X[:, i] for i in range(k)] + [X[:, i] ** 2 for i in range(k)]
    for i, j in combinations(range(k), 2):
        names.append(f"x{i+1}x{j+1}")
        cols.append(X[:, i] * X[:, j])
    return names, np.column_stack(cols)


def quad_to_bB(beta: np.ndarray, k: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Split quadratic coefficients into b0, b (linear) and symmetric B, with
    B_ii = b_ii and B_ij = b_ij / 2 (NIST 5.5.3.1.4, Technical Appendix 5C)."""
    beta = np.asarray(beta, dtype=float)
    b0 = float(beta[0])
    b = beta[1:1 + k].copy()
    B = np.diag(beta[1 + k:1 + 2 * k])
    idx = 1 + 2 * k
    for i, j in combinations(range(k), 2):
        B[i, j] = B[j, i] = beta[idx] / 2.0
        idx += 1
    return b0, b, B


def stationary_point(b: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray | str]:
    """x* = -1/2 B^{-1} b and its nature from the eigenvalues of B."""
    xs = -0.5 * np.linalg.solve(B, b)
    ev = np.linalg.eigvalsh(B)
    if np.all(ev > 0):
        nature = "minimum"
    elif np.all(ev < 0):
        nature = "maximum"
    else:
        nature = "saddle"
    return {"x": xs, "eigenvalues": ev, "nature": nature}


def quad_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    _, M = quad_model_matrix(np.atleast_2d(X))
    return M @ beta
