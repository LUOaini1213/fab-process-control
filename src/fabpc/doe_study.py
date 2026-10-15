"""DOE study on the SIMULATED toy CMP non-uniformity response.

Stage 1 - screening: 2^(4-1) resolution IV fraction (I = 1234), 2 replicates.
          Factors with main-effect p < 0.20 are kept (liberal screening level).
Stage 2 - 2^k full factorial on the kept factors (others fixed at 0), 2 replicates
          + 4 centre points; ANOVA incl. a centre-point curvature term.
Stage 3 - rotatable CCD (alpha = (2^k)^(1/4)), 6 centre points; quadratic fit,
          stationary point, comparison with the true simulator optimum and with
          the naive 'best observed run'.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize

from . import doe
from .simulator import TOY_FACTORS, toy_cmp_nonuniformity, toy_observe

NOISE_SD = 0.15
SEED = 2026
N_REPEAT = 200   # repeated end-to-end CCD experiments for the optimum-error distribution
REGION = 2.0     # coded search box for the true optimum
SCREEN_ALPHA = 0.20  # liberal screening level: keep a factor if its main-effect p < 0.20


def true_optimum() -> dict:
    """True minimum of the noise-free toy (x4 is inert, reported as 0).

    Closed form instead of a numerical optimiser (whose last digits differ
    between platforms): with a = x1 - 0.35 and c = x2 + 0.25 the (x1, x2) part
    0.9 a^2 + 0.6 c^2 + 0.45 a c + 0.08 a^4 is convex with its minimum at
    a = c = 0, and the x3 part 0.35 (x3 - 0.15)^2 + 0.3 exp(0.5 x3) is strictly
    convex with its minimum where 0.7 (x3 - 0.15) + 0.15 exp(0.5 x3) = 0,
    found by bracketing root search. ``tests/test_doe_study.py`` checks this
    against a numerical minimiser.
    """
    g = lambda x3: 0.7 * (x3 - 0.15) + 0.15 * np.exp(0.5 * x3)
    x3 = brentq(g, -REGION, REGION, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
    x = np.array([0.35, -0.25, x3, 0.0])
    return {"x": x, "y": float(toy_cmp_nonuniformity(x)[0])}


def numeric_optimum() -> dict:
    """Numerical cross-check of ``true_optimum`` (L-BFGS-B over x1..x3, multi-start)."""
    f = lambda z: float(toy_cmp_nonuniformity(np.r_[z, 0.0])[0])
    best = None
    for start in ([0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [-1.0, 1.0, -0.5]):
        r = minimize(f, np.array(start), method="L-BFGS-B", bounds=[(-REGION, REGION)] * 3,
                     options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000})
        if best is None or r.fun < best.fun:
            best = r
    return {"x": np.r_[best.x, 0.0], "y": float(best.fun)}


def stage1(rng: np.random.Generator) -> dict:
    D = doe.fractional_2_4_1()
    Drep = np.vstack([D, D])
    y = toy_observe(np.column_stack([Drep]), rng, NOISE_SD)
    eff = doe.effects(Drep, y, max_order=2)
    # in the 2^(4-1)_IV design, 2FI columns are aliased in pairs; keep one of each pair
    an = doe.factorial_anova(Drep, y, max_order=1,
                             extra={"12+34": Drep[:, 0] * Drep[:, 1],
                                    "13+24": Drep[:, 0] * Drep[:, 2],
                                    "14+23": Drep[:, 0] * Drep[:, 3]})
    active = [j for j in range(4) if an[str(j + 1)]["p"] < SCREEN_ALPHA]
    return {"design": Drep, "y": y, "effects": {k: eff[k] for k in ("1", "2", "3", "4")}, "active": active,
            "alias_pairs": {"12+34": eff["12"], "13+24": eff["13"], "14+23": eff["14"]},
            "anova": an}


def embed(Xa: np.ndarray, active: list[int]) -> np.ndarray:
    """Place active-factor settings into the 4-factor space (inactive factors at 0)."""
    Xa = np.atleast_2d(Xa)
    X = np.zeros((Xa.shape[0], 4))
    X[:, active] = Xa
    return X


def stage2(rng: np.random.Generator, active: list[int]) -> dict:
    F = doe.full_factorial(len(active))
    Frep = np.vstack([F, F])
    C = np.zeros((4, len(active)))
    X = np.vstack([Frep, C])
    y = toy_observe(embed(X, active), rng, NOISE_SD)
    is_centre = np.r_[np.zeros(len(Frep)), np.ones(len(C))]
    eff = doe.effects(Frep, y[: len(Frep)])
    # factorial columns are 0 at centre points; curvature indicator = 1 at centre points
    an = doe.factorial_anova(X, y, extra={"curvature": is_centre})
    return {"design": X, "y": y, "effects": eff, "anova": an,
            "mean_factorial": float(y[: len(Frep)].mean()), "mean_centre": float(y[len(Frep):].mean())}


def stage3_once(rng: np.random.Generator, active: list[int]) -> dict:
    k = len(active)
    Xa = doe.ccd(k, n_center=6)
    y = toy_observe(embed(Xa, active), rng, NOISE_SD)
    names, M = doe.quad_model_matrix(Xa)
    fit = doe.fit_ols(M, y)
    b0, b, B = doe.quad_to_bB(fit["beta"], k)
    sp = doe.stationary_point(b, B)
    xs = embed(np.asarray(sp["x"]), active)[0]
    best_run = embed(Xa[int(np.argmin(y))], active)[0]
    return {"X": Xa, "y": y, "names": names, "fit": fit, "stationary": sp, "x_star": xs,
            "y_true_at_xstar": float(toy_cmp_nonuniformity(xs)[0]),
            "y_pred_at_xstar": float(doe.quad_predict(fit["beta"], np.asarray(sp["x"])[None, :])[0]),
            "best_run_x": best_run,
            "y_true_at_best_run": float(toy_cmp_nonuniformity(best_run)[0])}


N_SCREEN_SIM = 2000
SCREEN_SIM_SEED = 10000


def screening_power(n_sim: int = N_SCREEN_SIM) -> dict:
    """Repeat the stage-1 screen with fresh noise: how often is each factor kept
    (main-effect p below the screening level, and below 0.05)?  Factor 4 is inert,
    so its rate is the false-inclusion rate."""
    kept = np.zeros((n_sim, 4), dtype=bool)
    kept05 = np.zeros((n_sim, 4), dtype=bool)
    for s in range(n_sim):
        an = stage1(np.random.default_rng(SCREEN_SIM_SEED + s))["anova"]
        p = np.array([an[str(j + 1)]["p"] for j in range(4)])
        kept[s] = p < SCREEN_ALPHA
        kept05[s] = p < 0.05
    names = list(TOY_FACTORS)
    return {"n_sim": n_sim, "seed0": SCREEN_SIM_SEED, "screening_alpha": SCREEN_ALPHA,
            "kept_rate": dict(zip(names, kept.mean(axis=0).tolist())),
            "kept_rate_at_0.05": dict(zip(names, kept05.mean(axis=0).tolist())),
            "all_three_active_kept_and_x4_dropped": float(np.mean(kept[:, :3].all(axis=1) & ~kept[:, 3]))}


def run() -> dict:
    rng = np.random.default_rng(SEED)
    s1 = stage1(rng)
    active = s1["active"]
    s2 = stage2(rng, active)
    s3 = stage3_once(rng, active)
    opt = true_optimum()
    confirm = toy_observe(np.tile(s3["x_star"], (5, 1)), rng, NOISE_SD)

    # repeated experiments: distribution of the optimum error
    dist, regret, regret_best, best_is_centre = [], [], [], []
    rng_rep = np.random.default_rng(SEED + 1)
    for _ in range(N_REPEAT):
        r = stage3_once(rng_rep, active)
        dist.append(float(np.linalg.norm(r["x_star"] - opt["x"])))
        regret.append(r["y_true_at_xstar"] - opt["y"])
        regret_best.append(r["y_true_at_best_run"] - opt["y"])
        best_is_centre.append(bool(np.all(r["best_run_x"] == 0.0)))
    dist, regret, regret_best = map(np.asarray, (dist, regret, regret_best))

    fit = s3["fit"]
    return {
        "factors": list(TOY_FACTORS),
        "noise_sd": NOISE_SD,
        "seed": SEED,
        "stage1_screening": {
            "design": "2^(4-1) resolution IV, I=1234, 2 replicates (16 runs)",
            "effects": s1["effects"], "alias_pair_effects": s1["alias_pairs"],
            "anova": s1["anova"], "screening_alpha": SCREEN_ALPHA,
            "active_factors": [TOY_FACTORS[j] for j in active],
        },
        "stage2_full_factorial": {
            "design": "2^k full factorial on the active factors x2 replicates + 4 centre points, inactive at 0",
            "effects": s2["effects"], "anova": s2["anova"],
            "mean_factorial": s2["mean_factorial"], "mean_centre": s2["mean_centre"],
        },
        "stage3_ccd": {
            "design": "rotatable CCD on the active factors, alpha = (2^k)^(1/4), 6 centre points",
            "alpha": doe.rotatable_alpha(2 ** len(active)),
            "coefficients": dict(zip(s3["names"], fit["beta"])),
            "p_values": dict(zip(s3["names"], fit["p"])),
            "r2": fit["r2"], "resid_sd": float(np.sqrt(fit["s2"])),
            "stationary_point": s3["x_star"], "nature": s3["stationary"]["nature"],
            "eigenvalues_B": s3["stationary"]["eigenvalues"],
            "y_pred_at_xstar": s3["y_pred_at_xstar"], "y_true_at_xstar": s3["y_true_at_xstar"],
            "confirmation_runs_mean": float(confirm.mean()),
            "true_optimum_x": opt["x"], "true_optimum_y": opt["y"],
            "distance_coded": float(np.linalg.norm(s3["x_star"] - opt["x"])),
            "regret": s3["y_true_at_xstar"] - opt["y"],
            "best_observed_run_x": s3["best_run_x"],
            "best_observed_run_regret": s3["y_true_at_best_run"] - opt["y"],
        },
        "repeated_ccd": {
            "n_repeat": N_REPEAT,
            "distance_median": float(np.median(dist)), "distance_p90": float(np.quantile(dist, 0.9)),
            "regret_median": float(np.median(regret)), "regret_p90": float(np.quantile(regret, 0.9)),
            "best_run_regret_median": float(np.median(regret_best)),
            "best_run_regret_p90": float(np.quantile(regret_best, 0.9)),
            "frac_rsm_better_than_best_run": float(np.mean(regret < regret_best)),
            "frac_best_run_is_centre_point": float(np.mean(best_is_centre)),
        },
        "screening_power": screening_power(),
    }
