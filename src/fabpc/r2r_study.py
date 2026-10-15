"""R2R study on the SIMULATED linear-drift process: no control vs EWMA (several
lambda) vs double EWMA, plus a numerical check of the published stability
condition and of the steady-state drift bias."""

from __future__ import annotations

import numpy as np

from .r2r import ControllerSpec, ewma_drift_bias, ewma_stable, simulate, summarize
from .simulator import LinearDriftProcess

PROC = LinearDriftProcess(alpha=35.0, beta=4.0, delta=-0.5, sigma=6.0)
TARGET = 600.0       # nm removed
B_MODEL = 3.4        # model gain (true gain 4.0 -> xi = beta/b = 1.176)
A0 = 0.0             # off-line model intercept (true alpha = 35 -> initial offset)
SPEC_HALF = 20.0     # assumed spec: target +/- 20 nm
N_RUNS = 400
BURN_IN = 50
N_REP = 200
SEED0 = 1000

CONTROLLERS = [
    ControllerSpec("none", label="no control"),
    ControllerSpec("ewma", 0.1, label="EWMA lambda=0.1"),
    ControllerSpec("ewma", 0.3, label="EWMA lambda=0.3"),
    ControllerSpec("ewma", 0.6, label="EWMA lambda=0.6"),
    ControllerSpec("dewma", 0.3, 0.1, label="d-EWMA w1=0.3 w2=0.1"),
]


def run_comparison(n_rep: int = N_REP, n_runs: int = N_RUNS) -> dict:
    lsl, usl = TARGET - SPEC_HALF, TARGET + SPEC_HALF
    per: dict[str, dict[str, list[float]]] = {c.label: {} for c in CONTROLLERS}
    for r in range(n_rep):
        rng = np.random.default_rng(SEED0 + r)
        eps = rng.normal(0.0, PROC.sigma, size=n_runs)   # common random numbers
        for c in CONTROLLERS:
            res = simulate(PROC, c, TARGET, B_MODEL, n_runs, eps, a0=A0)
            s = summarize(res["y"], TARGET, lsl, usl, BURN_IN)
            for k, v in s.items():
                per[c.label].setdefault(k, []).append(v)
    table = []
    for c in CONTROLLERS:
        row = {"controller": c.label, "kind": c.kind, "w1": c.w1, "w2": c.w2}
        for k, vals in per[c.label].items():
            v = np.asarray(vals)
            row[k] = float(v.mean())
            row[k + "_p2.5"] = float(np.quantile(v, 0.025))
            row[k + "_p97.5"] = float(np.quantile(v, 0.975))
        if c.kind == "ewma":
            row["theory_bias"] = ewma_drift_bias(c.w1, PROC.beta, B_MODEL, PROC.delta)
            row["xi_lambda"] = c.w1 * PROC.beta / B_MODEL
        table.append(row)
    return {
        "setup": {"alpha": PROC.alpha, "beta": PROC.beta, "delta": PROC.delta, "sigma": PROC.sigma,
                  "target": TARGET, "b_model": B_MODEL, "a0": A0, "spec_lsl": lsl, "spec_usl": usl,
                  "n_runs": n_runs, "burn_in": BURN_IN, "n_rep": n_rep, "seed0": SEED0},
        "controllers": table,
    }


EWMA_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8)
DEWMA_W1_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
DEWMA_W2_GRID = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
N_REP_SELECT = 100
SEED_SELECT = 50000   # selection replications: seeds 50000.., disjoint from the evaluation seeds SEED0..


def _mean_rmse(c: ControllerSpec, seeds: range, n_runs: int) -> np.ndarray:
    lsl, usl = TARGET - SPEC_HALF, TARGET + SPEC_HALF
    out = []
    for s in seeds:
        eps = np.random.default_rng(s).normal(0.0, PROC.sigma, size=n_runs)
        out.append(summarize(simulate(PROC, c, TARGET, B_MODEL, n_runs, eps, a0=A0)["y"], TARGET, lsl, usl,
                             BURN_IN)["rmse"])
    return np.asarray(out)


def tune(n_select: int = N_REP_SELECT, n_eval: int = N_REP, n_runs: int = N_RUNS) -> dict:
    """Tune EWMA (lambda) and d-EWMA (w1, w2) by mean RMSE on *selection*
    replications, then compare the two winners on independent *evaluation*
    replications (common random numbers, paired)."""
    sel = range(SEED_SELECT, SEED_SELECT + n_select)
    ev = range(SEED0, SEED0 + n_eval)
    grid = [ControllerSpec("ewma", lam, label=f"EWMA lambda={lam:g}") for lam in EWMA_GRID]
    grid += [ControllerSpec("dewma", w1, w2, label=f"d-EWMA w1={w1:g} w2={w2:g}")
             for w1 in DEWMA_W1_GRID for w2 in DEWMA_W2_GRID]
    table = [{"controller": c.label, "kind": c.kind, "w1": c.w1, "w2": c.w2,
              "rmse_selection": float(_mean_rmse(c, sel, n_runs).mean())} for c in grid]
    best = {}
    for kind in ("ewma", "dewma"):
        rows = [r for r in table if r["kind"] == kind]
        b = min(rows, key=lambda r: r["rmse_selection"])
        on_edge = (b["w1"] in (min(EWMA_GRID), max(EWMA_GRID))) if kind == "ewma" else (
            b["w1"] in (min(DEWMA_W1_GRID), max(DEWMA_W1_GRID)) or b["w2"] in (min(DEWMA_W2_GRID), max(DEWMA_W2_GRID)))
        best[kind] = {**b, "on_grid_edge": bool(on_edge)}
    spec = {k: ControllerSpec(k, best[k]["w1"], best[k]["w2"]) for k in best}
    ev_rmse = {k: _mean_rmse(spec[k], ev, n_runs) for k in spec}
    diff = ev_rmse["dewma"] - ev_rmse["ewma"]
    for k in best:
        best[k]["rmse_evaluation"] = float(ev_rmse[k].mean())
        best[k]["rmse_evaluation_p2.5"] = float(np.quantile(ev_rmse[k], 0.025))
        best[k]["rmse_evaluation_p97.5"] = float(np.quantile(ev_rmse[k], 0.975))
    se = float(diff.std(ddof=1) / np.sqrt(diff.size))
    return {"selection_seeds": [sel.start, sel.stop - 1], "evaluation_seeds": [ev.start, ev.stop - 1],
            "n_select": n_select, "n_eval": n_eval, "grid": table, "best": best,
            "paired_rmse_diff_dewma_minus_ewma": {"mean": float(diff.mean()), "se": se,
                                                  "ci95_lo": float(diff.mean() - 1.96 * se),
                                                  "ci95_hi": float(diff.mean() + 1.96 * se),
                                                  "frac_reps_dewma_better": float(np.mean(diff < 0))}}


def example_trajectories(seed: int = SEED0, n_runs: int = N_RUNS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, PROC.sigma, size=n_runs)
    return {c.label: simulate(PROC, c, TARGET, B_MODEL, n_runs, eps, a0=A0)["y"] for c in CONTROLLERS}


def stability_sweep(n_runs: int = 3000) -> dict:
    """Noise-free EWMA-I runs over a grid of xi*lambda; compare empirical
    convergence with the condition 0 < lambda*beta/b < 2 (agreement means:
    theory 'stable' <=> the offset converged to the predicted drift bias)."""
    proc = LinearDriftProcess(PROC.alpha, PROC.beta, PROC.delta, 0.0)
    grid = [round(0.1 * i, 2) for i in range(1, 25)] + [1.95, 2.05]
    grid = sorted(set(grid))
    rows = []
    for g in grid:
        lam = g * B_MODEL / PROC.beta
        with np.errstate(all="ignore"):
            res = simulate(proc, ControllerSpec("ewma", lam), TARGET, B_MODEL, n_runs, np.zeros(n_runs), a0=A0)
        e = res["y"] - TARGET
        bias = ewma_drift_bias(lam, PROC.beta, B_MODEL, PROC.delta)
        last = e[-1]
        if np.isfinite(last) and abs(last - bias) < 1e-6:
            empirical = "converged"
        elif not np.isfinite(last) or abs(last) > 1e6:
            empirical = "diverged"
        else:
            empirical = "not converged (bounded)"
        theory = "stable" if ewma_stable(lam, PROC.beta, B_MODEL) else "unstable"
        agree = (theory == "stable") == (empirical == "converged")
        rows.append({"xi_lambda": g, "lambda": lam, "theory": theory, "empirical": empirical,
                     "final_offset": float(last) if np.isfinite(last) and abs(last) < 1e6 else None,
                     "theory_bias": bias, "agree": agree})
    return {"n_runs": n_runs, "rows": rows, "n_agree": sum(r["agree"] for r in rows), "n_total": len(rows)}


def run() -> dict:
    return {"comparison": run_comparison(), "tuned": tune(), "stability": stability_sweep()}
