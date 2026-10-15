"""Static PNG figures (matplotlib, Agg backend, deterministic metadata)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# palette (categorical slots in fixed order) and neutral inks
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "lines.linewidth": 1.2, "svg.hashsalt": "fabpc",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def spc_sensor(name: str, res: dict, ewma_lambda: float, cusum_h: float, path: Path) -> Path:
    s = res["series"]
    lim = res["phase1"]["limits"]
    x1, ret, x2, y2 = s["phase1_x"], s["phase1_retained"], s["phase2_x"], s["phase2_y"]
    n1 = x1.size
    fig, ax = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=False)
    i1 = np.arange(n1)
    i2 = np.arange(n1, n1 + x2.size)
    ax[0].plot(i1, x1, color=C1, lw=0.8, label="Phase I (limit estimation)")
    ax[0].plot(i2, x2, color=C3, lw=0.8, label="Phase II (monitoring)")
    ax[0].plot(i1[~ret], x1[~ret], "x", color=INK2, ms=4, label="excluded in Phase I")
    ax[0].plot(i2[y2 == 1], x2[y2 == 1], "o", color=C2, ms=4, label="fail row (Phase II)")
    for v, ls in ((lim["center"], "-"), (lim["ucl"], "--"), (lim["lcl"], "--")):
        ax[0].axhline(v, color=INK2, ls=ls, lw=0.8)
    ax[0].axvline(n1 - 0.5, color=INK, lw=0.8)
    ax[0].set_title(f"{name}: individuals chart (3-sigma, sigma = MRbar/1.128)")
    ax[0].legend(loc="upper left", fontsize=7, ncol=2)
    ew, ewl = s["ewma"], s["ewma_limits"]
    ax[1].plot(i2, ew, color=C1)
    ax[1].axhline(ewl["ucl"], color=INK2, ls="--", lw=0.8)
    ax[1].axhline(ewl["lcl"], color=INK2, ls="--", lw=0.8)
    ax[1].axhline(ewl["center"], color=INK2, lw=0.8)
    ax[1].set_title(f"EWMA (lambda = {ewma_lambda}), Phase II")
    sig = lim["sigma"]
    ax[2].plot(i2, s["cusum_hi"] / sig, color=C1, label="S_hi")
    ax[2].plot(i2, s["cusum_lo"] / sig, color=C2, label="S_lo")
    ax[2].axhline(cusum_h, color=INK2, ls="--", lw=0.8)
    ax[2].set_title("Tabular CUSUM in sigma units (k = 0.5, h = %g), Phase II" % cusum_h)
    ax[2].set_xlabel("row index (time order, non-missing values)")
    ax[2].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def t2_spe(ex: dict, lt2: float, lspe: float, path: Path) -> Path:
    s = ex["_series"]
    n = s["t2"].size
    idx = np.arange(n)
    fig, ax = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    for a, v, lim, lab in ((ax[0], s["t2"], lt2, "Hotelling T2"), (ax[1], s["spe"], lspe, "SPE (Q)")):
        a.semilogy(idx, v, color=C1, lw=0.8)
        a.axhline(lim, color=INK2, ls="--", lw=0.8)
        f = s["y"] == 1
        a.plot(idx[f], v[f], "o", color=C2, ms=4, label="fail row")
        w = s["win"]
        a.axvspan(w[0] - 0.5, w[-1] + 0.5, color=C4, alpha=0.12, lw=0, label="densest alarm window")
        a.set_ylabel(lab + " (log)")
    ax[0].set_title("PCA monitoring on the test period (80% variance model, 99% empirical limits from calibration passes)")
    ax[0].legend(loc="upper left", fontsize=7)
    ax[1].set_xlabel("test-period row index (time order)")
    fig.tight_layout()
    return _save(fig, path)


def contributions(ex: dict, path: Path, top: int = 10) -> Path:
    s = ex["_series"]
    share = s["spe_share"]
    order = np.argsort(-share, kind="mergesort")[:top]
    names = [s["kept_names"][j] for j in order]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(np.arange(top)[::-1], share[order], color=C1, height=0.6)
    ax.set_yticks(np.arange(top)[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel("mean share of SPE over alarmed rows in the window")
    ax.set_title("SPE contribution plot, excursion window")
    fig.tight_layout()
    return _save(fig, path)


def fdc_methods(methods: list[dict], random_ref: dict, path: Path) -> Path:
    labs = [m["method"] for m in methods]
    pt = np.array([m["pr_auc"]["point"] for m in methods])
    lo = np.array([m["pr_auc"]["lo"] for m in methods])
    hi = np.array([m["pr_auc"]["hi"] for m in methods])
    y = np.arange(len(labs))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(labs) + 1.2))
    ax.errorbar(pt, y, xerr=[pt - lo, hi - pt], fmt="o", color=C1, ms=4, capsize=2, lw=1)
    ax.axvline(random_ref["expected_exact"], color=C2, ls="--", lw=1,
               label=f"random ranking: expected {random_ref['expected_exact']:.3f}")
    ax.axvline(random_ref["simulated_p95"], color=C2, ls=":", lw=1,
               label=f"random ranking: 95th percentile {random_ref['simulated_p95']:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("PR-AUC on held-out test period (95% stratified bootstrap CI)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def r2r_trajectories(traj: dict, target: float, spec_half: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    cols = {"no control": INK2, "EWMA lambda=0.3": C1, "d-EWMA w1=0.3 w2=0.1": C2}
    for lab, c in cols.items():
        y = traj[lab]
        ax.plot(np.arange(1, y.size + 1), y, color=c, lw=0.9, label=lab)
    ax.axhline(target, color=INK, lw=0.8)
    ax.axhspan(target - spec_half, target + spec_half, color=C3, alpha=0.10, lw=0, label="assumed spec")
    ax.set_xlabel("run")
    ax.set_ylabel("removed thickness (nm)")
    ax.set_title("SIMULATION: CMP-like process with intercept drift, one noise realisation")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, path)


def r2r_stability(rows: list[dict], path: Path) -> Path:
    g = np.array([r["xi_lambda"] for r in rows])
    conv = np.array([r["empirical"] == "converged" for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.plot(g[conv], np.ones(conv.sum()), "o", color=C1, ms=6, label="offset converged to predicted bias")
    ax.plot(g[~conv], np.zeros((~conv).sum()), "s", color=C2, ms=6, label="did not converge")
    ax.axvline(2.0, color=INK, ls="--", lw=1, label="theory: stable iff 0 < lambda*beta/b < 2")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("lambda * beta / b")
    ax.set_title("SIMULATION: EWMA-I stability check (noise-free, 3000 runs)")
    ax.legend(loc="center left", fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def doe_effects(effects: dict, path: Path, title: str) -> Path:
    names = list(effects)
    vals = np.array([effects[k] for k in names])
    order = np.argsort(-np.abs(vals), kind="mergesort")
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(np.arange(len(order)), np.abs(vals[order]), color=[C1 if vals[i] > 0 else C2 for i in order], width=0.6)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([names[i] for i in order])
    ax.set_ylabel("|effect| (% non-uniformity)")
    ax.set_title(title + "  (blue = positive, orange = negative)")
    fig.tight_layout()
    return _save(fig, path)


def doe_contours(f_true, f_fit, x_star: np.ndarray, x_opt: np.ndarray, path: Path) -> Path:
    g = np.linspace(-2, 2, 81)
    A, B = np.meshgrid(g, g)
    pts = np.column_stack([A.ravel(), B.ravel()])
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    for a, f, title in ((ax[0], f_true, "true toy surface (x3 at its optimum)"),
                        (ax[1], f_fit, "CCD quadratic fit (x3 at fitted optimum)")):
        Zs = f(pts).reshape(A.shape)
        cs = a.contour(A, B, Zs, levels=12, colors=INK2, linewidths=0.6)
        a.clabel(cs, fontsize=6, fmt="%.1f")
        a.plot(*x_opt[:2], "*", color=C2, ms=11, label="true optimum")
        a.plot(*x_star[:2], "o", color=C1, ms=6, label="fitted stationary point")
        a.set_title(title)
        a.set_xlabel("x1 down-force (coded)")
    ax[0].set_ylabel("x2 platen speed (coded)")
    ax[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def missing_profile(miss_frac: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.hist(miss_frac, bins=np.linspace(0, 1, 41), color=C1, edgecolor=SURFACE, linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("fraction of rows with a missing value")
    ax.set_ylabel("number of sensors (log)")
    ax.set_title("SECOM: missing-value profile per sensor")
    fig.tight_layout()
    return _save(fig, path)
