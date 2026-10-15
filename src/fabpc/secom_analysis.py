"""SECOM analyses: data-quality profile, univariate SPC (Phase I / Phase II),
PCA-based multivariate FDC vs supervised fail prediction, and an excursion
summary used by the worked 8D report.

Leakage rules (enforced here, checked in tests):
* rows are time-ordered; fit < cal < test are contiguous blocks;
* sensor selection, imputation medians, scaling, PCA, model coefficients use
  fit rows only;
* control limits, alarm thresholds, the logistic-regression C and the
  gradient-boosting hyperparameters use cal rows only; test rows are touched
  once, for evaluation;
* the rolling-origin evaluation repeats this with several chronological
  fit / cal / test cut points, each fold refitting everything on its own rows.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import metrics, secom, spc
from .mspc import PCAMonitor, empirical_limit, spe_limit_jm, t2_limit_phase2

FAR = 0.05                 # target false-alarm rate for the detection comparison
LIMIT_Q = 0.99             # control-chart style limit for T^2 / SPE monitoring
N_BOOT = 2000
BOOT_SEED = 7
N_PERM = 2000              # label permutations per method (one-sided test of PR-AUC / ROC-AUC)
PERM_SEED = 11
LR_C_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
# gradient boosting: small grid, chosen by calibration-period PR-AUC (fit rows only for fitting)
GB_FIXED = {"min_samples_leaf": 20, "l2_regularization": 1.0, "class_weight": "balanced",
            "early_stopping": False, "random_state": 0}
GB_FITS = ((0.05, 7), (0.05, 15), (0.1, 7), (0.1, 15))   # (learning_rate, max_leaf_nodes)
GB_ITERS = (100, 200)      # read off one 200-iteration fit with staged predictions
GB_THREADS = 1
# rolling-origin evaluation: (fit_end, cal_end, test_end) as fractions of the rows
ROLLING_FOLDS = ((0.35, 0.50, 0.65), (0.50, 0.65, 0.80), (0.65, 0.80, 1.00))
N_SCAN = 20000             # simulations for the scan-statistic null of the densest alarm window
SCAN_SEED = 5
N_SPC_SENSORS = 3
SUBGROUP_N = 5
EWMA_LAMBDA = 0.2          # NIST 6.3.2.4: lambda usually 0.2-0.3
CUSUM_K = 0.5              # in sigma units (NIST 6.3.2.3 rule of thumb)
CUSUM_H = 5.0              # in sigma units (NIST 6.3.2.3 rule of thumb: h around 4 or 5)
SPEC_K = 4.0               # ASSUMED spec limits: Phase-I mean +/- 4 Phase-I sigma
PHASE1_PASSES = 1          # Phase I: exclude points beyond 3 sigma once, then re-estimate
EXCURSION_WINDOW = 20
VAR_TARGETS = (0.7, 0.8, 0.9)


# --------------------------------------------------------------------------- profile
def profile(d: secom.Secom) -> dict:
    X, y, t = d.X, d.y, d.t
    n, p = X.shape
    miss = np.isnan(X).mean(axis=0)
    std = np.nanstd(X, axis=0)
    const = int(np.sum((std == 0) | np.isnan(std)))
    share = np.array([secom.most_frequent_share(X[:, j]) for j in range(p)])
    sp = secom.split_indices(n)
    month = t.to_period("M").astype(str)
    by_month = {}
    for m in sorted(set(month)):
        sel = month == m
        by_month[m] = {"n": int(sel.sum()), "fails": int(y[sel].sum()), "fail_rate": float(y[sel].mean())}
    gaps = np.diff(t.values).astype("timedelta64[s]").astype(float) / 3600.0
    return {
        "n_rows": n, "n_sensors_in_file": p,
        "n_features_listed_by_uci": 591,
        "note_feature_count": ("The UCI page and secom.names say 591 features ('a 1567 x 591 matrix'); "
                               "every row of secom.data has 590 fields. The discrepancy is in the source's "
                               "own documentation."),
        "n_fail": int(y.sum()), "n_pass": int((1 - y).sum()), "fail_rate": float(y.mean()),
        "time_first": str(t[0]), "time_last": str(t[-1]),
        "time_monotonic_non_decreasing": bool(t.is_monotonic_increasing),
        "n_duplicate_timestamps": int(t.duplicated().sum()),
        "max_gap_hours": float(gaps.max()), "median_gap_hours": float(np.median(gaps)),
        "missing_overall_frac": float(np.isnan(X).mean()),
        "rows_with_any_missing": int(np.isnan(X).any(axis=1).sum()),
        "sensors_missing_gt_50pct": int(np.sum(miss > 0.5)),
        "sensors_missing_gt_20pct": int(np.sum(miss > 0.2)),
        "sensors_missing_none": int(np.sum(miss == 0)),
        "sensors_constant": const,
        "sensors_near_constant_95pct_same_value": int(np.sum(share >= secom.NEAR_CONSTANT_SHARE)),
        "missing_frac_hist": {
            "0": int(np.sum(miss == 0)),
            "(0,0.05]": int(np.sum((miss > 0) & (miss <= 0.05))),
            "(0.05,0.2]": int(np.sum((miss > 0.05) & (miss <= 0.2))),
            "(0.2,0.5]": int(np.sum((miss > 0.2) & (miss <= 0.5))),
            "(0.5,1]": int(np.sum(miss > 0.5)),
        },
        "by_month": by_month,
        "split_fractions": {"fit_end": secom.FIT_END, "cal_end": secom.CAL_END},
        "split": {k: {"n": int(v.size), "fails": int(y[v].sum()),
                      "start": str(t[v[0]]), "end": str(t[v[-1]])} for k, v in sp.items()},
    }


# --------------------------------------------------------------------------- shared prep
def prepare(d: secom.Secom) -> dict:
    sp = secom.split_indices(len(d.y))
    fit, cal, test = sp["fit"], sp["cal"], sp["test"]
    keep = secom.select_sensors(d.X[fit])
    ref = fit[d.y[fit] == 0]                       # in-control reference: pass rows, fit period
    stdz = secom.Standardizer.fit(d.X[ref], keep)
    return {"split": sp, "keep": keep, "ref": ref, "stdz": stdz}


def smd_rank(d: secom.Secom, rows: np.ndarray, keep: np.ndarray) -> list[tuple[int, float]]:
    """Rank kept sensors by |standardized mean difference| fail vs pass on ``rows``."""
    out = []
    yf = d.y[rows]
    for j in np.where(keep)[0]:
        col = d.X[rows, j]
        a = col[(yf == 1) & ~np.isnan(col)]
        b = col[(yf == 0) & ~np.isnan(col)]
        if a.size < 3 or b.size < 3:
            continue
        s = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
        if s == 0:
            continue
        out.append((int(j), float((a.mean() - b.mean()) / s)))
    out.sort(key=lambda z: (-abs(z[1]), z[0]))
    return out


# --------------------------------------------------------------------------- SPC
def spc_sensor(x_all: np.ndarray, y_all: np.ndarray, phase1_rows: np.ndarray, phase2_rows: np.ndarray) -> dict:
    x1 = x_all[phase1_rows]
    ok1 = ~np.isnan(x1)
    x1 = x1[ok1]
    lim, retained = spc.phase1_individuals(x1, max_iter=PHASE1_PASSES)
    center, sigma = lim["center"], lim["sigma"]

    x2_full = x_all[phase2_rows]
    y2_full = y_all[phase2_rows]
    ok2 = ~np.isnan(x2_full)
    x2 = x2_full[ok2]
    y2 = y2_full[ok2]

    rules1 = spc.weco_rules(x1[retained], center, sigma)
    rules2 = spc.weco_rules(x2, center, sigma)
    any2 = np.zeros(x2.size, dtype=bool)
    for v in rules2.values():
        any2 |= v

    ew = spc.ewma(x2, EWMA_LAMBDA, center)
    ewl = spc.ewma_limits(center, sigma, EWMA_LAMBDA)
    ew_sig = (ew > ewl["ucl"]) | (ew < ewl["lcl"])
    cs = spc.tabular_cusum(x2, center, CUSUM_K * sigma, CUSUM_H * sigma)

    m1 = (x1[retained].size // SUBGROUP_N) * SUBGROUP_N
    xr = spc.xbar_r_limits(x1[retained][:m1].reshape(-1, SUBGROUP_N))
    m2 = (x2.size // SUBGROUP_N) * SUBGROUP_N
    sg2 = x2[:m2].reshape(-1, SUBGROUP_N)
    xb2 = sg2.mean(axis=1)
    r2 = sg2.max(axis=1) - sg2.min(axis=1)

    lsl, usl = center - SPEC_K * sigma, center + SPEC_K * sigma
    cap1 = spc.capability(x1[retained], lsl, usl)
    cap2 = spc.capability(x2, lsl, usl)

    r1_alarm = rules2["rule1"]
    r1_high = x2 > center + 3.0 * sigma
    r1_low = x2 < center - 3.0 * sigma

    def first(mask: np.ndarray) -> int | None:
        idx = np.where(mask)[0]
        return int(idx[0]) if idx.size else None

    def fail_rate(mask: np.ndarray) -> float | None:
        return float(y2[mask].mean()) if mask.any() else None

    table = [[int((r1_alarm & (y2 == 1)).sum()), int((r1_alarm & (y2 == 0)).sum())],
             [int((~r1_alarm & (y2 == 1)).sum()), int((~r1_alarm & (y2 == 0)).sum())]]
    fisher_p = float(stats.fisher_exact(table, alternative="two-sided")[1])

    return {
        "phase1": {"n_used": int(x1.size), "n_missing": int((~ok1).sum()),
                   "n_excluded_beyond_3sigma": int((~retained).sum()),
                   "limits": lim,
                   "sd_overall_retained": float(x1[retained].std(ddof=1)),
                   "ratio_sd_to_mr_sigma": float(x1[retained].std(ddof=1) / sigma),
                   "rule_rates": {k: float(v.mean()) for k, v in rules1.items()},
                   "xbar_r": xr, "capability": cap1},
        "phase2": {"n_used": int(x2.size), "n_missing": int((~ok2).sum()),
                   "rule_counts": {k: int(v.sum()) for k, v in rules2.items()},
                   "rule_rates": {k: float(v.mean()) for k, v in rules2.items()},
                   "any_rule_rate": float(any2.mean()),
                   "ewma_signals": int(ew_sig.sum()), "ewma_first_signal": first(ew_sig),
                   "cusum_signals": int(cs["signal"].sum()), "cusum_first_signal": first(cs["signal"]),
                   "xbar_beyond_limits": int(np.sum((xb2 > xr["xbar_ucl"]) | (xb2 < xr["xbar_lcl"]))),
                   "r_beyond_limits": int(np.sum((r2 > xr["r_ucl"]) | (r2 < xr["r_lcl"]))),
                   "n_subgroups": int(sg2.shape[0]),
                   "capability": cap2,
                   "fail_rate_rule1_alarm": fail_rate(r1_alarm),
                   "fail_rate_no_rule1_alarm": fail_rate(~r1_alarm),
                   "n_rule1_alarm": int(r1_alarm.sum()),
                   "n_rule1_high": int(r1_high.sum()), "n_rule1_low": int(r1_low.sum()),
                   "fails_rule1_high": int(y2[r1_high].sum()), "fails_rule1_low": int(y2[r1_low].sum()),
                   "rule1_vs_fail_table": table,
                   "rule1_vs_fail_fisher_p": fisher_p},
        "spec_assumption": {"lsl": lsl, "usl": usl, "k_sigma": SPEC_K},
        "series": {"phase1_x": x1, "phase1_retained": retained, "phase2_x": x2, "phase2_y": y2,
                   "ewma": ew, "ewma_limits": ewl, "cusum_hi": cs["s_hi"], "cusum_lo": cs["s_lo"]},
    }


def run_spc(d: secom.Secom, prep: dict) -> dict:
    sp = prep["split"]
    phase1 = np.concatenate([sp["fit"], sp["cal"]])
    phase2 = sp["test"]
    ranked = smd_rank(d, sp["fit"], prep["keep"])
    chosen = ranked[:N_SPC_SENSORS]
    out = {"selection": "top |SMD| fail-vs-pass on the fit period among kept sensors",
           "phase1_rows": int(phase1.size), "phase2_rows": int(phase2.size),
           "params": {"ewma_lambda": EWMA_LAMBDA, "ewma_k": 3.0, "cusum_k_sigma": CUSUM_K,
                      "cusum_h_sigma": CUSUM_H, "subgroup_n": SUBGROUP_N, "spec_k_sigma": SPEC_K},
           "sensors": {}}
    for j, smd in chosen:
        res = spc_sensor(d.X[:, j], d.y, phase1, phase2)
        res["smd_fit"] = smd
        out["sensors"][d.sensors[j]] = res
    out["fisher_scan"] = fisher_scan(d, prep["keep"], phase1, phase2)
    return out


def rule1_fisher(x_all: np.ndarray, y_all: np.ndarray, phase1_rows: np.ndarray,
                 phase2_rows: np.ndarray) -> float | None:
    """Fisher p for 'Phase II rule-1 alarm' vs fail, with the same Phase I limits as
    ``spc_sensor`` (None if the Phase I sigma is zero)."""
    x1 = x_all[phase1_rows]
    x1 = x1[~np.isnan(x1)]
    lim, _ = spc.phase1_individuals(x1, max_iter=PHASE1_PASSES)
    if not lim["sigma"] > 0:
        return None
    x2 = x_all[phase2_rows]
    ok = ~np.isnan(x2)
    x2, y2 = x2[ok], y_all[phase2_rows][ok]
    a = np.abs(x2 - lim["center"]) > 3.0 * lim["sigma"]
    table = [[int((a & (y2 == 1)).sum()), int((a & (y2 == 0)).sum())],
             [int((~a & (y2 == 1)).sum()), int((~a & (y2 == 0)).sum())]]
    return float(stats.fisher_exact(table, alternative="two-sided")[1])


def fisher_scan(d: secom.Secom, keep: np.ndarray, phase1: np.ndarray, phase2: np.ndarray,
                alpha: float = 0.05) -> dict:
    """The rule-1 / fail association test of the SPC table repeated on every kept
    sensor, to see how many p < alpha one gets when nothing is selected."""
    ps = []
    for j in np.where(keep)[0]:
        p = rule1_fisher(d.X[:, j], d.y, phase1, phase2)
        if p is not None:
            ps.append((d.sensors[j], p))
    pv = np.array([p for _, p in ps])
    order = np.argsort(pv, kind="mergesort")
    bonf = alpha / len(ps)
    return {"n_tested": len(ps), "alpha": alpha, "n_below_alpha": int(np.sum(pv < alpha)),
            "expected_below_alpha_if_independent_and_exact": alpha * len(ps),
            "bonferroni_threshold": bonf, "n_below_bonferroni": int(np.sum(pv < bonf)),
            "smallest": [{"sensor": ps[i][0], "p": float(pv[i])} for i in order[:5]]}


# --------------------------------------------------------------------------- FDC
def _sigma_count_score(Z: np.ndarray) -> np.ndarray:
    """Univariate baseline: number of sensors with |z| > 3 (z from the reference rows)."""
    return np.sum(np.abs(Z) > 3.0, axis=1).astype(float)


def _perm(s_test: np.ndarray, y_test: np.ndarray, seed: int = PERM_SEED) -> dict:
    return metrics.permutation_test(s_test, y_test, N_PERM, seed)


def _evaluate(name: str, s_cal: np.ndarray, y_cal: np.ndarray, s_test: np.ndarray, y_test: np.ndarray) -> dict:
    thr = metrics.threshold_at_far(s_cal[y_cal == 0], FAR)
    ar = metrics.alarm_rates(s_test, y_test, thr)
    rec = lambda s, yy: metrics.alarm_rates(s, yy, thr)["recall"]
    far_ = lambda s, yy: metrics.alarm_rates(s, yy, thr)["far"]
    perm = _perm(s_test, y_test)
    return {
        "method": name,
        "threshold": thr,
        "cal_far_at_cal_threshold": float(np.mean(s_cal[y_cal == 0] > thr)),
        "recall_at_cal_threshold": metrics.stratified_bootstrap(s_test, y_test, rec, N_BOOT, BOOT_SEED),
        "test_far_at_cal_threshold": metrics.stratified_bootstrap(s_test, y_test, far_, N_BOOT, BOOT_SEED),
        "tp": ar["tp"], "fp": ar["fp"],
        "pr_auc": metrics.stratified_bootstrap(s_test, y_test, metrics.average_precision, N_BOOT, BOOT_SEED),
        "roc_auc": metrics.stratified_bootstrap(s_test, y_test, metrics.roc_auc, N_BOOT, BOOT_SEED),
        "perm_pr_auc": perm["pr_auc"], "perm_roc_auc": perm["roc_auc"],
        "recall_at_test_far_oracle": metrics.recall_at_far_oracle(s_test, y_test, FAR),
    }


def fit_lr(Za: np.ndarray, y: np.ndarray, fit: np.ndarray, cal: np.ndarray) -> tuple[float, np.ndarray, list]:
    """L2 logistic regression on fit rows; C chosen by calibration-period PR-AUC.
    Returns (C, scores on all rows, selection table)."""
    choice, best = [], None
    for c in LR_C_GRID:
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
        lr.fit(Za[fit], y[fit])
        ap = metrics.average_precision(lr.decision_function(Za[cal]), y[cal])
        choice.append({"C": c, "cal_pr_auc": ap})
        if best is None or ap > best[0]:
            best = (ap, c, lr)
    return best[1], best[2].decision_function(Za), choice


def fit_gb(X: np.ndarray, y: np.ndarray, fit: np.ndarray, cal: np.ndarray,
           keep: np.ndarray) -> tuple[dict, np.ndarray, list]:
    """HistGradientBoosting on fit rows (raw kept sensors, NaN handled natively);
    learning rate, leaf count and number of iterations chosen by calibration-period
    PR-AUC. Returns (chosen config, probabilities on all rows, selection table)."""
    from threadpoolctl import threadpool_limits
    choice, best = [], None
    Xk = X[:, keep]
    for rate, leaves in GB_FITS:
        m = HistGradientBoostingClassifier(max_iter=max(GB_ITERS), learning_rate=rate, max_leaf_nodes=leaves,
                                           **GB_FIXED)
        # single OpenMP thread: same result, and no spin-wait slow-down on a busy machine
        with threadpool_limits(limits=GB_THREADS, user_api="openmp"):
            m.fit(Xk[fit], y[fit])
            stages = list(m.staged_predict_proba(Xk))
        for it, P in enumerate(stages, start=1):
            if it not in GB_ITERS:
                continue
            s = P[:, 1]
            ap = metrics.average_precision(s[cal], y[cal])
            cfg = {"learning_rate": rate, "max_leaf_nodes": leaves, "max_iter": it}
            choice.append({**cfg, "cal_pr_auc": ap})
            if best is None or ap > best[0]:
                best = (ap, cfg, s.copy())
    return best[1], best[2], choice


def gb_scores(X: np.ndarray, y: np.ndarray, rows: np.ndarray, keep: np.ndarray, cfg: dict) -> np.ndarray:
    """Fit HistGradientBoosting with a given config on ``rows``; probabilities on all rows."""
    from threadpoolctl import threadpool_limits
    m = HistGradientBoostingClassifier(**cfg, **GB_FIXED)
    with threadpool_limits(limits=GB_THREADS, user_api="openmp"):
        m.fit(X[rows][:, keep], y[rows])
        return m.predict_proba(X[:, keep])[:, 1]


def _limit_comparison(mon: PCAMonitor, t2: np.ndarray, spe: np.ndarray, lt2: float, lspe: float,
                      ref: np.ndarray, cal_pass: np.ndarray, test_pass: np.ndarray) -> dict:
    """Empirical 99% limits (used for alarms) vs the distribution-based limits, and the
    bootstrap spread of the empirical limits (they rest on the top few calibration passes)."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, cal_pass.size, size=(N_BOOT, cal_pass.size))
    bt2 = np.quantile(t2[cal_pass][idx], LIMIT_Q, axis=1)
    bspe = np.quantile(spe[cal_pass][idx], LIMIT_Q, axis=1)
    out = {"n_cal_pass": int(cal_pass.size), "n_ref": int(mon.n_ref)}
    for stat, v, emp, theo, bs, nm in (("t2", t2, lt2, t2_limit_phase2(mon.n_components, mon.n_ref, LIMIT_Q), bt2,
                                        "Phase II F limit (NIST 6.5.4.3.4)"),
                                       ("spe", spe, lspe, spe_limit_jm(mon.resid_eig, LIMIT_Q), bspe,
                                        "Jackson-Mudholkar")):
        out[stat] = {"empirical": emp, "empirical_boot_lo": float(np.quantile(bs, 0.025)),
                     "empirical_boot_hi": float(np.quantile(bs, 0.975)),
                     "theoretical": theo, "theoretical_name": nm}
        for lab, lim in (("empirical", emp), ("theoretical", theo)):
            out[stat][f"exceed_{lab}"] = {"ref_pass": float(np.mean(v[ref] > lim)),
                                          "cal_pass": float(np.mean(v[cal_pass] > lim)),
                                          "test_pass": float(np.mean(v[test_pass] > lim))}
    return out


def run_fdc(d: secom.Secom, prep: dict) -> dict:
    sp, stdz, ref, keep = prep["split"], prep["stdz"], prep["ref"], prep["keep"]
    fit, cal, test = sp["fit"], sp["cal"], sp["test"]
    Z = stdz.transform(d.X)
    y = d.y
    cal_pass = cal[y[cal] == 0]
    test_pass = test[y[test] == 0]

    scores_cal: dict[str, np.ndarray] = {}
    scores_test: dict[str, np.ndarray] = {}
    pca_info = {}
    monitors = {}
    for vt in VAR_TARGETS:
        mon = PCAMonitor.fit(Z[ref], var_target=vt)
        monitors[vt] = mon
        t2, spe = mon.t2(Z), mon.spe(Z)
        lt2 = empirical_limit(t2[cal_pass], LIMIT_Q)
        lspe = empirical_limit(spe[cal_pass], LIMIT_Q)
        comb = np.maximum(t2 / lt2, spe / lspe)
        tag = f"{int(vt * 100)}"
        pca_info[tag] = {"n_components": mon.n_components, "explained": mon.explained,
                         "t2_ucl99": lt2, "spe_ucl99": lspe,
                         "limits": _limit_comparison(mon, t2, spe, lt2, lspe, ref, cal_pass, test_pass)}
        for nm, s in ((f"PCA T2 ({tag}% var)", t2), (f"PCA SPE ({tag}% var)", spe),
                      (f"PCA T2/SPE combined ({tag}% var)", comb)):
            scores_cal[nm], scores_test[nm] = s[cal], s[test]

    # univariate baselines
    sc = _sigma_count_score(Z)
    scores_cal["Count of sensors beyond 3 sigma"], scores_test["Count of sensors beyond 3 sigma"] = sc[cal], sc[test]
    ranked = smd_rank(d, fit, keep)
    jbest, smd = ranked[0]
    col = d.X[:, jbest]
    med = np.nanmedian(col[fit])
    sb = np.sign(smd) * np.where(np.isnan(col), med, col)
    scores_cal[f"Best single sensor ({d.sensors[jbest]})"] = sb[cal]
    scores_test[f"Best single sensor ({d.sensors[jbest]})"] = sb[test]

    # supervised: logistic regression on all fit rows (scaler fit on fit rows), C chosen on cal
    stdz_all = secom.Standardizer.fit(d.X[fit], keep)
    Za = stdz_all.transform(d.X)
    c_best, s_lr, lr_choice = fit_lr(Za, y, fit, cal)
    nm = f"Logistic regression (L2, C={c_best:g})"
    scores_cal[nm], scores_test[nm] = s_lr[cal], s_lr[test]

    gb_cfg, s_h, gb_choice = fit_gb(d.X, y, fit, cal, keep)
    nm = "Gradient boosting (HistGB, lr={learning_rate:g}, leaves={max_leaf_nodes}, iter={max_iter})".format(**gb_cfg)
    scores_cal[nm], scores_test[nm] = s_h[cal], s_h[test]

    y_cal, y_test = y[cal], y[test]
    results = [_evaluate(k, scores_cal[k], y_cal, scores_test[k], y_test) for k in scores_cal]
    # no-skill reference for PR-AUC: a random ranking of the test rows (no ties)
    rnd = metrics.permutation_test(np.arange(test.size, dtype=float), y_test, N_PERM, PERM_SEED)
    return {
        "far_target": FAR, "limit_quantile": LIMIT_Q, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "n_perm": N_PERM, "perm_seed": PERM_SEED,
        "n_sensors_kept": int(keep.sum()), "n_reference_rows": int(ref.size),
        "test": {"n": int(test.size), "fails": int(y_test.sum()), "prevalence": float(y_test.mean())},
        "cal": {"n": int(cal.size), "fails": int(y_cal.sum())},
        "random_ranking_pr_auc": {"expected_exact": metrics.random_ap_expected(test.size, int(y_test.sum())),
                                  "simulated_mean": rnd["pr_auc"]["null_mean"],
                                  "simulated_p95": rnd["pr_auc"]["null_p95"],
                                  "source": "Bestgen (2015), PBML 103; simulated with the permutation seed"},
        "pca": pca_info,
        "lr_c_grid": list(LR_C_GRID),
        "lr_c_selection": lr_choice,
        "gb_selection": gb_choice, "gb_chosen": gb_cfg, "gb_fixed": GB_FIXED,
        "methods": results,
        "_monitor80": monitors[0.8],
        "_Z": Z,
    }


# --------------------------------------------------------------------------- rolling origin
def rolling_fold(d: secom.Secom, sp: dict[str, np.ndarray], perm_seed: int) -> dict:
    """One rolling-origin fold: everything (sensor filter, scaling, PCA, model
    coefficients, hyperparameters) is refitted on this fold's fit/cal rows; the
    test rows are only scored."""
    fit, cal, test = sp["fit"], sp["cal"], sp["test"]
    y = d.y
    keep = secom.select_sensors(d.X[fit])
    ref = fit[y[fit] == 0]
    Z = secom.Standardizer.fit(d.X[ref], keep).transform(d.X)
    mon = PCAMonitor.fit(Z[ref], var_target=0.8)
    scores = {"PCA T2 (80% var)": mon.t2(Z), "PCA SPE (80% var)": mon.spe(Z),
              "Count of sensors beyond 3 sigma": _sigma_count_score(Z)}
    Za = secom.Standardizer.fit(d.X[fit], keep).transform(d.X)
    c_best, scores["Logistic regression (L2, C chosen on cal)"], lr_choice = fit_lr(Za, y, fit, cal)
    gb_cfg, scores["Gradient boosting (HistGB, chosen on cal)"], gb_choice = fit_gb(d.X, y, fit, cal, keep)
    # same hyperparameters, refitted on fit + cal (more and more recent training rows; test still unseen)
    fc = np.concatenate([fit, cal])
    Zfc = secom.Standardizer.fit(d.X[fc], keep).transform(d.X)
    lr = LogisticRegression(C=c_best, class_weight="balanced", max_iter=5000).fit(Zfc[fc], y[fc])
    scores["Logistic regression (refit on fit + cal)"] = lr.decision_function(Zfc)
    scores["Gradient boosting (refit on fit + cal)"] = gb_scores(d.X, y, fc, keep, gb_cfg)
    yt = y[test]
    methods, nulls = [], {}
    for name, s in scores.items():
        perm = metrics.permutation_test(s[test], yt, N_PERM, perm_seed)
        nulls[name] = perm["_null"]
        methods.append({"method": name, "pr_auc": perm["pr_auc"]["observed"], "roc_auc": perm["roc_auc"]["observed"],
                        "perm_pr_auc": perm["pr_auc"], "perm_roc_auc": perm["roc_auc"]})
    blocks = {k: {"n": int(v.size), "fails": int(y[v].sum()), "start": str(d.t[v[0]]), "end": str(d.t[v[-1]])}
              for k, v in sp.items()}
    return {"blocks": blocks, "n_sensors_kept": int(keep.sum()), "pca_components": mon.n_components,
            "lr_c": c_best, "lr_c_selection": lr_choice, "gb_chosen": gb_cfg, "gb_selection": gb_choice,
            "random_ranking_pr_auc": metrics.random_ap_expected(test.size, int(yt.sum())),
            "methods": methods, "_nulls": nulls, "_scores": scores}


def run_rolling(d: secom.Secom) -> dict:
    n = len(d.y)
    folds = []
    for i, (a, b, c) in enumerate(ROLLING_FOLDS):
        sp = secom.split_indices(n, a, b, c)
        f = rolling_fold(d, sp, PERM_SEED + i)
        f["fractions"] = {"fit_end": a, "cal_end": b, "test_end": c}
        folds.append(f)
    names = [m["method"] for m in folds[0]["methods"]]
    pooled = []
    for k, name in enumerate(names):
        row = {"method": name}
        for metric in ("pr_auc", "roc_auc"):
            obs = float(np.mean([f["methods"][k][metric] for f in folds]))
            # within-fold permutations are independent across folds -> row-wise mean is a draw
            # from the null of the mean over folds
            null = np.mean([f["_nulls"][name][metric] for f in folds], axis=0)
            row[metric] = {"mean_over_folds": obs, "p": float((1 + np.sum(null >= obs)) / (1 + null.size)),
                           "null_mean": float(null.mean())}
        pooled.append(row)
    return {"folds": folds, "pooled": pooled, "n_perm": N_PERM, "perm_seed_base": PERM_SEED,
            "fold_fractions": [list(f) for f in ROLLING_FOLDS]}


# --------------------------------------------------------------------------- excursion
def repeated_share_baseline(x: np.ndarray, w: int) -> float | None:
    """Mean over every window of ``w`` consecutive rows of the share of non-missing
    values that occur more than once inside that window."""
    shares = []
    for s in range(0, x.size - w + 1):
        v = x[s:s + w]
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        _, cnt = np.unique(v, return_counts=True)
        shares.append(cnt[cnt > 1].sum() / v.size)
    return float(np.mean(shares)) if shares else None


def find_excursion(d: secom.Secom, prep: dict, fdc: dict) -> dict:
    """Densest window of PCA (80% var) alarms in the test period + contributions."""
    sp = prep["split"]
    test = sp["test"]
    mon: PCAMonitor = fdc["_monitor80"]
    Z = fdc["_Z"]
    lt2, lspe = fdc["pca"]["80"]["t2_ucl99"], fdc["pca"]["80"]["spe_ucl99"]
    t2, spe = mon.t2(Z[test]), mon.spe(Z[test])
    alarm = (t2 > lt2) | (spe > lspe)
    w = EXCURSION_WINDOW
    counts = np.convolve(alarm.astype(int), np.ones(w, dtype=int), mode="valid")
    start = int(np.argmax(counts))          # earliest densest window
    win = np.arange(start, start + w)
    rows = test[win]
    al_rows = rows[alarm[win]]
    kept_names = [n for n, k in zip(d.sensors, prep["keep"]) if k]
    spe_c = mon.spe_contrib(Z[al_rows])
    t2_c = mon.t2_contrib(Z[al_rows])
    spe_share = (spe_c / spe_c.sum(axis=1, keepdims=True)).mean(axis=0)
    top_spe = np.argsort(-spe_share, kind="mergesort")[:5]
    t2_mean = t2_c.mean(axis=0)
    top_t2 = np.argsort(-t2_mean, kind="mergesort")[:5]
    ref = prep["ref"]
    zwin = Z[rows]

    def sensor_rows(idx):
        out = []
        for j in idx:
            out.append({"sensor": kept_names[j],
                        "mean_spe_share": float(spe_share[j]),
                        "mean_t2_contrib": float(t2_mean[j]),
                        "window_mean_z": float(zwin[:, j].mean()),
                        "window_missing_frac": float(np.isnan(d.X[rows][:, prep["keep"]][:, j]).mean())})
        return out

    # data-signature checks on the contributing sensors (raw values, reference = fit-period passes)
    sig_idx = list(dict.fromkeys(list(top_spe) + list(top_t2)))
    kept_cols = np.where(prep["keep"])[0]
    fit = sp["fit"]
    block_of = np.full(len(d.y), "", dtype=object)
    for k_, v_ in sp.items():
        block_of[v_] = k_
    signatures = []
    for j in sig_idx:
        col = d.X[:, kept_cols[j]]
        r = col[ref]
        r = r[~np.isnan(r)]
        q01, q50, q99 = np.quantile(r, [0.01, 0.5, 0.99])
        span = q99 - q01
        w_raw = col[rows]
        ok = ~np.isnan(w_raw)
        vals = w_raw[ok]
        uniq, cnt = np.unique(vals, return_counts=True)
        far_all = ~np.isnan(col) & (np.abs(col - q50) > 10.0 * span)
        zero_all = (col == 0.0) if np.mean(r == 0.0) == 0.0 else np.zeros(col.size, dtype=bool)

        def where_rows(mask):
            idx = np.where(mask)[0]
            return {"n": int(idx.size), "fails": int(d.y[idx].sum()),
                    "by_block": {k_: int(np.sum(block_of[idx] == k_)) for k_ in ("fit", "cal", "test")},
                    "times": [str(d.t[i]) for i in idx]}

        signatures.append({
            "sensor": kept_names[j],
            "ref_q01": float(q01), "ref_median": float(q50), "ref_q99": float(q99),
            "ref_zero_frac": float(np.mean(r == 0.0)),
            "window_missing": int((~ok).sum()),
            "window_exact_zero": int(np.sum(vals == 0.0)),
            "window_far_outside_ref": int(np.sum(np.abs(vals - q50) > 10.0 * span)),
            "window_max": float(vals.max()) if vals.size else None,
            "window_values_repeated": int(cnt[cnt > 1].sum()),
            "window_repeated_share": float(cnt[cnt > 1].sum() / vals.size) if vals.size else None,
            "fit_windows_repeated_share_mean": repeated_share_baseline(col[fit], w),
            "test_windows_repeated_share_mean": repeated_share_baseline(col[test], w),
            "window_pos_missing": [int(k) for k in np.where(~ok)[0]],
            "window_pos_exact_zero": [int(k) for k in np.where(w_raw == 0.0)[0]],
            "window_pos_far_outside_ref": [int(k) for k in np.where(ok & (np.abs(w_raw - q50) > 10.0 * span))[0]],
            "dataset_far_outside_ref": where_rows(far_all),
            "dataset_exact_zero_never_zero_in_ref": where_rows(zero_all),
            "window_values": [None if np.isnan(v) else float(v) for v in w_raw],
        })
    scan = metrics.window_max_null(alarm, w, N_SCAN, SCAN_SEED)
    sc = _sigma_count_score(Z)
    thr_sc = None
    for m in fdc["methods"]:
        if m["method"] == "Count of sensors beyond 3 sigma":
            thr_sc = m["threshold"]
    win_rows = []
    wc = mon.spe_contrib(Z[rows])
    for k, i in enumerate(win):
        r_ = test[i]
        jtop = int(np.argmax(wc[k]))
        win_rows.append({"pos": k, "time": str(d.t[r_]), "fail": int(d.y[r_]),
                       "t2_over_limit": float(t2[i] / lt2), "spe_over_limit": float(spe[i] / lspe),
                       "alarm": bool(alarm[i]), "n_sensors_beyond_3sigma": int(sc[r_]),
                       "n_missing_kept_sensors": int(np.isnan(d.X[r_, prep["keep"]]).sum()),
                       "top_spe_sensor": kept_names[jtop],
                       "top_spe_share": float(wc[k, jtop] / wc[k].sum())})

    base_alarm = float(alarm.mean())
    return {
        "unit_note": "rows are SECOM 'production entities' (secom.names); whether one row is one wafer is not stated",
        "signatures": signatures,
        "window_rows": win_rows,
        "scan_statistic": scan,
        "sigma_count_threshold": thr_sc,
        "window_size": w,
        "test_index_start": start,
        "time_start": str(d.t[rows[0]]), "time_end": str(d.t[rows[-1]]),
        "n_alarms_in_window": int(alarm[win].sum()),
        "n_t2_alarms": int((t2[win] > lt2).sum()), "n_spe_alarms": int((spe[win] > lspe).sum()),
        "test_alarm_rate": base_alarm,
        "expected_alarms_at_test_rate": base_alarm * w,
        "fails_in_window": int(d.y[rows].sum()),
        "fail_rate_window": float(d.y[rows].mean()),
        "fail_rate_test": float(d.y[test].mean()),
        "fail_rate_alarmed_rows_test": float(d.y[test][alarm].mean()) if alarm.any() else None,
        "fail_rate_non_alarmed_rows_test": float(d.y[test][~alarm].mean()),
        "n_alarmed_test": int(alarm.sum()),
        "top_spe_sensors": sensor_rows(top_spe),
        "top_t2_sensors": sensor_rows(top_t2),
        "n_reference_rows": int(ref.size),
        "_series": {"t2": t2, "spe": spe, "alarm": alarm, "y": d.y[test], "win": win,
                    "spe_share": spe_share, "t2_mean": t2_mean, "kept_names": kept_names},
    }


def run_all(d: secom.Secom) -> dict:
    prep = prepare(d)
    prof = profile(d)
    sp = run_spc(d, prep)
    fd = run_fdc(d, prep)
    ex = find_excursion(d, prep, fd)
    ro = run_rolling(d)
    return {"profile": prof, "spc": sp, "fdc": fd, "excursion": ex, "rolling": ro, "prep": prep}
