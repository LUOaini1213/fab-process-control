"""Markdown report generator.

Builds every results table from ``results/*.json`` and
* writes ``results/summary.md``;
* rewrites the README blocks delimited by ``<!-- BEGIN:name -->`` / ``<!-- END:name -->``.
``--check`` only compares and returns 1 when the README is stale, so a test can
enforce that the README numbers come from the results files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .fmea import to_markdown as fmea_markdown

RESULTS = Path("results")
README = Path("README.md")


def _load(name: str, results: Path) -> dict | None:
    p = results / name
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def f(x, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def pct(x, nd: int = 1) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{nd}f}%"


def ci(d: dict, nd: int = 3) -> str:
    return f"{f(d['point'], nd)} [{f(d['lo'], nd)}, {f(d['hi'], nd)}]"


def block_data(p: dict) -> str:
    s = p["split"]
    rows = [
        ("Rows (production entities)", f(p["n_rows"])),
        ("Sensor columns in `secom.data`", f"{p['n_sensors_in_file']} (UCI page and `secom.names` say {p['n_features_listed_by_uci']})"),
        ("Fails / passes", f"{p['n_fail']} / {p['n_pass']} (fail rate {pct(p['fail_rate'])})"),
        ("Time span", f"{p['time_first']} to {p['time_last']}"),
        ("Rows time-ordered / duplicate timestamps", f"{p['time_monotonic_non_decreasing']} / {p['n_duplicate_timestamps']}"),
        ("Missing cells overall", pct(p["missing_overall_frac"], 2)),
        ("Rows with at least one missing value", f(p["rows_with_any_missing"])),
        ("Sensors with no missing value", f(p["sensors_missing_none"])),
        ("Sensors > 50% missing", f(p["sensors_missing_gt_50pct"])),
        ("Constant sensors", f(p["sensors_constant"])),
        ("Near-constant sensors (one value >= 95% of rows)", f(p["sensors_near_constant_95pct_same_value"])),
    ]
    out = "| Item | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows) + "\n\n"
    out += p["note_feature_count"] + "\n\n"
    fr = p["split_fractions"]
    out += (f"Time-ordered split: fit = first {pct(fr['fit_end'], 0)} of rows, cal = up to {pct(fr['cal_end'], 0)}, "
            "test = the rest.\n\n")
    out += "| Block | Rows | Fails | From | To |\n|---|---|---|---|---|\n"
    for k in ("fit", "cal", "test"):
        out += f"| {k} | {s[k]['n']} | {s[k]['fails']} | {s[k]['start']} | {s[k]['end']} |\n"
    out += "\nFail rate by month: " + ", ".join(
        f"{m} {pct(v['fail_rate'])} ({v['fails']}/{v['n']})" for m, v in p["by_month"].items()) + "\n"
    return out


def block_spc(s: dict) -> str:
    out = ("| Sensor | SMD (fit) | Phase I n / excluded | SD / sigma_MR | Phase II: R1 / R2 / R3 / R4 | "
           "EWMA / CUSUM signals | X-bar / R subgroups out (n = 5) | Cpk I -> II | "
           "Fail rate with / without R1 alarm (Fisher p) | R1 alarms high / low (fails among them) |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    for name, r in s["sensors"].items():
        p1, p2 = r["phase1"], r["phase2"]
        rc = p2["rule_counts"]
        out += (f"| {name} | {f(r['smd_fit'], 2)} | {p1['n_used']} / {p1['n_excluded_beyond_3sigma']} | "
                f"{f(p1['ratio_sd_to_mr_sigma'], 2)} | {rc['rule1']} / {rc['rule2']} / {rc['rule3']} / {rc['rule4']} "
                f"(of {p2['n_used']}) | {p2['ewma_signals']} / {p2['cusum_signals']} | "
                f"{p2['xbar_beyond_limits']} / {p2['r_beyond_limits']} (of {p2['n_subgroups']}) | "
                f"{f(p1['capability']['cpk'], 2)} -> {f(p2['capability']['cpk'], 2)} | "
                f"{pct(p2['fail_rate_rule1_alarm'])} / {pct(p2['fail_rate_no_rule1_alarm'])} "
                f"(p = {f(p2['rule1_vs_fail_fisher_p'], 3)}) | "
                f"{p2['n_rule1_high']} / {p2['n_rule1_low']} ({p2['fails_rule1_high']} / {p2['fails_rule1_low']}) |\n")
    sig = [n for n, r in s["sensors"].items() if r["phase2"]["rule1_vs_fail_fisher_p"] < 0.05]
    out += ("\nSensors whose Phase II rule-1 alarms are associated with fails at p < 0.05 (Fisher exact, "
            f"no multiplicity correction, {len(s['sensors'])} sensors tested): {', '.join(sig) if sig else 'none'}.\n")
    fs = s["fisher_scan"]
    sm = ", ".join(f"{x['sensor']} {x['p']:.4f}" for x in fs["smallest"][:3])
    out += (f"\nSame test on all {fs['n_tested']} kept sensors: {fs['n_below_alpha']} reach p < {fs['alpha']} "
            f"(about {f(fs['expected_below_alpha_if_independent_and_exact'], 1)} expected by chance if the tests were "
            f"independent and exact); {fs['n_below_bonferroni']} pass the Bonferroni threshold "
            f"p < {fs['bonferroni_threshold']:.2e}. Smallest p: {sm}.\n")
    return out


def block_arl(r: dict) -> str:
    out = ("| h | Shift (sigma) | NIST table (one chart) | Simulated one-sided ARL | Simulated two-sided ARL |\n"
           "|---|---|---|---|---|\n")
    for row in r["rows"]:
        o, t = row["one_sided"], row["two_sided"]
        out += (f"| {f(row['h'], 0)} | {f(row['shift_sigma'], 1)} | {f(row['nist_one_sided'], 2)} | "
                f"{f(o['arl'], 1)} (SE {f(o['se'], 1)}) | {f(t['arl'], 1)} (SE {f(t['se'], 1)}) |\n")
    out += (f"\nk = {r['k']} sigma; {r['n_rep']} simulated runs per cell, N(shift, 1) data, signal when the CUSUM "
            "exceeds h. Two-sided = S_hi and S_lo both run, signal on either (what the SPC section does).\n")
    return out


def _p(x: float) -> str:
    return f"{x:.4f}" if x >= 0.001 else f"{x:.1e}"


def block_fdc(r: dict) -> str:
    t = r["test"]
    rr = r["random_ranking_pr_auc"]
    out = (f"Test period: {t['n']} rows, {t['fails']} fails (prevalence {f(t['prevalence'])}). "
           f"No-skill PR-AUC reference = a random ranking of these rows: expected {f(rr['expected_exact'])} "
           f"(exact, Bestgen 2015), 95th percentile {f(rr['simulated_p95'])} (simulated) - not the prevalence. "
           f"Alarm threshold per method = score quantile giving at most {pct(r['far_target'], 0)} false alarms on "
           f"calibration-period passes ({r['cal']['n']} rows). CIs: {r['n_boot']} stratified bootstrap resamples "
           f"of the test period. p: one-sided label-permutation test ({r['n_perm']} permutations, seed "
           f"{r['perm_seed']}), H0 = ranking unrelated to the label.\n\n")
    out += ("| Method | Recall at calibrated threshold | Cal / test FAR | PR-AUC | PR-AUC p | ROC-AUC | ROC-AUC p |\n"
            "|---|---|---|---|---|---|---|\n")
    for m in r["methods"]:
        out += (f"| {m['method']} | {ci(m['recall_at_cal_threshold'])} | "
                f"{f(m['cal_far_at_cal_threshold'])} / {f(m['test_far_at_cal_threshold']['point'])} | "
                f"{ci(m['pr_auc'])} | {_p(m['perm_pr_auc']['p'])} | {ci(m['roc_auc'])} | {_p(m['perm_roc_auc']['p'])} |\n")
    pca = r["pca"]
    out += "\nPCA models (fit on " + f"{r['n_reference_rows']} fit-period pass rows, {r['n_sensors_kept']} sensors): " + \
        "; ".join(f"{k}% target -> {v['n_components']} PCs ({pct(v['explained'])})" for k, v in pca.items()) + "\n"
    n_m = len(r["methods"])
    bonf = 0.05 / n_m
    sig_pr = [m["method"] for m in r["methods"] if m["perm_pr_auc"]["p"] < 0.05]
    sig_roc = [m["method"] for m in r["methods"] if m["perm_roc_auc"]["p"] < 0.05]
    bonf_roc = [m["method"] for m in r["methods"] if m["perm_roc_auc"]["p"] < bonf]
    bonf_pr = [m["method"] for m in r["methods"] if m["perm_pr_auc"]["p"] < bonf]
    out += (f"\nPermutation p < 0.05 for PR-AUC: {', '.join(sig_pr) if sig_pr else 'none'}. "
            f"For ROC-AUC: {', '.join(sig_roc) if sig_roc else 'none'}. "
            f"After Bonferroni over the {n_m} methods (p < {bonf:.4f}): PR-AUC {', '.join(bonf_pr) if bonf_pr else 'none'}; "
            f"ROC-AUC {', '.join(bonf_roc) if bonf_roc else 'none'}.\n")
    lrs = ", ".join(f"{x['C']:g}: {f(x['cal_pr_auc'])}" for x in r["lr_c_selection"])
    gbs = "; ".join(f"lr {x['learning_rate']:g} / {x['max_leaf_nodes']} leaves / {x['max_iter']} iter: "
                    f"{f(x['cal_pr_auc'])}" for x in r["gb_selection"])
    out += (f"\nCalibration-period PR-AUC used for model selection - logistic regression C: {lrs}. "
            f"Gradient boosting: {gbs}.\n")
    out += ("\n| PCA model | T2 limit: empirical 99% [bootstrap 95% range] / Phase II F | "
            "test-pass exceedance (empirical / F) | SPE limit: empirical 99% [bootstrap] / Jackson-Mudholkar | "
            "test-pass exceedance (empirical / JM) |\n|---|---|---|---|---|\n")
    for k, v in pca.items():
        L = v["limits"]
        a, b = L["t2"], L["spe"]
        out += (f"| {k}% var | {f(a['empirical'], 1)} [{f(a['empirical_boot_lo'], 1)}, {f(a['empirical_boot_hi'], 1)}] / "
                f"{f(a['theoretical'], 1)} | {f(a['exceed_empirical']['test_pass'])} / {f(a['exceed_theoretical']['test_pass'])} | "
                f"{f(b['empirical'], 1)} [{f(b['empirical_boot_lo'], 1)}, {f(b['empirical_boot_hi'], 1)}] / "
                f"{f(b['theoretical'], 1)} | {f(b['exceed_empirical']['test_pass'])} / {f(b['exceed_theoretical']['test_pass'])} |\n")
    first = next(iter(pca.values()))["limits"]
    out += (f"\nEmpirical limits are the 99th percentile of {first['n_cal_pass']} calibration-period passes, so each rests "
            f"on the top two or three of them; the bracket is the 2.5-97.5% range over {r['n_boot']} bootstrap resamples. "
            f"Distribution-based limits use the {first['n_ref']} reference rows and assume normality.\n")
    return out


def block_rolling(r: dict) -> str:
    out = ("| Fold | fit / cal / test rows | test fails | random-ranking PR-AUC | LR C (cal) | GB config (cal) |\n"
           "|---|---|---|---|---|---|\n")
    for i, fo in enumerate(r["folds"], start=1):
        b = fo["blocks"]
        g = fo["gb_chosen"]
        out += (f"| {i} | {b['fit']['n']} / {b['cal']['n']} / {b['test']['n']} ({b['test']['start'][:10]} to "
                f"{b['test']['end'][:10]}) | {b['test']['fails']} | {f(fo['random_ranking_pr_auc'])} | {fo['lr_c']:g} | "
                f"lr {g['learning_rate']:g}, {g['max_leaf_nodes']} leaves, {g['max_iter']} iter |\n")
    names = [m["method"] for m in r["folds"][0]["methods"]]
    out += "\n| Method | " + " | ".join(f"Fold {i} PR-AUC (p) / ROC-AUC (p)" for i in range(1, len(r["folds"]) + 1)) + \
        " | Mean PR-AUC (p) | Mean ROC-AUC (p) |\n|---|" + "---|" * (len(r["folds"]) + 2) + "\n"
    for k, name in enumerate(names):
        cells = []
        for fo in r["folds"]:
            m = fo["methods"][k]
            cells.append(f"{f(m['pr_auc'])} ({_p(m['perm_pr_auc']['p'])}) / {f(m['roc_auc'])} ({_p(m['perm_roc_auc']['p'])})")
        pl = r["pooled"][k]
        out += (f"| {name} | " + " | ".join(cells) + f" | {f(pl['pr_auc']['mean_over_folds'])} ({_p(pl['pr_auc']['p'])}) | "
                f"{f(pl['roc_auc']['mean_over_folds'])} ({_p(pl['roc_auc']['p'])}) |\n")
    out += (f"\np: one-sided label permutation within each fold ({r['n_perm']} permutations); for the means, the "
            "labels are permuted independently in every fold and the fold means recomputed.\n")
    return out


def block_excursion(e: dict) -> str:
    top = e["top_spe_sensors"]
    sc = e["scan_statistic"]
    out = (f"- Densest window of PCA alarms in the test period: {e['window_size']} consecutive rows, "
           f"{e['time_start']} to {e['time_end']}: {e['n_alarms_in_window']} alarms "
           f"(T2 {e['n_t2_alarms']}, SPE {e['n_spe_alarms']}) vs {f(e['expected_alarms_at_test_rate'], 2)} expected "
           f"at the test-period alarm rate ({pct(e['test_alarm_rate'])}).\n")
    out += (f"- Is that window unusual? The densest of the {sc['n_windows']} windows is the maximum of many counts, so it is "
            f"compared with the same maximum after placing the {sc['n_flags']} test-period alarms at random positions "
            f"({sc['n_sim']} simulations): mean maximum {f(sc['null_mean_max'], 2)}, 95th percentile "
            f"{f(sc['null_p95_max'], 0)}, P(maximum >= {sc['observed_max']}) = {_p(sc['p_value'])}.\n")
    out += (f"- Fails in the window: {e['fails_in_window']} ({pct(e['fail_rate_window'])}) vs test-period fail rate "
            f"{pct(e['fail_rate_test'])}. Across the whole test period, alarmed rows failed at "
            f"{pct(e['fail_rate_alarmed_rows_test'])} ({e['n_alarmed_test']} rows) vs "
            f"{pct(e['fail_rate_non_alarmed_rows_test'])} for non-alarmed rows.\n")
    out += "- Top SPE contributors (mean share of SPE over alarmed rows in the window): " + ", ".join(
        f"{t['sensor']} {pct(t['mean_spe_share'])} (window mean z = {f(t['window_mean_z'], 1)})" for t in top[:4]) + "\n"
    return out


def block_doe(r: dict) -> str:
    s1, s2, s3, rep = r["stage1_screening"], r["stage2_full_factorial"], r["stage3_ccd"], r["repeated_ccd"]
    names = r["factors"]
    out = "Screening, " + s1["design"] + ":\n\n| Term | Effect | p |\n|---|---|---|\n"
    for k in ("1", "2", "3", "4"):
        out += f"| x{k} {names[int(k) - 1]} | {f(s1['effects'][k])} | {f(s1['anova'][k]['p'], 4)} |\n"
    for k in ("12+34", "13+24", "14+23"):
        out += f"| {k} (aliased) | {f(s1['alias_pair_effects'][k])} | {f(s1['anova'][k]['p'], 4)} |\n"
    sig05 = [k for k in ("1", "2", "3", "4", "12+34", "13+24", "14+23") if s1["anova"][k]["p"] < 0.05]
    sp = r["screening_power"]
    out += (f"\nScreening terms with p < 0.05: {', '.join(sig05)}. "
            f"Kept for the next stage (p < {s1['screening_alpha']}): {', '.join(s1['active_factors'])}.\n\n")
    out += (f"How reliable is one screen? Over {sp['n_sim']} repeated screens with fresh noise, the share of screens "
            f"that keep each factor at p < {sp['screening_alpha']} (at p < 0.05): " + ", ".join(
                f"{k} {pct(v)} ({pct(sp['kept_rate_at_0.05'][k])})" for k, v in sp["kept_rate"].items()) +
            f". All three active factors kept and the inert one dropped: {pct(sp['all_three_active_kept_and_x4_dropped'])}.\n\n")
    out += (f"Full factorial + centre points: centre mean {f(s2['mean_centre'])} vs factorial mean "
            f"{f(s2['mean_factorial'])}; curvature term p = {s2['anova']['curvature']['p']:.2e}.\n\n")
    out += ("| CCD result | Value |\n|---|---|\n"
            f"| Quadratic fit R^2 / residual SD | {f(s3['r2'])} / {f(s3['resid_sd'])} (true noise SD {r['noise_sd']}) |\n"
            f"| Fitted stationary point (coded x1..x4) | {', '.join(f(v) for v in s3['stationary_point'])} ({s3['nature']}) |\n"
            f"| True optimum (coded) | {', '.join(f(v) for v in s3['true_optimum_x'])} |\n"
            f"| Distance fitted vs true (coded units) | {f(s3['distance_coded'])} |\n"
            f"| True response at fitted point vs true minimum | {f(s3['y_true_at_xstar'], 4)} vs {f(s3['true_optimum_y'], 4)} (regret {f(s3['regret'], 4)}) |\n"
            f"| Model prediction at fitted point / mean of 5 confirmation runs | {f(s3['y_pred_at_xstar'])} / {f(s3['confirmation_runs_mean'])} |\n"
            f"| Regret of simply taking the best observed CCD run | {f(s3['best_observed_run_regret'], 4)} |\n"
            f"| Over {rep['n_repeat']} repeated CCD experiments: median / 90th pct distance | {f(rep['distance_median'])} / {f(rep['distance_p90'])} |\n"
            f"| ... median / 90th pct regret (RSM vs best observed run) | {f(rep['regret_median'], 4)} / {f(rep['regret_p90'], 4)} vs {f(rep['best_run_regret_median'], 4)} / {f(rep['best_run_regret_p90'], 4)} |\n"
            f"| ... share of repeats where the best observed run is a centre point | {pct(rep['frac_best_run_is_centre_point'])} |\n")
    return out


def block_r2r(r: dict) -> str:
    c = r["comparison"]
    su = c["setup"]
    out = (f"Process: y = {su['alpha']} + {su['beta']} u + n*({su['delta']}) + N(0, {su['sigma']}^2); model gain b = {su['b_model']}, "
           f"target {su['target']}, assumed spec [{su['spec_lsl']}, {su['spec_usl']}]; {su['n_rep']} replications x "
           f"{su['n_runs']} runs, first {su['burn_in']} runs discarded. Mean over replications "
           "(2.5-97.5 percentile across replications in brackets).\n\n")
    out += ("Fixed, untuned settings (illustration):\n\n| Controller | lambda*beta/b | Mean offset (theory) | SD | RMSE | % in spec |\n"
            "|---|---|---|---|---|---|\n")
    for row in c["controllers"]:
        th = f" ({f(row['theory_bias'], 3)})" if "theory_bias" in row else ""
        xl = f(row["xi_lambda"], 3) if "xi_lambda" in row else "-"
        out += (f"| {row['controller']} | {xl} | {f(row['mean_offset'], 3)}{th} | {f(row['std'], 2)} | "
                f"{f(row['rmse'], 2)} [{f(row['rmse_p2.5'], 2)}, {f(row['rmse_p97.5'], 2)}] | {f(row['pct_in_spec'], 1)} |\n")
    gap = max(abs(z["mean_offset"] - z["theory_bias"]) for z in c["controllers"] if "theory_bias" in z)
    out += f"\nLargest |simulated mean offset - theory| over the EWMA rows: {f(gap, 3)} nm.\n"
    t = r["tuned"]
    out += (f"\nTuned comparison: each controller's weights chosen by mean RMSE on {t['n_select']} selection "
            f"replications (seeds {t['selection_seeds'][0]}-{t['selection_seeds'][1]}; EWMA {sum(g['kind'] == 'ewma' for g in t['grid'])} "
            f"lambda values, d-EWMA {sum(g['kind'] == 'dewma' for g in t['grid'])} (w1, w2) pairs), then evaluated on the "
            f"{t['n_eval']} independent replications above (seeds {t['evaluation_seeds'][0]}-{t['evaluation_seeds'][1]}).\n\n"
            "| Controller | Chosen setting | RMSE, selection reps | RMSE, evaluation reps | On grid edge |\n|---|---|---|---|---|\n")
    for kind in ("ewma", "dewma"):
        b = t["best"][kind]
        out += (f"| {'EWMA' if kind == 'ewma' else 'd-EWMA'} | {b['controller']} | {f(b['rmse_selection'], 2)} | "
                f"{f(b['rmse_evaluation'], 2)} [{f(b['rmse_evaluation_p2.5'], 2)}, {f(b['rmse_evaluation_p97.5'], 2)}] | "
                f"{'yes' if b['on_grid_edge'] else 'no'} |\n")
    d = t["paired_rmse_diff_dewma_minus_ewma"]
    out += (f"\nPaired difference (tuned d-EWMA minus tuned EWMA) on the evaluation replications: {f(d['mean'], 3)} nm "
            f"(95% CI {f(d['ci95_lo'], 3)} to {f(d['ci95_hi'], 3)}); d-EWMA lower in {pct(d['frac_reps_dewma_better'])} "
            "of replications.\n")
    st = r["stability"]
    out += (f"\nStability check: theory (stable iff 0 < lambda*beta/b < 2) and noise-free simulation agree at "
            f"{st['n_agree']} of {st['n_total']} grid points.\n")
    return out


def block_fmea(r: dict) -> str:
    rows = r["rows"]
    out = "| Failure mode | S x O x D = RPN | After action |\n|---|---|---|\n"
    for x in rows[:4]:
        out += (f"| {x['failure_mode']} | {x['s']} x {x['o']} x {x['d']} = {x['rpn']} | "
                f"{x['s_after']} x {x['o_after']} x {x['d_after']} = {x['rpn_after']} |\n")
    return out


BLOCKS = {
    "data": ("data_profile.json", block_data),
    "spc": ("spc_summary.json", block_spc),
    "arl": ("cusum_arl.json", block_arl),
    "fdc": ("fdc_results.json", block_fdc),
    "fdc_rolling": ("fdc_rolling.json", block_rolling),
    "excursion": ("excursion.json", block_excursion),
    "doe": ("doe_results.json", block_doe),
    "r2r": ("r2r_results.json", block_r2r),
    "fmea": ("fmea.json", block_fmea),
    "fmea_full": ("fmea.json", lambda r: fmea_markdown(r["rows"])),
}
DOCS_WITH_BLOCKS = (Path("docs") / "fmea.md",)


def build(results: Path = RESULTS) -> dict[str, str]:
    out = {}
    for name, (fn, func) in BLOCKS.items():
        data = _load(fn, results)
        if data is not None:
            out[name] = func(data)
    return out


def replace_blocks(text: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n)(.*?)(<!-- END:{name} -->)", re.S)
        text = pat.sub(lambda m: m.group(1) + body + m.group(3), text)
    return text


def main(check: bool = False, results: Path = RESULTS, readme: Path = README,
         docs: tuple[Path, ...] | None = None) -> int:
    from .common import write_text
    blocks = build(results)
    summary = "# Results summary (generated by `python -m fabpc.cli report`)\n\n" + "\n".join(
        f"## {k}\n\n{v}" for k, v in blocks.items() if k != "fmea_full")
    targets = [readme] + list(DOCS_WITH_BLOCKS if docs is None else docs)
    stale = []
    for path in targets:
        if not path.is_file():
            continue
        old = path.read_text(encoding="utf-8")
        new = replace_blocks(old, blocks)
        if new != old:
            stale.append(path)
            if not check:
                write_text(path, new)
    if check:
        if stale:
            print("stale generated blocks in: " + ", ".join(map(str, stale)) +
                  "; run `python -m fabpc.cli report`", file=sys.stderr)
            return 1
        return 0
    write_text(results / "summary.md", summary)
    return 0
