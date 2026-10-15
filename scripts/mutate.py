"""Scripted mutation testing.

Each mutant is one deliberate bug (a single exact-string replacement in one
source file). For every mutant the repository is copied to ``mutation_work/``
(git-ignored), the bug is applied there, and the test suite is run with
``pytest -x``. A mutant is *killed* when the suite fails, *survived* otherwise.

    python scripts/mutate.py                 # all mutants
    python scripts/mutate.py --only M07 M12  # selected mutants
    python scripts/mutate.py --check         # only verify that every pattern matches exactly once
    python scripts/mutate.py --jobs 3        # three mutants at a time, each in its own copy

Writes docs/mutation_report.md (and mutation_work/mutation_results.json).
SECOM-dependent tests run when SECOM_DIR points at the data; otherwise they are
skipped, which weakens the suite - the report records which mode was used.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "mutation_work"
COPY = ("src", "tests", "results", "docs", "README.md", "pyproject.toml")

# (id, file, old, new, description)
MUTANTS: list[tuple[str, str, str, str, str]] = [
    # --- spc.py
    ("M01", "src/fabpc/spc.py", "5: (0.577, 0.0, 2.115)", "5: (0.583, 0.0, 2.115)", "wrong A2 for n=5"),
    ("M02", "src/fabpc/spc.py", "D2_N2 = 1.128", "D2_N2 = 1.13", "wrong d2 for n=2"),
    ("M03", "src/fabpc/spc.py", '"ucl": center + 3.0 * sigma', '"ucl": center + 2.0 * sigma', "2-sigma individuals UCL"),
    ("M04", "src/fabpc/spc.py", "prev = lam * v + (1.0 - lam) * prev", "prev = lam * prev + (1.0 - lam) * v", "EWMA weights swapped"),
    ("M05", "src/fabpc/spc.py", "np.sqrt(lam / (2.0 - lam))", "np.sqrt(lam / (2.0 + lam))", "EWMA variance factor"),
    ("M06", "src/fabpc/spc.py", "sh = max(0.0, sh + v - mu0 - k)", "sh = max(0.0, sh + v - mu0 + k)", "CUSUM reference value sign"),
    ("M07", "src/fabpc/spc.py", "signal = (hi > h) | (lo > h)", "signal = (hi > h)", "CUSUM ignores the lower side"),
    ("M08", "src/fabpc/spc.py", "(_window_count(z > 2.0, 3) >= 2)", "(_window_count(z > 2.0, 3) >= 3)", "WECO rule 2 needs 3 of 3"),
    ("M09", "src/fabpc/spc.py", "(_window_count(z > 0.0, 8) == 8)", "(_window_count(z > 0.0, 7) == 7)", "WECO rule 4 run length 7"),
    ("M10", "src/fabpc/spc.py", 'out["rule1"] = np.abs(z) > 3.0', 'out["rule1"] = np.abs(z) >= 3.0', "rule 1 fires on the limit"),
    ("M11", "src/fabpc/spc.py", "(_window_count(up, 5) == 5)", "(_window_count(up, 4) == 4)", "trend rule with 5 points"),
    ("M12", "src/fabpc/spc.py", "cpk = min(", "cpk = max(", "Cpk uses the far side"),
    ("M13", "src/fabpc/spc.py", "s = float(x.std(ddof=1))", "s = float(x.std(ddof=0))", "capability with population SD"),
    ("M14", "src/fabpc/spc.py", "new_keep = keep & ~out", "new_keep = ~out", "Phase I re-admits excluded points"),
    # --- mspc.py
    ("M15", "src/fabpc/mspc.py", "return np.sum(T ** 2 / self.lam, axis=1)", "return np.sum(T ** 2 / np.sqrt(self.lam), axis=1)", "T2 scaled by SD not variance"),
    ("M16", "src/fabpc/mspc.py", "var = s ** 2 / (n - 1)", "var = s ** 2 / n", "PC variance with n"),
    ("M17", "src/fabpc/mspc.py", "a = int(np.searchsorted(ratio, var_target) + 1)", "a = int(np.searchsorted(ratio, var_target))", "one PC too few"),
    ("M18", "src/fabpc/mspc.py", "if P[j, k] < 0:", "if P[j, k] > 0:", "sign convention inverted"),
    ("M19", "src/fabpc/mspc.py", "w = (T / self.lam) @ self.P.T", "w = (T * self.lam) @ self.P.T", "T2 contribution weights"),
    ("M20", "src/fabpc/mspc.py", "return np.asarray(Z, dtype=float) - self.mu", "return np.asarray(Z, dtype=float)", "reference mean not removed"),
    # --- metrics.py
    ("M21", "src/fabpc/metrics.py", 'np.quantile(s, 1.0 - far, method="higher")', 'np.quantile(s, 1.0 - far, method="lower")', "threshold quantile rounding"),
    ("M22", "src/fabpc/metrics.py", "alarm = scores > thr", "alarm = scores >= thr", "alarm on equality"),
    ("M23", "src/fabpc/metrics.py", "last = np.r_[np.where(np.diff(s) != 0)[0], s.size - 1]", "last = np.arange(s.size)", "AP ignores ties"),
    ("M24", "src/fabpc/metrics.py", "ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0", "ranks[order[i:j + 1]] = i + 1.0", "ROC ties not averaged"),
    ("M25", "src/fabpc/metrics.py", "idx = np.concatenate([rng.choice(pos, pos.size, replace=True),", "idx = np.concatenate([rng.choice(neg, pos.size, replace=True),", "bootstrap loses the fails"),
    # --- secom.py / secom_analysis.py
    ("M26", "src/fabpc/secom.py", "i = int(round(fit_end * n))", "i = int(fit_end * n) + 1", "split boundary off by one"),
    ("M27", "src/fabpc/secom.py", "keep = miss <= max_missing", "keep = miss <= max_missing + 0.1", "missing filter loosened"),
    ("M28", "src/fabpc/secom.py", "med = np.nanmedian(Z, axis=0)", "med = np.nanmean(Z, axis=0)", "mean instead of median imputation"),
    ("M29", "src/fabpc/secom_analysis.py", "ref = fit[d.y[fit] == 0]", "ref = fit", "reference set includes fails"),
    ("M30", "src/fabpc/secom_analysis.py", "lt2 = empirical_limit(t2[cal_pass], LIMIT_Q)", "lt2 = empirical_limit(t2[test], LIMIT_Q)", "T2 limit leaks test data"),
    ("M31", "src/fabpc/secom_analysis.py", "stdz_all = secom.Standardizer.fit(d.X[fit], keep)", "stdz_all = secom.Standardizer.fit(d.X, keep)", "LR scaler fit on all rows (leak)"),
    # --- r2r.py / simulator.py
    ("M32", "src/fabpc/r2r.py", "u_next = (target - a) / b", "u_next = (target - a) / proc.beta", "EWMA uses the true gain"),
    ("M33", "src/fabpc/r2r.py", "a = ctrl.w1 * resid + (1.0 - ctrl.w1) * (a_prev + p)", "a = ctrl.w1 * resid + (1.0 - ctrl.w1) * a_prev", "d-EWMA reverts to PCC"),
    ("M34", "src/fabpc/r2r.py", "p = ctrl.w2 * (resid - a_prev) + (1.0 - ctrl.w2) * p", "p = ctrl.w2 * (resid - a) + (1.0 - ctrl.w2) * p", "d-EWMA drift uses new a"),
    ("M35", "src/fabpc/r2r.py", "return 0.0 < g < 2.0", "return 0.0 < g <= 2.0", "stability boundary inclusive"),
    ("M36", "src/fabpc/r2r.py", "return b * delta / (beta * w)", "return beta * delta / (b * w)", "drift bias formula inverted"),
    ("M37", "src/fabpc/simulator.py", "return self.alpha + self.beta * u + n * self.delta + eps", "return self.alpha + self.beta * u + (n - 1) * self.delta + eps", "drift index off by one"),
    # --- doe.py
    ("M38", "src/fabpc/doe.py", "D[:, j] = np.where((np.arange(n) // (2 ** j)) % 2 == 0, -1.0, 1.0)", "D[:, j] = np.where((np.arange(n) // (2 ** (k - 1 - j))) % 2 == 0, -1.0, 1.0)", "reversed standard order"),
    ("M39", "src/fabpc/doe.py", "return np.column_stack([base, base[:, 0] * base[:, 1] * base[:, 2]])", "return np.column_stack([base, base[:, 0] * base[:, 1]])", "generator 4 = 12 (resolution III)"),
    ("M40", "src/fabpc/doe.py", "B[i, j] = B[j, i] = beta[idx] / 2.0", "B[i, j] = B[j, i] = beta[idx]", "interaction not halved in B"),
    ("M41", "src/fabpc/doe.py", "xs = -0.5 * np.linalg.solve(B, b)", "xs = -1.0 * np.linalg.solve(B, b)", "stationary point factor"),
    ("M42", "src/fabpc/doe.py", "return float(n_factorial_runs) ** 0.25", "return float(n_factorial_runs) ** 0.5", "CCD alpha"),
    ("M43", "src/fabpc/doe.py", "out[name] = float(y[c > 0].mean() - y[c < 0].mean())", "out[name] = float(y[c > 0].mean() - y[c < 0].mean()) / 2.0", "effect = coefficient"),
    ("M44", "src/fabpc/doe.py", "f = ss / mse", "f = ss / full[\"sse\"]", "F uses SSE not MSE"),
    # --- common / fmea / report
    ("M45", "src/fabpc/common.py", "SIG_DIGITS = 10", "SIG_DIGITS = 12", "rounding precision"),
    ("M46", "src/fabpc/fmea.py", "return self.s * self.o * self.d", "return self.s * self.o + self.d", "RPN formula"),
    ("M47", "src/fabpc/report.py", "return f\"{f(d['point'], nd)} [{f(d['lo'], nd)}, {f(d['hi'], nd)}]\"", "return f\"{f(d['point'], nd)} [{f(d['hi'], nd)}, {f(d['lo'], nd)}]\"", "CI bounds swapped in tables"),
    ("M48", "src/fabpc/doe_study.py", "SCREEN_ALPHA = 0.20", "SCREEN_ALPHA = 0.05", "screening level changed"),
    # --- review round: permutation tests, random-ranking baseline, scan statistic
    ("M49", "src/fabpc/metrics.py", "np.sum(v >= o)", "np.sum(v > o)", "permutation p ignores ties with the observed value"),
    ("M50", "src/fabpc/metrics.py", "float((1 + np.sum(v >= o)) / (1 + n_perm))", "float(np.sum(v >= o) / n_perm)", "permutation p without the +1"),
    ("M51", "src/fabpc/metrics.py", "Yp[i] = rng.permutation(y)", "Yp[i] = y", "labels not permuted"),
    ("M52", "src/fabpc/metrics.py", "last = np.r_[np.where(np.diff(sd) != 0)[0], n - 1]", "last = np.arange(n)", "vectorised AP ignores ties"),
    ("M53", "src/fabpc/metrics.py", "ranks[order[a:b]] = 0.5 * (a + b - 1) + 1.0", "ranks[order[a:b]] = a + 1.0", "midranks not averaged"),
    ("M54", "src/fabpc/metrics.py", "(i / k) ** 2", "(i / k)", "random-AP formula without the square"),
    ("M55", "src/fabpc/metrics.py", "F[i] = rng.permutation(f)", "F[i] = f", "scan null does not move the alarms"),
    ("M56", "src/fabpc/metrics.py", "mx = _window_counts(F, w).max(axis=1)", "mx = _window_counts(F, w + 1).max(axis=1)", "scan null window one too long"),
    # --- CUSUM ARL simulation
    ("M57", "src/fabpc/spc.py", "sl[active] = np.maximum(0.0, sl[active] - x - k)", "sl[active] = np.maximum(0.0, sl[active] + x - k)", "ARL sim: lower CUSUM sign"),
    ("M58", "src/fabpc/spc.py", "x = rng.standard_normal(active.size) + shift", "x = rng.standard_normal(active.size) - shift", "ARL sim: shift sign"),
    ("M59", "src/fabpc/spc.py", "rl[active[sig]] = t", "rl[active[sig]] = t - 1", "ARL sim: run length off by one"),
    ("M60", "src/fabpc/spc.py", "        if two_sided:\n            sig |= sl[active] > h", "        if not two_sided:\n            sig |= sl[active] > h", "ARL sim: sides swapped"),
    # --- distribution-based limits
    ("M61", "src/fabpc/mspc.py", "h0 = 1.0 - 2.0 * t1 * t3 / (3.0 * t2 ** 2)", "h0 = 1.0", "JM limit: h0 dropped"),
    ("M62", "src/fabpc/mspc.py", "return float(A * (m + 1) * (m - 1) / (m * m - m * A) * f_dist.ppf(q, A, m - A))", "return float(A * (m - 1) / (m - A) * f_dist.ppf(q, A, m - A))", "T2 limit: Phase I form"),
    # --- model selection and rolling origin (leakage)
    ("M63", "src/fabpc/secom_analysis.py", "gb_cfg, s_h, gb_choice = fit_gb(d.X, y, fit, cal, keep)", "gb_cfg, s_h, gb_choice = fit_gb(d.X, y, fit, test, keep)", "GB tuned on the test block (leak)"),
    ("M64", "src/fabpc/secom_analysis.py", "    keep = secom.select_sensors(d.X[fit])\n    ref = fit[y[fit] == 0]", "    keep = secom.select_sensors(d.X)\n    ref = fit[y[fit] == 0]", "rolling fold: sensor filter on all rows (leak)"),
    ("M65", "src/fabpc/secom_analysis.py", 'gb_choice = fit_gb(d.X, y, fit, cal, keep)\n    # same', 'gb_choice = fit_gb(d.X, y, fit, test, keep)\n    # same', "rolling fold: GB tuned on its test block (leak)"),
    ("M77", "src/fabpc/secom_analysis.py", "fc = np.concatenate([fit, cal])", "fc = np.concatenate([fit, cal, test])", "rolling fold: refit includes the test block (leak)"),
    ("M66", "src/fabpc/secom_analysis.py", "Za = secom.Standardizer.fit(d.X[fit], keep).transform(d.X)", "Za = secom.Standardizer.fit(d.X[np.r_[fit, test]], keep).transform(d.X)", "rolling fold: LR scaler sees test rows (leak)"),
    ("M67", "src/fabpc/secom_analysis.py", 'null = np.mean([f["_nulls"][name][metric] for f in folds], axis=0)', 'null = folds[-1]["_nulls"][name][metric]', "pooled p uses one fold's null"),
    ("M68", "src/fabpc/secom_analysis.py", "LR_C_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)", "LR_C_GRID = (1e-3, 1e-2, 1e-1, 1.0)", "C grid back to the old edge"),
    ("M69", "src/fabpc/secom_analysis.py", "        f = rolling_fold(d, sp, PERM_SEED + i)", "        f = rolling_fold(d, sp, PERM_SEED)", "same permutation seed in every fold"),
    # --- excursion / SPC additions
    ("M70", "src/fabpc/secom_analysis.py", "shares.append(cnt[cnt > 1].sum() / v.size)", "shares.append(cnt[cnt > 1].size / v.size)", "repetition share counts distinct values"),
    ("M71", "src/fabpc/secom_analysis.py", "bonf = alpha / len(ps)", "bonf = alpha", "Bonferroni threshold not divided"),
    ("M72", "src/fabpc/secom_analysis.py", '"cal_far_at_cal_threshold": float(np.mean(s_cal[y_cal == 0] > thr))', '"cal_far_at_cal_threshold": float(np.mean(s_cal[y_cal == 0] >= thr))', "calibration FAR counts ties as alarms"),
    ("M73", "src/fabpc/secom_analysis.py", "scan = metrics.window_max_null(alarm, w, N_SCAN, SCAN_SEED)", "scan = metrics.window_max_null(alarm[win], w, N_SCAN, SCAN_SEED)", "scan null built on the window only"),
    # --- R2R tuning / DOE power
    ("M74", "src/fabpc/r2r_study.py", "sel = range(SEED_SELECT, SEED_SELECT + n_select)", "sel = range(SEED0, SEED0 + n_select)", "tuning selects on the evaluation seeds"),
    ("M75", "src/fabpc/r2r_study.py", 'b = min(rows, key=lambda r: r["rmse_selection"])', 'b = max(rows, key=lambda r: r["rmse_selection"])', "tuning picks the worst setting"),
    ("M76", "src/fabpc/doe_study.py", "kept[s] = p < SCREEN_ALPHA", "kept[s] = p < 0.05", "screening power uses the wrong level"),
]


NOTES = """\
* Earlier run (same 48 mutants): 47 / 48 killed. M14 ("Phase I re-admits excluded
  points") survived: with the original data the iterated and the 're-admitting'
  Phase I give the same points, but the mutant is not equivalent - a point
  excluded early can fall inside later, wider limits. A distinguishing sequence
  was found by random search and added as
  `tests/test_spc.py::test_phase1_never_readmits_excluded_points`.
* Review round (M49-M77): mutants for the permutation tests, the exact random-ranking
  AP, the scan statistic, the CUSUM run-length simulation, the distribution-based
  T2/SPE limits, hyperparameter selection and rolling-origin leakage, the excursion
  statistics, R2R tuning and DOE screening power. Two checks were written
  specifically so that a mutant would be visible: constant scores
  (`test_permutation_p_counts_ties_with_the_observed_value`, for M49) and the
  fit + cal score comparison in `test_rolling_folds_are_chronological_and_leak_free`
  (for M77).
* A first run of the review round was cut short from outside (the parent shell hit
  a time limit); two mutants were then recorded as "killed" although pytest had
  only been terminated. The runner now counts a mutant as killed only when pytest
  exits with "tests failed" and names the failing test; anything else is an error.
* No mutant in the list is known to be equivalent.
* Mutants whose first failing test is in `test_golden.py` were not caught by
  any unit test (tests run unit-first); they are caught by the golden
  byte-comparison or README-sync tests, which are part of the suite on purpose."""


def check_patterns() -> list[str]:
    errors = []
    ids = set()
    for mid, f, old, new, _ in MUTANTS:
        if mid in ids:
            errors.append(f"{mid}: duplicate id")
        ids.add(mid)
        text = (ROOT / f).read_text(encoding="utf-8")
        c = text.count(old)
        if c != 1:
            errors.append(f"{mid}: pattern occurs {c} times in {f}")
        if old == new:
            errors.append(f"{mid}: no-op mutation")
    return errors


def fresh_copy(work: Path) -> None:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for name in COPY:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, work / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, work / name)


def test_order(work: Path) -> list[str]:
    """Fast unit tests first, then data-dependent and golden (byte-comparison) tests,
    so the 'first failing test' column shows the most specific test that caught a bug."""
    last = ["tests/test_secom_data.py", "tests/test_golden.py"]
    first = sorted(p.relative_to(work).as_posix() for p in (work / "tests").glob("test_*.py"))
    return [t for t in first if t not in last] + last


def run_tests(work: Path, timeout: int) -> tuple[int, str, float]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(work / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *test_order(work)],
                           cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "TIMEOUT", -1
    return rc, out, time.time() - t0


def first_failure(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            return line.split(" - ")[0].replace("FAILED ", "").replace("ERROR ", "")
    return "TIMEOUT" if out == "TIMEOUT" else ""


def _mutant_key(mid: str) -> int:
    return int(mid[1:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--jobs", type=int, default=1, help="mutants run in parallel, each in its own copy")
    a = ap.parse_args()
    errs = check_patterns()
    if errs:
        print("\n".join(errs))
        return 2
    if a.check:
        print(f"{len(MUTANTS)} mutants, all patterns match exactly once")
        return 0
    works = [WORK / f"w{i}" for i in range(max(1, a.jobs))]
    if WORK.exists():
        shutil.rmtree(WORK)
    for w in works:
        fresh_copy(w)
    rc0, out0, _ = run_tests(works[0], a.timeout)
    if rc0 != 0:
        print("baseline test suite fails in the work copy:\n" + out0[-3000:])
        return 2
    secom_mode = "with SECOM data" if "skipped" not in out0.splitlines()[-1] else "without SECOM data (data tests skipped)"
    todo = [m for m in MUTANTS if not a.only or m[0] in a.only]
    q: queue.Queue = queue.Queue()
    for m in todo:
        q.put(m)
    results = []
    lock = threading.Lock()

    def worker(work: Path) -> None:
        while True:
            try:
                mid, f, old, new, desc = q.get_nowait()
            except queue.Empty:
                return
            target = work / f
            original = target.read_text(encoding="utf-8")
            target.write_text(original.replace(old, new, 1), encoding="utf-8", newline="\n")
            rc, out, dt = run_tests(work, a.timeout)
            target.write_text(original, encoding="utf-8", newline="\n")
            by = first_failure(out) if rc != 0 else ""
            # killed only when pytest itself reports a failing test; anything else (timeout,
            # interrupted run, collection crash without a test id) is an error, not a kill
            status = "survived" if rc == 0 else ("killed" if rc == 1 and by else "error")
            r = {"id": mid, "file": f, "description": desc, "killed": status == "killed", "status": status,
                 "returncode": rc, "killed_by": by, "seconds": round(dt, 1)}
            with lock:
                results.append(r)
                print(f"{mid} {status.upper():8} {desc}  {by}  (rc {rc}, {dt:.0f} s)", flush=True)

    threads = [threading.Thread(target=worker, args=(w,)) for w in works]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    results.sort(key=lambda r: _mutant_key(r["id"]))
    WORK.joinpath("mutation_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    killed = sum(r["killed"] for r in results)
    lines = [
        "# Mutation testing report",
        "",
        "Generated by `python scripts/mutate.py`. Each mutant is one deliberate bug applied to a copy of the",
        "repository; the full test suite is then run with `pytest -x`. Killed = the suite failed.",
        "",
        f"Test-suite mode: {secom_mode}.",
        "",
        f"**Killed {killed} / {len(results)} mutants** (survived: {sum(r['status'] == 'survived' for r in results)}, "
        f"errors: {sum(r['status'] == 'error' for r in results)}).",
        "",
        "Tests run in this order with `-x`: unit tests, then SECOM-data tests, then golden byte-comparison",
        "tests; the last column is the first test that failed.",
        "",
        "| ID | File | Bug | Result | First failing test |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['id']} | `{r['file']}` | {r['description']} | {r['status'] if r['status'] == 'killed' else '**' + r['status'] + '**'} | "
                     f"{r['killed_by'] or '-'} |")
    lines += ["", "## History and equivalent mutants", "", NOTES, ""]
    if a.only:
        print(f"killed {killed}/{len(results)} (subset run; report not written)")
        return 0
    (ROOT / "docs" / "mutation_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"killed {killed}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
