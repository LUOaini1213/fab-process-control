"""Command-line entry point.

    python -m fabpc.cli all        # everything (SECOM parts need SECOM_DIR)
    python -m fabpc.cli secom      # data profile, SPC, FDC, excursion (+ figures)
    python -m fabpc.cli arl | doe | r2r | fmea
    python -m fabpc.cli report     # results/summary.md + README blocks from results/*.json
    python -m fabpc.cli report --check   # exit 1 if README blocks are stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import doe, doe_study, fmea, plots, r2r_study, report, secom
from .common import write_json, write_text

RESULTS = Path("results")
FIG = RESULTS / "figures"


def _strip(d):
    """Drop private keys (leading underscore) and raw series before writing JSON."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if not str(k).startswith("_") and k != "series"}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def cmd_secom(results: Path = RESULTS, figures: bool = True) -> int:
    from . import secom_analysis as sa
    if not secom.data_available():
        print("SECOM data not found; set SECOM_DIR (see scripts/fetch_secom.py). Skipping.", file=sys.stderr)
        return 2
    d = secom.load()
    out = sa.run_all(d)
    write_json(results / "data_profile.json", out["profile"])
    write_json(results / "spc_summary.json", _strip(out["spc"]))
    write_json(results / "fdc_results.json", _strip(out["fdc"]))
    write_json(results / "excursion.json", _strip(out["excursion"]))
    write_json(results / "fdc_rolling.json", _strip(out["rolling"]))
    if figures:
        fig = results / "figures"
        plots.missing_profile(np.isnan(d.X).mean(axis=0), fig / "data_missing_profile.png")
        for name, res in out["spc"]["sensors"].items():
            plots.spc_sensor(name, res, sa.EWMA_LAMBDA, sa.CUSUM_H, fig / f"spc_{name}.png")
        pca80 = out["fdc"]["pca"]["80"]
        plots.t2_spe(out["excursion"], pca80["t2_ucl99"], pca80["spe_ucl99"], fig / "fdc_t2_spe_test.png")
        plots.contributions(out["excursion"], fig / "fdc_excursion_contributions.png")
        plots.fdc_methods(out["fdc"]["methods"], out["fdc"]["random_ranking_pr_auc"], fig / "fdc_pr_auc.png")
    return 0


def cmd_doe(results: Path = RESULTS, figures: bool = True) -> int:
    r = doe_study.run()
    write_json(results / "doe_results.json", r)
    if figures:
        fig = results / "figures"
        plots.doe_effects(r["stage2_full_factorial"]["effects"], fig / "doe_factorial_effects.png",
                          "SIMULATION: 2^3 factorial effects")
        s3 = r["stage3_ccd"]
        beta = np.array([s3["coefficients"][k] for k in doe.quad_model_matrix(np.zeros((1, 3)))[0]])
        xs = np.asarray(s3["stationary_point"])
        xo = np.asarray(s3["true_optimum_x"])
        from .simulator import toy_cmp_nonuniformity
        f_true = lambda P: toy_cmp_nonuniformity(np.column_stack([P, np.full(len(P), xo[2]), np.zeros(len(P))]))
        f_fit = lambda P: doe.quad_predict(beta, np.column_stack([P, np.full(len(P), xs[2])]))
        plots.doe_contours(f_true, f_fit, xs, xo, fig / "doe_ccd_contours.png")
    return 0


def cmd_r2r(results: Path = RESULTS, figures: bool = True) -> int:
    r = r2r_study.run()
    write_json(results / "r2r_results.json", r)
    if figures:
        fig = results / "figures"
        plots.r2r_trajectories(r2r_study.example_trajectories(), r2r_study.TARGET, r2r_study.SPEC_HALF,
                               fig / "r2r_trajectories.png")
        plots.r2r_stability(r["stability"]["rows"], fig / "r2r_stability.png")
    return 0


def cmd_arl(results: Path = RESULTS, figures: bool = False) -> int:
    from . import spc_arl
    write_json(results / "cusum_arl.json", spc_arl.run())
    return 0


def cmd_fmea(results: Path = RESULTS) -> int:
    rows = fmea.table()
    write_json(results / "fmea.json", {"rows": rows, "rpn_definition": "S x O x D, each 1-10"})
    write_text(results / "fmea_table.md", fmea.to_markdown(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fabpc")
    ap.add_argument("command", choices=["all", "secom", "arl", "doe", "r2r", "fmea", "report"])
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--check", action="store_true", help="report: fail if README blocks are stale")
    a = ap.parse_args(argv)
    figs = not a.no_figures
    rc = 0
    if a.command in ("all", "secom"):
        rc = max(rc, cmd_secom(figures=figs))
    if a.command in ("all", "doe"):
        rc = max(rc, cmd_doe(figures=figs))
    if a.command in ("all", "r2r"):
        rc = max(rc, cmd_r2r(figures=figs))
    if a.command in ("all", "fmea"):
        rc = max(rc, cmd_fmea())
    if a.command in ("all", "arl"):
        rc = max(rc, cmd_arl())
    if a.command in ("all", "report"):
        rc = max(rc, report.main(check=a.check))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
