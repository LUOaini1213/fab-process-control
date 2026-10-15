import numpy as np
import pytest

from fabpc import r2r_study
from fabpc.r2r import ControllerSpec, ewma_drift_bias, ewma_stable, simulate, summarize
from fabpc.simulator import LinearDriftProcess

PROC = LinearDriftProcess(alpha=12.0, beta=2.5, delta=0.3, sigma=0.0)


def _run(ctrl, n=4000, b=2.1, target=80.0, proc=PROC, a0=0.0):
    return simulate(proc, ctrl, target, b, n, np.zeros(n), a0=a0)


def test_process_model():
    p = LinearDriftProcess(alpha=1.5, beta=3.0, delta=-0.2, sigma=1.0)
    assert p.output(4, 2.0, 0.7) == pytest.approx(1.5 + 6.0 - 0.8 + 0.7)


def test_first_steps_hand_computed():
    proc = LinearDriftProcess(alpha=10.0, beta=2.0, delta=1.0, sigma=0.0)
    eps = np.array([0.5, -0.25, 0.0])
    r = simulate(proc, ControllerSpec("ewma", 0.4), 50.0, 2.5, 3, eps, a0=4.0)
    u1 = (50.0 - 4.0) / 2.5
    y1 = 10.0 + 2.0 * u1 + 1.0 + 0.5
    a1 = 0.4 * (y1 - 2.5 * u1) + 0.6 * 4.0
    u2 = (50.0 - a1) / 2.5
    y2 = 10.0 + 2.0 * u2 + 2.0 - 0.25
    assert r["u"][:2] == pytest.approx([u1, u2])
    assert r["y"][:2] == pytest.approx([y1, y2])


def test_dewma_first_steps_hand_computed():
    proc = LinearDriftProcess(alpha=10.0, beta=2.0, delta=1.0, sigma=0.0)
    eps = np.array([0.5, -0.25, 0.0])
    r = simulate(proc, ControllerSpec("dewma", 0.4, 0.3), 50.0, 2.5, 3, eps, a0=4.0)
    u1 = 46.0 / 2.5
    y1 = 10.0 + 2.0 * u1 + 1.0 + 0.5
    res1 = y1 - 2.5 * u1
    a1 = 0.4 * res1 + 0.6 * (4.0 + 0.0)
    p1 = 0.3 * (res1 - 4.0) + 0.7 * 0.0
    u2 = (50.0 - (a1 + p1)) / 2.5
    y2 = 10.0 + 2.0 * u2 + 2.0 - 0.25
    res2 = y2 - 2.5 * u2
    a2 = 0.4 * res2 + 0.6 * (a1 + p1)
    p2 = 0.3 * (res2 - a1) + 0.7 * p1
    u3 = (50.0 - (a2 + p2)) / 2.5
    assert r["u"] == pytest.approx([u1, u2, u3])
    assert r["y"][1] == pytest.approx(y2)


@pytest.mark.parametrize("lam", [0.15, 0.45, 0.7])
def test_ewma_drift_bias_matches_theory(lam):
    r = _run(ControllerSpec("ewma", lam))
    expected = ewma_drift_bias(lam, PROC.beta, 2.1, PROC.delta)
    assert expected == pytest.approx(2.1 * 0.3 / (2.5 * lam))
    assert r["y"][-1] - 80.0 == pytest.approx(expected, abs=1e-6)


def test_dewma_removes_drift_bias():
    r = _run(ControllerSpec("dewma", 0.35, 0.12))
    assert r["y"][-1] - 80.0 == pytest.approx(0.0, abs=1e-6)
    e = _run(ControllerSpec("ewma", 0.35))
    assert abs(e["y"][-1] - 80.0) > 0.5


def test_dewma_with_zero_drift_weight_equals_ewma():
    rng = np.random.default_rng(12)
    eps = rng.normal(0, 1.0, 200)
    p = LinearDriftProcess(alpha=3.0, beta=1.7, delta=0.05, sigma=1.0)
    a = simulate(p, ControllerSpec("dewma", 0.3, 0.0), 40.0, 1.5, 200, eps, a0=1.0)
    b = simulate(p, ControllerSpec("ewma", 0.3), 40.0, 1.5, 200, eps, a0=1.0)
    assert a["y"] == pytest.approx(b["y"], abs=1e-9)


def test_no_control_keeps_recipe():
    r = _run(ControllerSpec("none"), n=50, a0=3.0)
    assert np.all(r["u"] == (80.0 - 3.0) / 2.1)
    assert r["y"][-1] - r["y"][0] == pytest.approx(49 * PROC.delta)


@pytest.mark.parametrize("g,stable", [(0.05, True), (1.9, True), (2.1, False), (-0.2, False)])
def test_stability_condition_numerically(g, stable):
    b = 2.1
    lam = g * b / PROC.beta
    assert ewma_stable(lam, PROC.beta, b) is stable
    with np.errstate(all="ignore"):
        r = _run(ControllerSpec("ewma", lam), n=6000, b=b)
    err = r["y"][-1] - 80.0
    if stable:
        assert err == pytest.approx(ewma_drift_bias(lam, PROC.beta, b, PROC.delta), abs=1e-6)
    else:
        assert not np.isfinite(err) or abs(err) > 1e6


def test_stability_boundary_inclusive_exclusive():
    assert not ewma_stable(2.0, 1.0, 1.0)
    assert not ewma_stable(0.0, 1.0, 1.0)
    assert ewma_stable(1.999, 1.0, 1.0)


def test_simulate_input_validation():
    with pytest.raises(ValueError):
        simulate(PROC, ControllerSpec("ewma", 0.2), 1.0, 1.0, 5, np.zeros(4))
    with pytest.raises(ValueError):
        simulate(PROC, ControllerSpec("pid", 0.2), 1.0, 1.0, 3, np.zeros(3))


def test_summarize():
    y = np.array([0.0, 0.0, 9.0, 11.0, 10.5, 14.0])
    s = summarize(y, 10.0, 8.0, 12.0, burn_in=2)
    assert s["mean_offset"] == pytest.approx((-1 + 1 + 0.5 + 4) / 4)
    assert s["rmse"] == pytest.approx(np.sqrt((1 + 1 + 0.25 + 16) / 4))
    assert s["pct_in_spec"] == pytest.approx(75.0)
    assert s["std"] == pytest.approx(np.std([9.0, 11.0, 10.5, 14.0], ddof=1))


def test_study_small_run_shapes_and_theory_agreement():
    r = r2r_study.run_comparison(n_rep=4, n_runs=150)
    labels = [c["controller"] for c in r["controllers"]]
    assert labels[0] == "no control" and len(labels) == 5
    for c in r["controllers"]:
        if c["kind"] == "ewma":
            assert c["theory_bias"] == pytest.approx(
                r2r_study.B_MODEL * r2r_study.PROC.delta / (r2r_study.PROC.beta * c["w1"]))
    st = r2r_study.stability_sweep(n_runs=3000)
    assert st["n_agree"] == st["n_total"]
    assert any(row["xi_lambda"] == 2.0 and row["empirical"] != "converged" for row in st["rows"])


def test_tuning_selects_on_selection_reps_and_evaluates_on_independent_reps():
    t = r2r_study.tune(n_select=3, n_eval=4, n_runs=150)
    s0, s1 = t["selection_seeds"]
    e0, e1 = t["evaluation_seeds"]
    assert s1 - s0 + 1 == 3 and e1 - e0 + 1 == 4
    assert s1 < e0 or e1 < s0                       # disjoint seed ranges
    for kind in ("ewma", "dewma"):
        rows = [r for r in t["grid"] if r["kind"] == kind]
        assert t["best"][kind]["rmse_selection"] == min(r["rmse_selection"] for r in rows)
        # the evaluation number is recomputed on the evaluation seeds, not copied
        spec = ControllerSpec(kind, t["best"][kind]["w1"], t["best"][kind]["w2"])
        direct = r2r_study._mean_rmse(spec, range(e0, e1 + 1), 150).mean()
        assert t["best"][kind]["rmse_evaluation"] == pytest.approx(direct, rel=1e-12)
    d = t["paired_rmse_diff_dewma_minus_ewma"]
    assert d["mean"] == pytest.approx(t["best"]["dewma"]["rmse_evaluation"] - t["best"]["ewma"]["rmse_evaluation"],
                                      rel=1e-9)
