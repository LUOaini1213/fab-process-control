import numpy as np
import pytest

from fabpc import doe_study
from fabpc.simulator import toy_cmp_nonuniformity


def test_closed_form_optimum_matches_numerical_minimiser():
    t = doe_study.true_optimum()
    n = doe_study.numeric_optimum()
    assert t["x"] == pytest.approx(n["x"], abs=1e-5)
    assert t["y"] <= n["y"] + 1e-12
    assert t["x"][3] == 0.0
    # finite-difference gradient is ~0 at the closed-form point
    h = 1e-6
    for j in range(3):
        e = np.zeros(4)
        e[j] = h
        g = (toy_cmp_nonuniformity(t["x"] + e)[0] - toy_cmp_nonuniformity(t["x"] - e)[0]) / (2 * h)
        assert abs(g) < 1e-6


def test_inert_factor_has_no_effect():
    x = np.array([[0.3, -0.7, 1.1, -2.0], [0.3, -0.7, 1.1, 1.5]])
    y = toy_cmp_nonuniformity(x)
    assert y[0] == y[1]


def test_embed_places_active_factors():
    X = doe_study.embed(np.array([[1.0, 2.0], [3.0, 4.0]]), [0, 2])
    assert X.tolist() == [[1.0, 0.0, 2.0, 0.0], [3.0, 0.0, 4.0, 0.0]]


def test_stage1_screens_out_the_inert_factor():
    s1 = doe_study.stage1(np.random.default_rng(doe_study.SEED))
    assert 3 not in s1["active"]
    assert s1["active"][:2] == [0, 1]
    assert s1["anova"]["4"]["p"] > doe_study.SCREEN_ALPHA


def test_ccd_optimum_close_to_truth():
    t = doe_study.true_optimum()
    r = doe_study.stage3_once(np.random.default_rng(5), [0, 1, 2])
    assert r["stationary"]["nature"] == "minimum"
    assert np.linalg.norm(r["x_star"] - t["x"]) < 0.3
    assert r["y_true_at_xstar"] - t["y"] < r["y_true_at_best_run"] - t["y"]


def test_screening_power_rates():
    r = doe_study.screening_power(n_sim=60)
    k = r["kept_rate"]
    assert k["downforce"] == 1.0
    assert 0.0 < k["conditioner_sweep"] < 0.5        # inert factor: false inclusions near the 0.20 level
    assert all(r["kept_rate_at_0.05"][f] <= k[f] for f in k)
    # the first simulated screen is the stage-1 screen with that seed
    an = doe_study.stage1(np.random.default_rng(doe_study.SCREEN_SIM_SEED))["anova"]
    one = doe_study.screening_power(n_sim=1)["kept_rate"]
    assert one["slurry_flow"] == float(an["3"]["p"] < doe_study.SCREEN_ALPHA)
