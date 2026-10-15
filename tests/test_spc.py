import numpy as np
import pytest

from fabpc import spc

# NIST e-Handbook 6.3.2.2 flow-rate example
FLOW = [49.6, 47.6, 49.9, 51.3, 47.8, 51.2, 52.6, 52.4, 53.6, 52.1]
# NIST e-Handbook 6.3.2.4 EWMA example
EWMA_DATA = [52.0, 47.0, 53.0, 49.3, 50.1, 47.0, 51.0, 50.1, 51.2, 50.5,
             49.6, 47.6, 49.9, 51.3, 47.8, 51.2, 52.6, 52.4, 53.6, 52.1]
EWMA_NIST = [50.60, 49.52, 50.56, 50.18, 50.16, 49.21, 49.75, 49.85, 50.26, 50.33,
             50.11, 49.36, 49.52, 50.05, 49.38, 49.92, 50.73, 51.23, 51.94, 51.99]
# NIST e-Handbook 6.3.2.3 CUSUM example (group means)
CUSUM_DATA = [324.925, 324.675, 324.725, 324.350, 325.350, 325.225, 324.125, 324.525,
              325.225, 324.600, 324.625, 325.150, 328.325, 327.250, 327.825, 328.500,
              326.675, 327.775, 326.875, 328.350]
CUSUM_HI_NIST = [0.00, 0.00, 0.00, 0.00, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00,
                 0.00, 0.00, 3.01, 4.94, 7.45, 10.63, 11.99, 14.44, 16.00, 19.04]
CUSUM_LO_NIST = [0.00, 0.01, 0.00, 0.33, 0.00, 0.00, 0.56, 0.72, 0.17, 0.25,
                 0.31, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]


def test_individuals_matches_nist_example():
    lim = spc.individuals_limits(np.array(FLOW))
    assert lim["center"] == pytest.approx(50.81, abs=1e-9)
    assert lim["mr_center"] == pytest.approx(16.9 / 9, abs=1e-12)
    assert lim["ucl"] == pytest.approx(55.8041, abs=5e-4)
    assert lim["lcl"] == pytest.approx(45.8159, abs=5e-4)
    assert lim["sigma"] == pytest.approx((16.9 / 9) / 1.128, rel=1e-12)
    assert lim["mr_ucl"] == pytest.approx(3.267 * 16.9 / 9, rel=1e-12)
    assert lim["mr_lcl"] == 0.0


def test_moving_range():
    assert spc.moving_range(np.array([3.0, 7.5, 6.0, 10.0])).tolist() == [4.5, 1.5, 4.0]


def test_individuals_needs_two_points():
    with pytest.raises(ValueError):
        spc.individuals_limits(np.array([1.0]))


def test_factor_table_consistent_with_d2():
    # A2 = 3 / (d2 sqrt(n)); for n = 2, d2 = 1.128
    a2 = spc.XBAR_R_FACTORS[2][0]
    assert a2 == pytest.approx(3.0 / (spc.D2_N2 * np.sqrt(2.0)), abs=1e-3)
    # D3 is zero up to n = 6 and positive from n = 7; D4 decreases with n
    d3 = [spc.XBAR_R_FACTORS[n][1] for n in range(2, 11)]
    d4 = [spc.XBAR_R_FACTORS[n][2] for n in range(2, 11)]
    assert d3[:5] == [0.0] * 5 and all(v > 0 for v in d3[5:])
    assert all(a > b for a, b in zip(d4, d4[1:]))
    assert spc.XBAR_R_FACTORS[5] == (0.577, 0.0, 2.115)
    assert spc.XBAR_R_FACTORS[8] == (0.373, 0.136, 1.864)


def test_xbar_r_limits_hand_computed():
    sg = np.array([[10.0, 12.0, 11.0, 15.0],
                   [9.0, 9.5, 13.0, 11.5],
                   [14.0, 10.0, 12.5, 12.5]])
    lim = spc.xbar_r_limits(sg)
    xbar = sg.mean(axis=1)
    r = np.array([5.0, 4.0, 4.0])
    assert lim["xbar_center"] == pytest.approx(xbar.mean())
    assert lim["r_center"] == pytest.approx(r.mean())
    assert lim["xbar_ucl"] == pytest.approx(xbar.mean() + 0.729 * r.mean())
    assert lim["xbar_lcl"] == pytest.approx(xbar.mean() - 0.729 * r.mean())
    assert lim["r_ucl"] == pytest.approx(2.282 * r.mean())
    assert lim["r_lcl"] == 0.0
    assert lim["n"] == 4 and lim["m"] == 3


def test_xbar_r_rejects_bad_shapes():
    with pytest.raises(ValueError):
        spc.xbar_r_limits(np.ones((3, 11)))
    with pytest.raises(ValueError):
        spc.xbar_r_limits(np.ones(5))


def test_ewma_matches_nist_example():
    out = spc.ewma(np.array(EWMA_DATA), 0.3, 50.0)
    assert np.round(out, 2).tolist() == pytest.approx(EWMA_NIST, abs=0.006)
    lim = spc.ewma_limits(50.0, 2.0539, 0.3)
    assert lim["ucl"] == pytest.approx(52.5884, abs=1e-3)
    assert lim["lcl"] == pytest.approx(47.4115, abs=1e-3)


def test_ewma_lambda_validation_and_lambda_one():
    x = np.array([2.0, -1.0, 4.0])
    assert spc.ewma(x, 1.0, 100.0).tolist() == x.tolist()
    with pytest.raises(ValueError):
        spc.ewma(x, 0.0, 0.0)
    with pytest.raises(ValueError):
        spc.ewma(x, 1.2, 0.0)


def test_tabular_cusum_matches_nist_example():
    res = spc.tabular_cusum(np.array(CUSUM_DATA), 325.0, 0.3175, 4.1959)
    assert np.round(res["s_hi"], 2).tolist() == pytest.approx(CUSUM_HI_NIST, abs=0.006)
    assert np.round(res["s_lo"], 2).tolist() == pytest.approx(CUSUM_LO_NIST, abs=0.006)
    sig = np.where(res["signal"])[0]
    assert sig[0] == 13            # NIST: first out-of-control signal at group 14
    assert sig.tolist() == list(range(13, 20))


def test_cusum_lower_side_signals_on_downward_shift():
    x = np.r_[np.zeros(5), np.full(6, -1.0)]
    res = spc.tabular_cusum(x, 0.0, 0.25, 2.0)
    assert not res["signal"][:7].any()
    assert res["signal"][8]
    assert res["s_hi"].max() == 0.0


def _rules(x, c=10.0, s=2.0):
    return spc.weco_rules(np.asarray(x, dtype=float), c, s)


def test_rule1_strictly_beyond_three_sigma():
    r = _rules([10.0, 16.0, 16.1, 3.9, 4.0])
    assert r["rule1"].tolist() == [False, False, True, True, False]


def test_rule2_two_of_three_same_side():
    up = _rules([10.0, 14.5, 9.0, 14.5, 10.0])
    assert up["rule2"].tolist() == [False, False, False, True, False]
    mixed = _rules([14.5, 5.5, 14.4, 10.0])     # 1st and 3rd are both above +2 sigma
    assert mixed["rule2"].tolist() == [False, False, True, False]
    split = _rules([14.5, 5.5, 10.0, 5.6])
    assert split["rule2"].tolist() == [False, False, False, True]
    none = _rules([14.5, 5.5, 10.0, 10.0])
    assert not none["rule2"].any()


def test_rule3_four_of_five_beyond_one_sigma():
    x = [12.5, 12.5, 9.0, 12.5, 12.5, 9.0]
    r = _rules(x)
    assert r["rule3"].tolist() == [False, False, False, False, True, False]
    low = _rules([7.5, 7.5, 7.5, 11.0, 7.4])
    assert low["rule3"].tolist() == [False, False, False, False, True]
    three = _rules([12.5, 12.5, 9.0, 12.5, 9.0])
    assert not three["rule3"].any()


def test_rule4_eight_consecutive_same_side():
    seven = _rules([10.5] * 7 + [9.0])
    assert not seven["rule4"].any()
    eight = _rules([10.5] * 8 + [9.5] * 8)
    assert eight["rule4"].tolist() == [False] * 7 + [True] + [False] * 7 + [True]
    onc = _rules([10.5] * 4 + [10.0] + [10.5] * 4)   # a point on the centre line breaks the run
    assert not onc["rule4"].any()


def test_trend_rule_six_points():
    five = _rules([1.0, 2.0, 3.0, 4.0, 5.0, 4.0])
    assert not five["trend6"].any()
    six = _rules([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.5])
    assert six["trend6"].tolist() == [False] * 5 + [True, False]
    down = _rules([9.0, 8.0, 7.5, 7.0, 6.0, 5.0])
    assert down["trend6"].tolist() == [False] * 5 + [True]


def test_alternating_rule_fourteen_points():
    alt14 = [10.0 + (1.0 if i % 2 else -1.0) for i in range(14)]
    r = _rules(alt14)
    assert r["alt14"].tolist() == [False] * 13 + [True]
    r13 = _rules(alt14[:13])
    assert not r13["alt14"].any()
    broken = alt14[:7] + [alt14[6]] + alt14[7:]
    assert not _rules(broken)["alt14"].any()


def test_capability_hand_computed():
    x = np.array([9.8, 10.1, 10.4, 9.9, 10.6, 10.2])
    cap = spc.capability(x, 9.0, 11.5)
    m, s = x.mean(), x.std(ddof=1)
    assert cap["cp"] == pytest.approx(2.5 / (6 * s))
    assert cap["cpk"] == pytest.approx(min((11.5 - m) / (3 * s), (m - 9.0) / (3 * s)))
    assert cap["cpk"] < cap["cp"]
    assert cap["frac_out_of_spec"] == 0.0
    cap2 = spc.capability(x, 10.0, 11.5)
    assert cap2["frac_out_of_spec"] == pytest.approx(2 / 6)
    with pytest.raises(ValueError):
        spc.capability(x, 5.0, 5.0)


def test_phase1_excludes_outlier_and_reestimates():
    rng = np.random.default_rng(3)
    x = rng.normal(20.0, 1.0, 60)
    x[17] = 40.0
    lim0 = spc.individuals_limits(x)
    lim, keep = spc.phase1_individuals(x, max_iter=1)
    assert not keep[17] and keep.sum() == 59
    assert lim["center"] == pytest.approx(np.delete(x, 17).mean())
    assert lim["ucl"] < lim0["ucl"]
    limf, keepf = spc.phase1_individuals(x)          # iterate to convergence
    assert not (keepf & ~keep).any() and keepf.sum() < 59
    xr = x[keepf]
    assert ((xr <= limf["ucl"]) & (xr >= limf["lcl"])).all()
    assert limf["center"] == pytest.approx(xr.mean())
    lim1, keep1 = spc.phase1_individuals(x, max_iter=0)
    assert keep1.all() and lim1["ucl"] == lim0["ucl"]


def test_phase1_never_readmits_excluded_points():
    # here an excluded point would fall back inside the later, wider limits
    x = np.array([-0.2, -0.9, -1.2, 0.4, 0.9, 2.4, 3.5, 2.8, 7.8, 2.7, 3.7])
    prev = np.ones(x.size, dtype=bool)
    for it in range(1, 6):
        _, keep = spc.phase1_individuals(x, max_iter=it)
        assert not (keep & ~prev).any()
        prev = keep
    _, keep = spc.phase1_individuals(x)
    assert keep.astype(int).tolist() == [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0]
    lim0 = spc.individuals_limits(x)
    lim1, keep1 = spc.phase1_individuals(x, max_iter=0)
    assert keep1.all() and lim1["ucl"] == lim0["ucl"]


@pytest.mark.parametrize("two_sided", [False, True])
def test_cusum_run_length_equals_first_signal_of_tabular_cusum(two_sided):
    """With one replication the simulator draws the same numbers as one batch draw,
    so its run length must be the first signal of ``tabular_cusum`` (same rules)."""
    for seed in range(5):
        rl = int(spc.cusum_run_lengths(0.5, 2.0, 1, seed, shift=0.3, two_sided=two_sided)[0])
        x = np.random.default_rng(seed).standard_normal(rl) + 0.3
        cs = spc.tabular_cusum(x, 0.0, 0.5, 2.0)
        sig = cs["signal"] if two_sided else cs["s_hi"] > 2.0
        assert np.where(sig)[0][0] == rl - 1


def test_cusum_run_length_exact_cases():
    # k = 0, h = 0: two-sided chart signals on the first point; one-sided waits for x > 0 (geometric, mean 2)
    assert (spc.cusum_run_lengths(0.0, 0.0, 500, 1, two_sided=True) == 1).all()
    rl = spc.cusum_run_lengths(0.0, 0.0, 20000, 1, two_sided=False)
    assert rl.min() == 1 and rl.mean() == pytest.approx(2.0, abs=0.05)


def test_cusum_arl_matches_nist_one_sided_table():
    # NIST 6.3.2.3.1, k = 0.5: one-sided in-control ARL 336 (h = 4); ARL 10.4 at a 1-sigma shift (h = 5)
    rl = spc.cusum_run_lengths(0.5, 4.0, 4000, 3, two_sided=False)
    se = rl.std(ddof=1) / np.sqrt(rl.size)
    assert abs(rl.mean() - 336.0) < 4 * se
    rl = spc.cusum_run_lengths(0.5, 5.0, 4000, 4, shift=1.0, two_sided=False)
    se = rl.std(ddof=1) / np.sqrt(rl.size)
    assert abs(rl.mean() - 10.4) < 4 * se
    # running both sides roughly halves the in-control ARL
    two = spc.cusum_run_lengths(0.5, 4.0, 4000, 5, two_sided=True).mean()
    assert two == pytest.approx(336.0 / 2, rel=0.08)


def test_arl_study_reports_both_versions():
    from fabpc import spc_arl
    r = spc_arl.run(n_rep=300)
    row = [x for x in r["rows"] if x["h"] == 5.0 and x["shift_sigma"] == 0.0][0]
    assert row["nist_one_sided"] == 930.0
    assert row["two_sided"]["arl"] < 0.75 * row["one_sided"]["arl"]
