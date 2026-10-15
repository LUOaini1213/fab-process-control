import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fabpc import metrics


def _scores(seed=5, n=240):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.12).astype(int)
    s = rng.normal(size=n) + 0.9 * y
    s = np.round(s, 1)          # create ties on purpose
    return s, y


@pytest.mark.parametrize("seed", [1, 5, 9])
def test_average_precision_matches_sklearn(seed):
    s, y = _scores(seed)
    assert metrics.average_precision(s, y) == pytest.approx(average_precision_score(y, s), abs=1e-12)


@pytest.mark.parametrize("seed", [1, 5, 9])
def test_roc_auc_matches_sklearn(seed):
    s, y = _scores(seed)
    assert metrics.roc_auc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_ap_hand_example():
    # ranked: 0.9(1) 0.8(0) 0.7(1) 0.1(0) -> AP = 0.5*1 + 0.5*(2/3)
    s = np.array([0.1, 0.7, 0.9, 0.8])
    y = np.array([0, 1, 1, 0])
    assert metrics.average_precision(s, y) == pytest.approx(0.5 + 1.0 / 3.0)
    assert metrics.roc_auc(s, y) == pytest.approx(0.75)
    assert np.isnan(metrics.average_precision(s, np.zeros(4, dtype=int)))
    assert np.isnan(metrics.roc_auc(s, np.zeros(4, dtype=int)))


def test_threshold_at_far_bounds_false_alarms():
    rng = np.random.default_rng(2)
    p = rng.normal(size=401)
    for far in (0.01, 0.05, 0.2):
        thr = metrics.threshold_at_far(p, far)
        realised = np.mean(p > thr)
        assert realised <= far
        assert realised > far - 2.0 / p.size
    with pytest.raises(ValueError):
        metrics.threshold_at_far(p, 0.0)


def test_alarm_rates_counts_and_strict_inequality():
    s = np.array([0.2, 0.5, 0.5, 0.9, 0.1, 0.7])
    y = np.array([0, 1, 0, 1, 0, 0])
    r = metrics.alarm_rates(s, y, 0.5)
    assert (r["tp"], r["fp"]) == (1, 1)
    assert r["recall"] == pytest.approx(0.5)
    assert r["far"] == pytest.approx(0.25)
    assert r["precision"] == pytest.approx(0.5)
    assert r["n_fail"] == 2 and r["n_pass"] == 4


def test_oracle_recall():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.05, 0.9, 0.25])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    # pass scores 0.1 .. 0.4; far=0.2 -> 'higher' quantile 0.4 -> only 0.9 alarms
    assert metrics.recall_at_far_oracle(s, y, 0.2) == pytest.approx(1.0 / 3.0)


def test_bootstrap_is_deterministic_and_brackets_point():
    s, y = _scores(7, 300)
    a = metrics.stratified_bootstrap(s, y, metrics.average_precision, n_boot=300, seed=4)
    b = metrics.stratified_bootstrap(s, y, metrics.average_precision, n_boot=300, seed=4)
    c = metrics.stratified_bootstrap(s, y, metrics.average_precision, n_boot=300, seed=5)
    assert a == b
    assert a != c
    assert a["lo"] < a["point"] < a["hi"]
    assert a["point"] == pytest.approx(metrics.average_precision(s, y))


def test_bootstrap_keeps_class_counts():
    s, y = _scores(3, 200)
    seen = []
    metrics.stratified_bootstrap(s, y, lambda ss, yy: seen.append(int(yy.sum())) or 0.0, n_boot=20, seed=1)
    assert set(seen[:-1]) == {int(y.sum())}


@pytest.mark.parametrize("seed", [1, 5, 9])
def test_vectorised_ranking_metrics_equal_single_versions(seed):
    s, y = _scores(seed)
    rng = np.random.default_rng(seed)
    Y = np.array([rng.permutation(y) for _ in range(25)] + [y])
    got = metrics.ranking_metrics_many(s, Y)
    for i, yy in enumerate(Y):
        assert got["pr_auc"][i] == pytest.approx(metrics.average_precision(s, yy), abs=1e-12)
        assert got["roc_auc"][i] == pytest.approx(metrics.roc_auc(s, yy), abs=1e-12)


def test_midranks_average_ties():
    assert metrics.midranks(np.array([3.0, 1.0, 3.0, 2.0, 3.0])).tolist() == [4.0, 1.0, 4.0, 2.0, 4.0]


def test_permutation_test_perfect_and_null_rankings():
    y = np.r_[np.ones(10, int), np.zeros(90, int)]
    perfect = metrics.permutation_test(y.astype(float) * 5.0 + np.arange(100) * 1e-3, y, n_perm=500, seed=2)
    assert perfect["pr_auc"]["observed"] == 1.0 and perfect["roc_auc"]["observed"] == 1.0
    assert perfect["pr_auc"]["p"] == pytest.approx(1 / 501) and perfect["roc_auc"]["p"] == pytest.approx(1 / 501)
    worst = metrics.permutation_test(-(y.astype(float)), y, n_perm=500, seed=2)
    assert worst["roc_auc"]["p"] == 1.0
    # null of ROC-AUC is centred on 0.5; null of PR-AUC on the random-ranking expectation
    rnd = metrics.permutation_test(np.arange(100, dtype=float), y, n_perm=4000, seed=3)
    assert rnd["roc_auc"]["null_mean"] == pytest.approx(0.5, abs=0.01)
    assert rnd["pr_auc"]["null_mean"] == pytest.approx(metrics.random_ap_expected(100, 10), abs=0.006)
    assert rnd["pr_auc"]["null_p95"] > rnd["pr_auc"]["null_mean"]
    again = metrics.permutation_test(np.arange(100, dtype=float), y, n_perm=4000, seed=3)
    assert again["pr_auc"] == rnd["pr_auc"]


def test_permutation_p_counts_ties_with_the_observed_value():
    # constant scores: every permutation gives the same PR-AUC and ROC-AUC = 0.5, so p must be 1
    y = np.r_[np.ones(6, int), np.zeros(30, int)]
    r = metrics.permutation_test(np.zeros(36), y, n_perm=200, seed=1)
    assert r["roc_auc"]["observed"] == 0.5 and r["roc_auc"]["p"] == 1.0
    assert r["pr_auc"]["p"] == 1.0


def test_random_ap_expected_matches_bestgen_examples():
    # Bestgen (2015), Table 1 / R example: N=5, R=2 -> 0.593; RandomAPExact(10, 4) = 0.5285979
    assert metrics.random_ap_expected(5, 2) == pytest.approx(0.593, abs=5e-4)
    assert metrics.random_ap_expected(10, 4) == pytest.approx(0.5285979, abs=1e-7)
    # brute force over all rankings for a small case
    from itertools import combinations
    vals = []
    for pos in combinations(range(7), 3):
        y = np.zeros(7, int)
        y[list(pos)] = 1
        vals.append(metrics.average_precision(-np.arange(7.0), y))
    assert metrics.random_ap_expected(7, 3) == pytest.approx(np.mean(vals), abs=1e-12)


def test_window_max_null_small_case():
    # 2 flags among 4 positions, windows of 2: max = 2 iff the flags are adjacent (3 of 6 placements)
    r = metrics.window_max_null(np.array([1, 1, 0, 0], bool), 2, n_sim=20000, seed=1)
    assert r["observed_max"] == 2 and r["n_windows"] == 3 and r["n_flags"] == 2
    assert r["p_value"] == pytest.approx(0.5, abs=0.02)
    assert r["null_mean_max"] == pytest.approx(1.5, abs=0.02)
    r2 = metrics.window_max_null(np.array([1, 0, 1, 0, 0, 0], bool), 3, n_sim=20000, seed=1)
    # windows of 3 over 6 positions, 2 flags: max = 1 iff the flags are >= 3 apart: 6 of 15 placements
    assert r2["observed_max"] == 2
    assert r2["p_value"] == pytest.approx(9 / 15, abs=0.02)
