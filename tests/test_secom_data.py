"""Tests on the real SECOM files. Skipped cleanly when the data are not present."""

import numpy as np
import pytest

from fabpc import secom

pytestmark = pytest.mark.skipif(not secom.data_available(), reason="SECOM data not available (set SECOM_DIR)")


@pytest.fixture(scope="module")
def data():
    return secom.load()


def test_shape_labels_and_time_order(data):
    assert data.X.shape == (1567, 590)
    assert int(data.y.sum()) == 104
    assert data.t.is_monotonic_increasing
    assert str(data.t[0]) == "2008-07-19 11:55:00"


def test_preprocessing_ignores_future_rows(data):
    """Leakage check: corrupting cal/test rows must not change anything fitted on fit rows."""
    from fabpc import secom_analysis as sa
    from fabpc.mspc import PCAMonitor

    prep = sa.prepare(data)
    bad = secom.Secom(X=data.X.copy(), y=data.y.copy(), t=data.t, sensors=data.sensors)
    later = np.concatenate([prep["split"]["cal"], prep["split"]["test"]])
    rng = np.random.default_rng(0)
    bad.X[later] = rng.normal(1e3, 1e2, size=bad.X[later].shape)
    bad.y[later] = 1 - bad.y[later]
    prep2 = sa.prepare(bad)
    assert np.array_equal(prep["keep"], prep2["keep"])
    assert np.array_equal(prep["ref"], prep2["ref"])
    for k in ("median", "mean", "std"):
        assert np.array_equal(getattr(prep["stdz"], k), getattr(prep2["stdz"], k))
    Z1 = prep["stdz"].transform(data.X)
    Z2 = prep2["stdz"].transform(bad.X)
    m1 = PCAMonitor.fit(Z1[prep["ref"]], 0.8)
    m2 = PCAMonitor.fit(Z2[prep2["ref"]], 0.8)
    assert np.array_equal(m1.P, m2.P)
    assert sa.smd_rank(data, prep["split"]["fit"], prep["keep"]) == sa.smd_rank(bad, prep2["split"]["fit"], prep2["keep"])


def test_reference_rows_are_fit_period_passes(data):
    from fabpc import secom_analysis as sa
    prep = sa.prepare(data)
    fit = prep["split"]["fit"]
    assert set(prep["ref"]).issubset(set(fit))
    assert (data.y[prep["ref"]] == 0).all()
    assert prep["ref"].size == int((data.y[fit] == 0).sum())


def test_thresholds_and_limits_do_not_depend_on_test_rows(data):
    from fabpc import secom_analysis as sa

    prep = sa.prepare(data)
    base = sa.run_fdc(data, prep)
    bad = secom.Secom(X=data.X.copy(), y=data.y.copy(), t=data.t, sensors=data.sensors)
    test = prep["split"]["test"]
    bad.X[test] = bad.X[test][::-1] * 3.0 + 7.0
    other = sa.run_fdc(bad, sa.prepare(bad))

    def no_test_eval(d):
        """Drop the test-period exceedance rates (evaluation, allowed to depend on test rows)."""
        if isinstance(d, dict):
            return {k: no_test_eval(v) for k, v in d.items() if k != "test_pass"}
        return d

    assert no_test_eval(base["pca"]) == no_test_eval(other["pca"])
    assert "test_pass" in str(base["pca"])           # the filter removed something real
    assert [m["threshold"] for m in base["methods"]] == [m["threshold"] for m in other["methods"]]
    assert base["lr_c_selection"] == other["lr_c_selection"]
    assert base["gb_selection"] == other["gb_selection"] and base["gb_chosen"] == other["gb_chosen"]


def test_rolling_folds_are_chronological_and_leak_free(data):
    """Each rolling-origin fold: fit < cal < test, contiguous; corrupting everything after
    the fold's calibration block changes nothing that was chosen or fitted."""
    from fabpc import secom_analysis as sa
    n = len(data.y)
    prev_test_end = None
    for a, b, c in sa.ROLLING_FOLDS:
        sp = secom.split_indices(n, a, b, c)
        assert sp["fit"][-1] + 1 == sp["cal"][0] and sp["cal"][-1] + 1 == sp["test"][0]
        if prev_test_end is not None:
            assert sp["test"][0] >= prev_test_end          # test blocks do not overlap
        prev_test_end = sp["test"][-1] + 1
    assert prev_test_end == n
    a, b, c = sa.ROLLING_FOLDS[0]
    sp = secom.split_indices(n, a, b, c)
    base = sa.rolling_fold(data, sp, 0)
    bad = secom.Secom(X=data.X.copy(), y=data.y.copy(), t=data.t, sensors=data.sensors)
    later = np.arange(sp["test"][0], n)
    rng = np.random.default_rng(1)
    bad.X[later] = rng.normal(5e2, 1e2, size=bad.X[later].shape)
    bad.y[later] = 1 - bad.y[later]
    other = sa.rolling_fold(bad, sp, 0)
    for k in ("n_sensors_kept", "pca_components", "lr_c", "lr_c_selection", "gb_chosen", "gb_selection"):
        assert base[k] == other[k], k
    fc = np.concatenate([sp["fit"], sp["cal"]])
    for name, s in base["_scores"].items():
        assert np.array_equal(s[fc], other["_scores"][name][fc]), name
