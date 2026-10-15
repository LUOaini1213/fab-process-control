"""Split and preprocessing logic on synthetic arrays (no SECOM download needed)."""

import numpy as np
import pytest

from fabpc import secom


def test_split_is_contiguous_ordered_and_complete():
    sp = secom.split_indices(1567)
    assert sp["fit"][0] == 0 and sp["test"][-1] == 1566
    assert sp["fit"][-1] + 1 == sp["cal"][0] and sp["cal"][-1] + 1 == sp["test"][0]
    assert (sp["fit"].size, sp["cal"].size, sp["test"].size) == (784, 235, 548)
    allidx = np.concatenate([sp["fit"], sp["cal"], sp["test"]])
    assert np.array_equal(allidx, np.arange(1567))
    with pytest.raises(ValueError):
        secom.split_indices(100, 0.7, 0.6)


def test_split_custom_fractions():
    sp = secom.split_indices(40, 0.25, 0.6)
    assert (sp["fit"].size, sp["cal"].size, sp["test"].size) == (10, 14, 16)


def _matrix():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(100, 6)) * np.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0]) + 10.0
    X[:25, 1] = np.nan                    # 25% missing -> dropped
    X[:15, 2] = np.nan                    # 15% missing -> kept
    X[:, 3] = 4.2                         # constant -> dropped
    X[:96, 4] = 0.0                       # 96% one value -> near-constant -> dropped
    X[:94, 5] = 1.0                       # 94% one value -> kept (below 95%)
    return X


def test_select_sensors_rules():
    keep = secom.select_sensors(_matrix())
    assert keep.tolist() == [True, False, True, False, False, True]


def test_select_sensors_thresholds_are_parameters():
    keep = secom.select_sensors(_matrix(), max_missing=0.3, near_constant=0.93)
    assert keep.tolist() == [True, True, True, False, False, False]


def test_most_frequent_share():
    assert secom.most_frequent_share(np.array([1.0, 1.0, 2.0, np.nan])) == pytest.approx(2 / 3)
    assert secom.most_frequent_share(np.array([np.nan, np.nan])) == 1.0


def test_standardizer_uses_reference_rows_only():
    X = _matrix()
    keep = np.array([True, False, True, False, False, True])
    ref = np.r_[0:10, 60:100]            # includes missing values (col 2) and variation (col 5)
    st = secom.Standardizer.fit(X[ref], keep)
    Z = st.transform(X)
    assert not np.isnan(Z).any()
    assert np.isnan(X[ref][:, 2]).sum() == 10
    assert Z[ref].mean(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)
    assert Z[ref].std(axis=0, ddof=1) == pytest.approx(np.ones(3), rel=1e-12)
    # imputation uses the reference median of column 2
    med = np.nanmedian(X[ref, 2])
    assert st.median[1] == pytest.approx(med)
    X2 = X.copy()
    X2[10:60] = X2[10:60] * 50.0 + 3.0   # changing non-reference rows must not change parameters
    st2 = secom.Standardizer.fit(X2[ref], keep)
    assert np.array_equal(st.mean, st2.mean) and np.array_equal(st.std, st2.std)


def test_standardizer_zero_std_guard():
    X = np.column_stack([np.full(10, 3.0), np.arange(10.0)])
    st = secom.Standardizer.fit(X, np.array([True, True]))
    assert st.std[0] == 1.0
    assert np.isfinite(st.transform(X)).all()


def test_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SECOM_DIR", str(tmp_path))
    assert secom.default_dir() == tmp_path
    assert not secom.data_available()
    for name in secom.SHA256:
        (tmp_path / name).write_text("x")
    assert secom.data_available()
    with pytest.raises(ValueError, match="SHA-256"):
        secom.load()


def test_sha256_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    assert secom.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
