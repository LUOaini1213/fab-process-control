import numpy as np
import pytest

from fabpc.mspc import PCAMonitor, empirical_limit


def _data(n=300, p=6, seed=11):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(p, p)) * np.array([3.0, 2.0, 1.2, 0.7, 0.4, 0.2])
    X = rng.normal(size=(n, p)) @ L.T + np.array([5.0, -2.0, 0.5, 1.0, 7.0, -3.0])
    return X


def test_full_rank_t2_equals_mahalanobis_and_spe_zero():
    X = _data()
    mon = PCAMonitor.fit(X, n_components=X.shape[1])
    S = np.cov(X, rowvar=False)
    D = X[:7] - X.mean(axis=0)
    maha = np.einsum("ij,jk,ik->i", D, np.linalg.inv(S), D)
    assert mon.t2(X[:7]) == pytest.approx(maha, rel=1e-8)
    assert np.allclose(mon.spe(X[:7]), 0.0, atol=1e-8)


def test_reduced_model_decomposition():
    X = _data()
    mon = PCAMonitor.fit(X, n_components=2)
    Zc = X - X.mean(axis=0)
    T = mon.scores(X)
    # SPE + squared score norm = squared norm of centred row (orthonormal loadings)
    assert mon.spe(X) == pytest.approx((Zc ** 2).sum(axis=1) - (T ** 2).sum(axis=1), rel=1e-8)
    assert mon.P.T @ mon.P == pytest.approx(np.eye(2), abs=1e-12)
    # score variances on the reference data equal lambda
    assert T.var(axis=0, ddof=1) == pytest.approx(mon.lam, rel=1e-10)
    assert mon.lam[0] > mon.lam[1]


def test_contributions_sum_to_statistics():
    X = _data()
    mon = PCAMonitor.fit(X[:200], n_components=3)
    new = X[200:] * np.array([1.0, 1.5, 1.0, 0.5, 2.0, 1.0])
    assert mon.spe_contrib(new).sum(axis=1) == pytest.approx(mon.spe(new), rel=1e-9)
    assert mon.t2_contrib(new).sum(axis=1) == pytest.approx(mon.t2(new), rel=1e-9)
    assert (mon.spe_contrib(new) >= 0).all()


def test_contribution_points_at_disturbed_sensor():
    X = _data(n=400)
    mon = PCAMonitor.fit(X, n_components=3)
    row = X[:1].copy()
    row[0, 4] += 25.0
    share = mon.spe_contrib(row)[0]
    assert int(np.argmax(share)) == 4


def test_reference_mean_is_removed():
    X = _data()
    mon = PCAMonitor.fit(X, n_components=2)
    assert mon.mu == pytest.approx(X.mean(axis=0))
    assert mon.t2(X.mean(axis=0)[None, :])[0] == pytest.approx(0.0, abs=1e-10)
    shifted = PCAMonitor.fit(X + 100.0, n_components=2)
    assert shifted.t2(X[:5] + 100.0) == pytest.approx(mon.t2(X[:5]), rel=1e-6)


def test_component_selection_by_variance_and_sign_convention():
    X = _data()
    full = PCAMonitor.fit(X, n_components=6)
    ratio = np.cumsum(full.lam) / full.lam.sum()
    for target in (0.5, 0.8, 0.95):
        mon = PCAMonitor.fit(X, var_target=target)
        a = mon.n_components
        assert ratio[a - 1] >= target - 1e-12
        if a > 1:
            assert ratio[a - 2] < target
        assert mon.explained == pytest.approx(ratio[a - 1])
    for k in range(full.P.shape[1]):
        j = int(np.argmax(np.abs(full.P[:, k])))
        assert full.P[j, k] > 0


def test_n_components_is_clamped():
    X = _data()
    assert PCAMonitor.fit(X, n_components=99).n_components == 6
    assert PCAMonitor.fit(X, n_components=0).n_components == 1


def test_empirical_limit():
    v = np.arange(1.0, 101.0)
    assert empirical_limit(v, 0.99) == pytest.approx(np.quantile(v, 0.99))
    assert empirical_limit(v, 0.5) == pytest.approx(50.5)
    with pytest.raises(ValueError):
        empirical_limit(v, 1.0)


def test_t2_phase2_limit_formula():
    from scipy.stats import chi2, f
    from fabpc.mspc import t2_limit_phase2
    A, m, q = 3, 50, 0.99
    assert t2_limit_phase2(A, m, q) == pytest.approx(A * (m + 1) * (m - 1) / (m * (m - A)) * f.ppf(q, A, m - A),
                                                      rel=1e-12)
    # large reference set -> chi-square(A) quantile
    assert t2_limit_phase2(4, 10 ** 7, 0.99) == pytest.approx(chi2.ppf(0.99, 4), rel=1e-4)
    with pytest.raises(ValueError):
        t2_limit_phase2(5, 5, 0.99)


def test_spe_jm_limit_against_monte_carlo():
    from fabpc.mspc import spe_limit_jm
    from scipy.stats import chi2
    # SPE of a Gaussian vector = sum_j lambda_j chi2_1; JM is an approximation to its quantiles,
    # accurate when many residual eigenvalues of similar size are summed (the SECOM case)
    lam = np.linspace(1.0, 0.1, 40)
    rng = np.random.default_rng(4)
    spe = (rng.standard_normal((200000, lam.size)) ** 2 * lam).sum(axis=1)
    for q in (0.95, 0.99):
        assert spe_limit_jm(lam, q) == pytest.approx(np.quantile(spe, q), rel=0.01)
    # equal eigenvalues: SPE / lambda is chi-square with k degrees of freedom
    assert spe_limit_jm(np.full(30, 2.0), 0.99) == pytest.approx(2.0 * chi2.ppf(0.99, 30), rel=0.005)


def test_monitor_keeps_discarded_eigenvalues():
    Z = _data()
    m = PCAMonitor.fit(Z, n_components=2)
    total = np.var(Z - Z.mean(axis=0), axis=0, ddof=1).sum()
    assert m.lam.sum() + m.resid_eig.sum() == pytest.approx(total, rel=1e-10)
    assert m.resid_eig.size == Z.shape[1] - 2 and m.n_ref == Z.shape[0]
