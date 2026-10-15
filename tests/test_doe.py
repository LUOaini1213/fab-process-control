import numpy as np
import pytest

from fabpc import doe

# NIST e-Handbook 5.3.3.4.7, Dataplot file 2TO4M1.DAT (2^(4-1), 4 = 123)
NIST_2_4_1 = np.array([
    [-1, -1, -1, -1],
    [+1, -1, -1, +1],
    [-1, +1, -1, +1],
    [+1, +1, -1, -1],
    [-1, -1, +1, +1],
    [+1, -1, +1, -1],
    [-1, +1, +1, -1],
    [+1, +1, +1, +1],
], dtype=float)


def test_full_factorial_standard_order():
    D = doe.full_factorial(3)
    assert D.shape == (8, 3)
    assert D[:, 0].tolist() == [-1, 1, -1, 1, -1, 1, -1, 1]
    assert D[:, 1].tolist() == [-1, -1, 1, 1, -1, -1, 1, 1]
    assert D[:, 2].tolist() == [-1, -1, -1, -1, 1, 1, 1, 1]
    assert len({tuple(r) for r in doe.full_factorial(4)}) == 16
    with pytest.raises(ValueError):
        doe.full_factorial(0)


def test_fractional_matches_nist_listing():
    D = doe.fractional_2_4_1()
    assert np.array_equal(D, NIST_2_4_1)
    # defining relation I = 1234 and aliasing 12 = 34
    assert (np.prod(D, axis=1) == 1).all()
    assert np.array_equal(D[:, 0] * D[:, 1], D[:, 2] * D[:, 3])


def test_term_columns_names_and_order():
    names, C = doe.term_columns(doe.full_factorial(3))
    assert names == ["1", "2", "3", "12", "13", "23", "123"]
    names2, _ = doe.term_columns(doe.full_factorial(3), max_order=2)
    assert names2 == ["1", "2", "3", "12", "13", "23"]
    D = doe.full_factorial(3)
    assert np.array_equal(C[:, 6], D[:, 0] * D[:, 1] * D[:, 2])


def test_effects_are_twice_the_coefficients():
    D = doe.full_factorial(3)
    y = 7.0 + 1.5 * D[:, 0] - 0.4 * D[:, 1] + 0.25 * D[:, 2] + 0.6 * D[:, 0] * D[:, 2] - 0.15 * np.prod(D, axis=1)
    e = doe.effects(D, y)
    assert e["1"] == pytest.approx(3.0)
    assert e["2"] == pytest.approx(-0.8)
    assert e["3"] == pytest.approx(0.5)
    assert e["13"] == pytest.approx(1.2)
    assert e["12"] == pytest.approx(0.0, abs=1e-12)
    assert e["123"] == pytest.approx(-0.3)


def test_factorial_anova_contrast_ss_and_ftest():
    rng = np.random.default_rng(4)
    D = np.vstack([doe.full_factorial(2)] * 3)
    y = 3.0 + 0.8 * D[:, 0] - 0.1 * D[:, 1] + 0.3 * D[:, 0] * D[:, 1] + rng.normal(0, 0.2, len(D))
    an = doe.factorial_anova(D, y)
    N = len(y)
    for name, col in (("1", D[:, 0]), ("2", D[:, 1]), ("12", D[:, 0] * D[:, 1])):
        contrast = float(col @ y)
        assert an[name]["ss"] == pytest.approx(contrast ** 2 / N, rel=1e-9)
        assert an[name]["coef"] == pytest.approx(contrast / N, rel=1e-9)
    resid = y - np.column_stack([np.ones(N), D, D[:, 0] * D[:, 1]]) @ np.linalg.lstsq(
        np.column_stack([np.ones(N), D, D[:, 0] * D[:, 1]]), y, rcond=None)[0]
    assert an["_error"]["df"] == N - 4
    assert an["_error"]["ms"] == pytest.approx(resid @ resid / (N - 4))
    assert an["1"]["f"] == pytest.approx(an["1"]["ss"] / an["_error"]["ms"])
    assert an["1"]["p"] < 1e-6 < an["2"]["p"]


def test_curvature_indicator_detects_centre_shift():
    F = np.vstack([doe.full_factorial(2)] * 2)
    X = np.vstack([F, np.zeros((4, 2))])
    centre = np.r_[np.zeros(len(F)), np.ones(4)]
    rng = np.random.default_rng(8)
    y = 5.0 + 0.5 * X[:, 0] + rng.normal(0, 0.05, len(X)) - 1.0 * centre
    an = doe.factorial_anova(X, y, extra={"curvature": centre})
    assert an["curvature"]["coef"] == pytest.approx(-1.0, abs=0.15)
    assert an["curvature"]["p"] < 1e-4
    y0 = 5.0 + 0.5 * X[:, 0] + rng.normal(0, 0.05, len(X))
    assert doe.factorial_anova(X, y0, extra={"curvature": centre})["curvature"]["p"] > 0.01


def test_fit_ols_against_numpy():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.3, 30)
    r = doe.fit_ols(X, y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    assert r["beta"] == pytest.approx(beta)
    assert r["df"] == 27
    s2 = float(((y - X @ beta) ** 2).sum() / 27)
    assert r["se"] == pytest.approx(np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))))
    with pytest.raises(ValueError):
        doe.fit_ols(X[:3], y[:3])


def test_ccd_and_rotatable_alpha():
    assert doe.rotatable_alpha(4) == pytest.approx(1.414, abs=1e-3)
    assert doe.rotatable_alpha(8) == pytest.approx(1.682, abs=1e-3)
    assert doe.rotatable_alpha(16) == pytest.approx(2.0)
    X = doe.ccd(3, n_center=5)
    assert X.shape == (8 + 6 + 5, 3)
    axial = X[8:14]
    assert np.allclose(np.abs(axial).sum(axis=1), 8 ** 0.25)
    assert (X[14:] == 0).all()
    assert doe.ccd(2, 1, alpha=1.0).max() == 1.0


def test_quadratic_fit_recovers_exact_surface():
    X = doe.ccd(3, n_center=3)
    beta_true = np.array([2.0, 0.5, -0.3, 0.8, 1.2, 0.7, 0.4, 0.25, -0.1, 0.05])
    names, M = doe.quad_model_matrix(X)
    assert names == ["1", "x1", "x2", "x3", "x1^2", "x2^2", "x3^2", "x1x2", "x1x3", "x2x3"]
    y = M @ beta_true
    beta = np.linalg.lstsq(M, y, rcond=None)[0]
    assert beta == pytest.approx(beta_true, abs=1e-10)
    b0, b, B = doe.quad_to_bB(beta, 3)
    assert b0 == pytest.approx(2.0)
    assert B[0, 1] == pytest.approx(0.125) and B[1, 0] == pytest.approx(0.125)
    assert B[0, 2] == pytest.approx(-0.05) and B[1, 2] == pytest.approx(0.025)
    sp = doe.stationary_point(b, B)
    grad = b + 2 * B @ sp["x"]
    assert grad == pytest.approx(np.zeros(3), abs=1e-10)
    assert sp["nature"] == "minimum"
    assert doe.quad_predict(beta, X[:2]) == pytest.approx(y[:2])


def test_stationary_point_nist_example():
    # NIST 5.5.3.1.4 Technical Appendix 5C: Y = 72.0 - 11.78 x1 + 0.74 x2 - 7.25 x1^2 - 7.55 x2^2 - 4.85 x1 x2
    beta = np.array([72.0, -11.78, 0.74, -7.25, -7.55, -4.85])
    _, b, B = doe.quad_to_bB(beta, 2)
    sp = doe.stationary_point(b, B)
    assert sp["x"] == pytest.approx([-0.9285, 0.3472], abs=1e-3)
    assert sp["nature"] == "maximum"


def test_saddle_detection():
    sp = doe.stationary_point(np.array([1.0, -1.0]), np.array([[2.0, 0.0], [0.0, -0.5]]))
    assert sp["nature"] == "saddle"
    assert sp["x"] == pytest.approx([-0.25, -1.0])
