"""PCA-based multivariate monitoring: Hotelling T^2 in the score space, SPE (Q)
in the residual space, empirical control limits and variable contributions.

T^2 = sum_a t_a^2 / lambda_a and SPE = sum_j e_j^2 follow the PCA monitoring
convention (e.g. the R package SCOUTer, function ``pcame``; see docs/sources.md).
Alarm limits are *empirical* quantiles on held-out in-control data; the
distribution-based limits (Phase II F limit for T^2, Jackson-Mudholkar limit
for SPE) are computed alongside for comparison (``t2_limit_phase2``,
``spe_limit_jm``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAMonitor:
    mu: np.ndarray       # (p,) reference mean (about 0 after standardization)
    P: np.ndarray        # (p, A) loadings
    lam: np.ndarray      # (A,) score variances on the reference data
    n_components: int
    explained: float     # cumulative explained variance ratio of the A PCs
    resid_eig: np.ndarray = None  # variances of the discarded PCs (for the SPE limit)
    n_ref: int = 0       # number of reference rows the model was fitted on

    @classmethod
    def fit(cls, Z: np.ndarray, var_target: float = 0.80, n_components: int | None = None) -> "PCAMonitor":
        """Fit on standardized reference data Z (rows = in-control observations).

        The reference mean is stored and subtracted from every projected row.
        The number of PCs is the smallest A reaching ``var_target`` cumulative
        explained variance unless ``n_components`` is given.
        """
        Z = np.asarray(Z, dtype=float)
        n = Z.shape[0]
        mu = Z.mean(axis=0)
        Zc = Z - mu
        _, s, vt = np.linalg.svd(Zc, full_matrices=False)
        var = s ** 2 / (n - 1)
        ratio = np.cumsum(var) / var.sum()
        if n_components is None:
            a = int(np.searchsorted(ratio, var_target) + 1)
        else:
            a = int(n_components)
        a = max(1, min(a, len(s)))
        P = vt[:a].T.copy()
        # deterministic sign: the largest-magnitude loading of each PC is positive
        for k in range(a):
            j = int(np.argmax(np.abs(P[:, k])))
            if P[j, k] < 0:
                P[:, k] = -P[:, k]
        return cls(mu=mu, P=P, lam=var[:a].copy(), n_components=a, explained=float(ratio[a - 1]),
                   resid_eig=var[a:].copy(), n_ref=n)

    def _center(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=float) - self.mu

    def scores(self, Z: np.ndarray) -> np.ndarray:
        return self._center(Z) @ self.P

    def residuals(self, Z: np.ndarray) -> np.ndarray:
        Zc = self._center(Z)
        return Zc - (Zc @ self.P) @ self.P.T

    def t2(self, Z: np.ndarray) -> np.ndarray:
        T = self.scores(Z)
        return np.sum(T ** 2 / self.lam, axis=1)

    def spe(self, Z: np.ndarray) -> np.ndarray:
        E = self.residuals(Z)
        return np.sum(E ** 2, axis=1)

    def spe_contrib(self, Z: np.ndarray) -> np.ndarray:
        """Per-variable SPE contributions e_j^2 (rows sum to SPE)."""
        return self.residuals(Z) ** 2

    def t2_contrib(self, Z: np.ndarray) -> np.ndarray:
        """Per-variable T^2 contributions c_j = z_j * sum_a (t_a / lambda_a) p_ja.

        (z = row centred on the reference mean.) Exact decomposition: rows sum to
        T^2 because T^2 = sum_a t_a^2/lambda_a = sum_a (t_a/lambda_a) sum_j p_ja z_j.
        Individual terms can be negative. This is the partial decomposition
        contribution z_j (M z)_j with M = P diag(1/lambda) P' (Alcala & Qin 2011,
        J. Process Control 21(3), 322-330; see docs/sources.md, M2).
        """
        Zc = self._center(Z)
        T = Zc @ self.P
        w = (T / self.lam) @ self.P.T
        return Zc * w


def t2_limit_phase2(n_components: int, m: int, q: float) -> float:
    """Phase II upper limit for T^2 of a *new* individual observation when mean and
    covariance were estimated from m reference observations (NIST/SEMATECH
    e-Handbook 6.5.4.3.4, applied to the A retained scores):
    UCL = A (m + 1)(m - 1) / (m^2 - m A) * F_q(A, m - A). Assumes multivariate normality."""
    from scipy.stats import f as f_dist
    A, m = int(n_components), int(m)
    if not 0 < A < m:
        raise ValueError("need 0 < n_components < m")
    return float(A * (m + 1) * (m - 1) / (m * m - m * A) * f_dist.ppf(q, A, m - A))


def spe_limit_jm(resid_eig: np.ndarray, q: float) -> float:
    """Jackson-Mudholkar upper limit for SPE from the variances of the discarded PCs
    (as implemented in the R package SCOUTer, ``pcamb_classic``):
    theta_i = sum lambda_j^i, h0 = 1 - 2 theta1 theta3 / (3 theta2^2),
    Q = theta1 [z_q sqrt(2 theta2 h0^2) / theta1 + 1 + theta2 h0 (h0 - 1) / theta1^2]^(1/h0)."""
    from scipy.stats import norm
    lam = np.asarray(resid_eig, dtype=float)
    t1, t2, t3 = float(lam.sum()), float((lam ** 2).sum()), float((lam ** 3).sum())
    h0 = 1.0 - 2.0 * t1 * t3 / (3.0 * t2 ** 2)
    z = float(norm.ppf(q))
    return float(t1 * (z * np.sqrt(2.0 * t2 * h0 ** 2) / t1 + 1.0 + t2 * h0 * (h0 - 1.0) / t1 ** 2) ** (1.0 / h0))


def empirical_limit(stat_ref: np.ndarray, q: float) -> float:
    """Upper control limit = q-quantile of the statistic on in-control reference rows
    (numpy 'linear' interpolation)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    return float(np.quantile(np.asarray(stat_ref, dtype=float), q))
