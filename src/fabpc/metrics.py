"""Detection metrics for fault detection / fail prediction.

Convention: higher score = more suspicious; label 1 = fail, 0 = pass.
"""

from __future__ import annotations

import numpy as np


def threshold_at_far(pass_scores: np.ndarray, far: float) -> float:
    """Threshold such that a fraction ``far`` of the given pass scores lie strictly above it.

    Uses the (1 - far) empirical quantile with numpy's 'higher' method, so the
    realised false-alarm rate on these scores is <= far.
    """
    if not 0.0 < far < 1.0:
        raise ValueError("far must be in (0, 1)")
    s = np.asarray(pass_scores, dtype=float)
    return float(np.quantile(s, 1.0 - far, method="higher"))


def alarm_rates(scores: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float]:
    """Recall on fails and false-alarm rate on passes for 'alarm if score > thr'."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    alarm = scores > thr
    n_fail = int((y == 1).sum())
    n_pass = int((y == 0).sum())
    tp = int((alarm & (y == 1)).sum())
    fp = int((alarm & (y == 0)).sum())
    return {
        "recall": tp / n_fail if n_fail else float("nan"),
        "far": fp / n_pass if n_pass else float("nan"),
        "tp": tp, "fp": fp, "n_fail": n_fail, "n_pass": n_pass,
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
    }


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """PR-AUC as average precision: sum_k (R_k - R_{k-1}) P_k over distinct thresholds
    (same definition as sklearn.metrics.average_precision_score)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    yy = y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    # keep the last index of every tied score block
    last = np.r_[np.where(np.diff(s) != 0)[0], s.size - 1]
    tp = tp[last]
    fp = fp[last]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney statistic (ties count 1/2)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def recall_at_far_oracle(scores: np.ndarray, y: np.ndarray, far: float) -> float:
    """Ranking-only view: recall when the threshold is set on the *same* pass wafers
    (not deployable; used only to separate ranking quality from threshold drift)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    thr = threshold_at_far(scores[y == 0], far)
    return alarm_rates(scores, y, thr)["recall"]


def midranks(scores: np.ndarray) -> np.ndarray:
    """Ranks 1..n of ``scores`` (ascending), tied values share their average rank."""
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s, kind="mergesort")
    sv = s[order]
    ranks = np.empty(s.size, dtype=float)
    starts = np.r_[0, np.where(np.diff(sv) != 0)[0] + 1]
    ends = np.r_[starts[1:], s.size]                 # exclusive
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = 0.5 * (a + b - 1) + 1.0
    return ranks


def ranking_metrics_many(scores: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
    """PR-AUC (average precision, as ``average_precision``) and ROC-AUC (as ``roc_auc``)
    of one fixed score vector against many label vectors (rows of ``Y``) at once.

    Used for label-permutation tests: the ranking of the scores is fixed, only the
    labels move, so ties are handled exactly as in the single-vector functions.
    """
    s = np.asarray(scores, dtype=float)
    Y = np.atleast_2d(np.asarray(Y, dtype=int))
    n = s.size
    n_pos = Y.sum(axis=1)
    if np.any(n_pos != n_pos[0]):
        raise ValueError("every label row must have the same number of positives")
    P = int(n_pos[0])
    N = n - P
    order = np.argsort(-s, kind="mergesort")
    sd = s[order]
    last = np.r_[np.where(np.diff(sd) != 0)[0], n - 1]
    tp = np.cumsum(Y[:, order], axis=1)[:, last].astype(float)
    k = (last + 1).astype(float)
    precision = tp / k
    recall = tp / P
    prev = np.concatenate([np.zeros((Y.shape[0], 1)), recall[:, :-1]], axis=1)
    ap = np.sum((recall - prev) * precision, axis=1)
    r = midranks(s)
    roc = (Y @ r - P * (P + 1) / 2.0) / (P * N)
    return {"pr_auc": ap, "roc_auc": roc}


def permutation_test(scores: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 0) -> dict:
    """One-sided label-permutation test of PR-AUC and ROC-AUC ("is the ranking better
    than chance?"). p = (1 + #{null >= observed}) / (1 + n_perm).

    Also returns the mean and 95th percentile of each null distribution: these are
    the no-skill reference for *this* score vector (ties included).
    """
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    Yp = np.empty((n_perm, y.size), dtype=np.int8)
    for i in range(n_perm):
        Yp[i] = rng.permutation(y)
    obs = ranking_metrics_many(scores, y[None, :])
    null = ranking_metrics_many(scores, Yp)
    out = {"n_perm": n_perm, "seed": seed}
    for k in ("pr_auc", "roc_auc"):
        o = float(obs[k][0])
        v = null[k]
        out[k] = {"observed": o, "p": float((1 + np.sum(v >= o)) / (1 + n_perm)),
                  "null_mean": float(v.mean()), "null_p95": float(np.quantile(v, 0.95))}
    out["_null"] = null
    return out


def random_ap_expected(n: int, n_pos: int) -> float:
    """Exact expected average precision of a uniformly random ranking of ``n`` items
    with ``n_pos`` relevant ones (Bestgen 2015, PBML 103, the R function
    ``RandomAPExact``): (1/R) sum_{i=1..R} sum_{k=i..N-R+i} P(i relevant in top k) (i/k)^2,
    P hypergeometric."""
    from scipy.stats import hypergeom
    R, N = int(n_pos), int(n)
    if not 0 < R <= N:
        raise ValueError("need 0 < n_pos <= n")
    total = 0.0
    for i in range(1, R + 1):
        k = np.arange(i, N - R + i + 1)
        total += float(np.sum(hypergeom.pmf(i, N, R, k) * (i / k) ** 2))
    return total / R


def window_max_null(flags: np.ndarray, w: int, n_sim: int = 20000, seed: int = 0) -> dict:
    """Scan statistic: null distribution of the largest count of flagged items in any
    window of ``w`` consecutive positions, when the same number of flags is placed at
    random positions (permutation of the observed flag sequence)."""
    f = np.asarray(flags, dtype=bool)
    n = f.size
    if not 1 <= w <= n:
        raise ValueError("need 1 <= w <= len(flags)")
    obs = int(_window_counts(f[None, :], w).max())
    rng = np.random.default_rng(seed)
    F = np.empty((n_sim, n), dtype=bool)
    for i in range(n_sim):
        F[i] = rng.permutation(f)
    mx = _window_counts(F, w).max(axis=1)
    return {"observed_max": obs, "n_windows": n - w + 1, "n_flags": int(f.sum()), "n_sim": n_sim, "seed": seed,
            "null_mean_max": float(mx.mean()), "null_p95_max": float(np.quantile(mx, 0.95)),
            "p_value": float((1 + np.sum(mx >= obs)) / (1 + n_sim))}


def _window_counts(F: np.ndarray, w: int) -> np.ndarray:
    """Counts of True in every window of w consecutive columns (rows independent)."""
    c = np.concatenate([np.zeros((F.shape[0], 1), dtype=np.int64), np.cumsum(F, axis=1, dtype=np.int64)], axis=1)
    return c[:, w:] - c[:, :-w]


def stratified_bootstrap(scores: np.ndarray, y: np.ndarray, stat, n_boot: int = 2000,
                         seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Percentile bootstrap CI, resampling fails and passes separately (keeps class counts)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        vals[b] = stat(scores[idx], y[idx])
    lo, hi = np.quantile(vals, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"point": float(stat(scores, y)), "lo": float(lo), "hi": float(hi), "n_boot": n_boot}
