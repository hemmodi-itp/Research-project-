"""Hypothesis testing (Sec. 3.4): DeLong test for correlated AUROCs, Holm correction, patient-level bootstrap."""

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------- DeLong
def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_auc_cov(y, scores):
    """Fast DeLong (Sun & Xu, 2014). `scores` is (k, n) for k models on the same cases.

    Returns (aucs (k,), covariance (k, k)).
    """
    y = np.asarray(y).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y, scores_a, scores_b, alternative="greater"):
    """Test AUROC(a) vs AUROC(b) on the same cases. alternative: "greater" (a > b) or "two-sided"."""
    aucs, cov = delong_auc_cov(y, np.vstack([scores_a, scores_b]))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_a": aucs[0], "auc_b": aucs[1], "diff": diff, "z": float("nan"), "p": float("nan")}
    z = diff / np.sqrt(var)
    p = stats.norm.sf(z) if alternative == "greater" else 2 * stats.norm.sf(abs(z))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(diff), "z": float(z), "p": float(p)}


def holm(pvalues):
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# ----------------------------------------------------------------------------- bootstrap
class PatientBootstrap:
    """Resample patients (clusters) with replacement; every stay of a drawn patient comes along."""

    def __init__(self, subject_ids, n_boot=2000, seed=0):
        subject_ids = np.asarray(subject_ids)
        _, inverse = np.unique(subject_ids, return_inverse=True)
        self.groups = [np.where(inverse == g)[0] for g in range(inverse.max() + 1)]
        self.n_boot = n_boot
        self.seed = seed
        self._samples = None

    def samples(self):
        """The same resamples are reused for every statistic, so differences between models are paired."""
        if self._samples is None:
            rng = np.random.default_rng(self.seed)
            n = len(self.groups)
            self._samples = [np.concatenate([self.groups[g] for g in rng.integers(0, n, n)])
                             for _ in range(self.n_boot)]
        return self._samples

    def ci(self, statistic, alpha=0.05):
        """Percentile CI of statistic(idx) -> float. Resamples where it is undefined (NaN) are skipped."""
        values = np.array([statistic(idx) for idx in self.samples()], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return float("nan"), float("nan")
        return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))
