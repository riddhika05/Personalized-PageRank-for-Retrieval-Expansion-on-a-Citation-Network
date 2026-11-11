"""
metrics.py – Retrieval evaluation metrics.

Metrics:
  - Precision@k
  - Mean Reciprocal Rank (MRR)
  - Paired Wilcoxon signed-rank test
"""

import numpy as np
from scipy.stats import wilcoxon


def precision_at_k(ranked: np.ndarray, relevant: set, k: int) -> float:
    """Fraction of the top-k ranked items that are relevant."""
    top_k = ranked[:k]
    return sum(1 for r in top_k if r in relevant) / k


def reciprocal_rank(ranked: np.ndarray, relevant: set) -> float:
    """1 / (rank of first relevant item).  Returns 0 if none found."""
    for i, r in enumerate(ranked):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_queries(
    rank_fn,
    queries: np.ndarray,
    labels: np.ndarray,
    ks: tuple = (5, 10, 20),
) -> dict:
    """Evaluate a ranking function over a set of queries.

    Parameters
    ----------
    rank_fn : callable(query_idx) -> np.ndarray of ranked indices
    queries : np.ndarray of query indices
    labels : np.ndarray of class labels for every paper
    ks : tuple of k values for P@k

    Returns
    -------
    results : dict
        Keys like 'P@5', 'P@10', 'P@20', 'MRR', each mapping to an array
        of per-query scores.
    """
    results = {f"P@{k}": [] for k in ks}
    results["MRR"] = []

    for q in queries:
        ranked = rank_fn(q)
        relevant = set(np.where(labels == labels[q])[0]) - {q}
        for k in ks:
            results[f"P@{k}"].append(precision_at_k(ranked, relevant, k))
        results["MRR"].append(reciprocal_rank(ranked, relevant))

    return {k: np.array(v) for k, v in results.items()}


def summary_table(results: dict) -> dict:
    """Return {metric: (mean, std)} from evaluate_queries output."""
    return {k: (v.mean(), v.std()) for k, v in results.items()}


def wilcoxon_test(a: np.ndarray, b: np.ndarray):
    """Paired Wilcoxon signed-rank test.

    Returns (statistic, p_value).  If all differences are zero the test
    is not applicable and we return (nan, 1.0).
    """
    diff = a - b
    if np.all(diff == 0):
        return float("nan"), 1.0
    stat, p = wilcoxon(diff)
    return stat, p

