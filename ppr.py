"""
ppr.py – Personalized PageRank re-ranking for retrieval expansion.

Given a query paper q and its top-k baseline results, we:
  1. Build a personalization vector from seed similarities.
  2. Remove q from the graph to prevent leaking its citation neighbors.
  3. Run nx.pagerank on the subgraph.
  4. Optionally blend the PPR score with the baseline similarity.
"""

import numpy as np
import networkx as nx
from baseline import baseline_scores


# ---------------------------------------------------------------------------
# Core PPR
# ---------------------------------------------------------------------------

def ppr_rerank(
    G: nx.Graph,
    S: np.ndarray,
    query: int,
    n_seeds: int = 10,
    damping: float = 0.85,
    min_personalization: float = 1e-6,
) -> np.ndarray:
    """Return PPR scores for every node except *query*.

    Parameters
    ----------
    G : nx.Graph
        Full citation graph.
    S : np.ndarray
        Pre-computed cosine similarity matrix.
    query : int
        Index of the query paper.
    n_seeds : int
        Number of top baseline results used as PPR seeds.
    damping : float
        Probability of following an edge (alpha in nx.pagerank).
    min_personalization : float
        Floor for the personalization value (avoids zeros).

    Returns
    -------
    ppr_scores : np.ndarray, shape (n,)
        PPR score for each node; ppr_scores[query] = 0.
    """
    n = S.shape[0]

    # --- seed selection from baseline ----------------------------------------
    b_scores = baseline_scores(S, query)
    seed_indices = np.argsort(b_scores)[::-1][:n_seeds]

    # --- personalization vector (only on seeds) ------------------------------
    personalization = {}
    for s in seed_indices:
        personalization[int(s)] = float(max(b_scores[s], min_personalization))

    # --- remove query node from graph ----------------------------------------
    nodes_without_q = [n for n in G.nodes() if n != query]
    H = G.subgraph(nodes_without_q).copy()

    # Remove any seed that is not in H (shouldn't happen, but safety)
    personalization = {k: v for k, v in personalization.items() if k in H}
    if not personalization:
        # fallback: uniform
        personalization = {n: 1.0 for n in H.nodes()}

    # --- run PPR -------------------------------------------------------------
    pr = nx.pagerank(H, alpha=damping, personalization=personalization,
                     max_iter=200, tol=1e-8)

    # --- assemble full-length score vector -----------------------------------
    ppr_scores = np.zeros(n, dtype=np.float64)
    for node, score in pr.items():
        ppr_scores[node] = score
    ppr_scores[query] = 0.0

    return ppr_scores


# ---------------------------------------------------------------------------
# Blended score
# ---------------------------------------------------------------------------

def _min_max(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; return zeros if constant."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def blend_scores(
    S: np.ndarray,
    ppr_scores: np.ndarray,
    query: int,
    lam: float = 0.5,
) -> np.ndarray:
    """Return blended ranking scores.

    score = lam * norm(baseline) + (1 - lam) * norm(ppr)

    Parameters
    ----------
    lam : float
        Weight on the text baseline (1 = pure baseline, 0 = pure PPR).
    """
    b = baseline_scores(S, query)
    b_norm = _min_max(b)
    p_norm = _min_max(ppr_scores)
    blended = lam * b_norm + (1 - lam) * p_norm
    blended[query] = -np.inf
    return blended


def ppr_rank(
    G: nx.Graph,
    S: np.ndarray,
    query: int,
    n_seeds: int = 10,
    damping: float = 0.85,
) -> np.ndarray:
    """Convenience: return paper indices sorted by PPR score (descending)."""
    scores = ppr_rerank(G, S, query, n_seeds=n_seeds, damping=damping)
    scores[query] = -np.inf
    return np.argsort(scores)[::-1]


def blend_rank(
    G: nx.Graph,
    S: np.ndarray,
    query: int,
    n_seeds: int = 10,
    damping: float = 0.85,
    lam: float = 0.5,
) -> np.ndarray:
    """Convenience: return paper indices sorted by blended score."""
    ppr_scores = ppr_rerank(G, S, query, n_seeds=n_seeds, damping=damping)
    blended = blend_scores(S, ppr_scores, query, lam=lam)
    return np.argsort(blended)[::-1]

