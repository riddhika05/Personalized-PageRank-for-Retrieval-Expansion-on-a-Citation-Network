"""
baseline.py – TF-IDF text baseline for retrieval on Cora.

The Cora features are already binary bag-of-words vectors.  We apply
TfidfTransformer to weight them, L2-normalize, and compute all-pairs
cosine similarity.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


def build_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Return (n, n) cosine-similarity matrix from BoW features.

    Steps:
      1. TfidfTransformer on the binary BoW matrix.
      2. L2-normalize each row.
      3. Cosine similarity = dot product of normalized rows.

    The matrix is cached by callers (see run_all.py).
    """
    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(features)          # sparse (n, d)
    X_norm = normalize(X, norm="l2")           # sparse (n, d)
    S = (X_norm @ X_norm.T).toarray()          # dense (n, n)
    return S.astype(np.float32)


def baseline_rank(S: np.ndarray, query: int) -> np.ndarray:
    """Return paper indices sorted by descending similarity to *query*.

    The query itself is excluded from the ranking.
    """
    scores = S[query].copy()
    scores[query] = -np.inf  # exclude self
    return np.argsort(scores)[::-1]


def baseline_scores(S: np.ndarray, query: int) -> np.ndarray:
    """Return raw cosine-similarity scores for all papers (query masked)."""
    scores = S[query].copy()
    scores[query] = 0.0
    return scores

