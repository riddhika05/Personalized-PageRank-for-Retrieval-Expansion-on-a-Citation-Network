"""
test_pipeline.py – Unit and regression tests for data, baseline, PPR, and metrics.
"""

import unittest
import numpy as np
import networkx as nx

from data import edge_homophily, rewire_edges
from baseline import build_similarity_matrix, baseline_rank, baseline_scores
from ppr import ppr_rerank, blend_scores, ppr_rank, blend_rank
from metrics import precision_at_k, reciprocal_rank, evaluate_queries, wilcoxon_test


class TestDataAndGraph(unittest.TestCase):
    def setUp(self):
        # Create a toy graph with 4 nodes: 0-1 (same class), 2-3 (same class), 1-2 (cross class)
        self.G = nx.Graph()
        self.G.add_edges_from([(0, 1), (2, 3), (1, 2)])
        self.labels = np.array([0, 0, 1, 1])

    def test_edge_homophily(self):
        # 2 edges join same class, 1 cross class -> homophily = 2/3
        h = edge_homophily(self.G, self.labels)
        self.assertAlmostEqual(h, 2.0 / 3.0, places=4)

    def test_rewire_edges_preserves_count(self):
        rng = np.random.default_rng(42)
        H = rewire_edges(self.G, fraction=0.5, rng=rng)
        self.assertEqual(H.number_of_nodes(), self.G.number_of_nodes())
        self.assertEqual(H.number_of_edges(), self.G.number_of_edges())


class TestBaselineAndPPR(unittest.TestCase):
    def setUp(self):
        # 4 documents, 3 features
        self.features = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.G = nx.cycle_graph(4)

    def test_similarity_matrix(self):
        S = build_similarity_matrix(self.features)
        self.assertEqual(S.shape, (4, 4))
        # Self similarity should be ~1.0
        for i in range(4):
            self.assertAlmostEqual(S[i, i], 1.0, places=4)
        # Symmetry
        np.testing.assert_allclose(S, S.T, atol=1e-5)

    def test_baseline_rank_excludes_query(self):
        S = build_similarity_matrix(self.features)
        ranked = baseline_rank(S, query=0)
        self.assertNotIn(0, ranked[:3])
        self.assertEqual(len(ranked), 4)

    def test_ppr_rank_excludes_query(self):
        S = build_similarity_matrix(self.features)
        ranked = ppr_rank(self.G, S, query=0, n_seeds=2, damping=0.85)
        self.assertNotIn(0, ranked[:3])
        self.assertEqual(len(ranked), 4)

    def test_blend_rank(self):
        S = build_similarity_matrix(self.features)
        ranked = blend_rank(self.G, S, query=1, n_seeds=2, damping=0.85, lam=0.5)
        self.assertNotIn(1, ranked[:3])
        self.assertEqual(len(ranked), 4)


class TestMetrics(unittest.TestCase):
    def test_precision_at_k(self):
        ranked = np.array([1, 2, 3, 4, 5])
        relevant = {1, 3, 5}
        self.assertAlmostEqual(precision_at_k(ranked, relevant, k=3), 2.0 / 3.0)
        self.assertAlmostEqual(precision_at_k(ranked, relevant, k=5), 3.0 / 5.0)

    def test_reciprocal_rank(self):
        ranked = np.array([2, 3, 1, 4])
        relevant = {1}
        self.assertAlmostEqual(reciprocal_rank(ranked, relevant), 1.0 / 3.0)

    def test_wilcoxon_test(self):
        a = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.85, 0.75, 0.65, 0.55])
        b = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.75, 0.65, 0.55, 0.45])
        stat, p = wilcoxon_test(a, b)
        self.assertLess(p, 0.05)


if __name__ == "__main__":
    unittest.main()

