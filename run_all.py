#!/usr/bin/env python
"""
run_all.py – Entry point for all experiments.

Usage
-----
  python run_all.py              # full run (500 queries)
  python run_all.py --quick      # quick debug (50 queries)
"""

import argparse
import time
import numpy as np

from data import load_cora, edge_homophily
from baseline import build_similarity_matrix
from experiments import (
    _sample_queries,
    experiment_e1,
    experiment_e2,
    experiment_e3,
    experiment_e4,
    experiment_e5,
    experiment_e6,
)


def main():
    parser = argparse.ArgumentParser(
        description="PPR retrieval expansion experiments on Cora."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run on 50 queries for fast debugging.",
    )
    args = parser.parse_args()

    n_queries = 50 if args.quick else 500
    print(f"{'='*60}")
    print(f"  PPR Retrieval Expansion on Cora")
    print(f"  Queries: {n_queries}  {'(quick mode)' if args.quick else ''}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.time()
    print("\n[1/3] Loading Cora dataset...")
    features, labels, G = load_cora()
    print(f"  Nodes : {G.number_of_nodes()}")
    print(f"  Edges : {G.number_of_edges()}")
    h = edge_homophily(G, labels)
    print(f"  Edge homophily: {h:.4f}")

    # ------------------------------------------------------------------
    # 2. Build similarity matrix (cached in memory)
    # ------------------------------------------------------------------
    print("\n[2/3] Building TF-IDF similarity matrix...")
    S = build_similarity_matrix(features)
    print(f"  S shape: {S.shape}  (dtype={S.dtype})")

    # ------------------------------------------------------------------
    # 3. Sample queries
    # ------------------------------------------------------------------
    queries = _sample_queries(G.number_of_nodes(), n_queries, seed=42)
    print(f"\n[3/3] Sampled {len(queries)} query papers.")
    print(f"  Setup time: {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    experiment_e1(features, labels, G, S, queries)
    experiment_e2(features, labels, G, S, queries)
    experiment_e3(features, labels, G, S, queries)
    experiment_e4(features, labels, G, S, queries)
    experiment_e5(features, labels, G, S, queries)
    experiment_e6(features, labels, G, S, queries)

    print(f"\n{'='*60}")
    print(f"  All experiments complete.  Results saved to results/")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

