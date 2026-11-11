"""
data.py – Load the Cora citation dataset and build graph utilities.

Provides:
  - load_cora(): returns features, labels, and an undirected networkx Graph.
  - edge_homophily(): fraction of edges joining same-class nodes.
  - rewire_edges(): randomly rewire a fraction of edges (preserving count).
"""

import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid


def load_cora(root: str = "data"):
    """Load Cora via PyG and return (features, labels, G).

    Returns
    -------
    features : np.ndarray, shape (2708, 1433)
        Binary bag-of-words features.
    labels : np.ndarray, shape (2708,)
        Integer class labels (0–6).
    G : nx.Graph
        Undirected citation graph.
    """
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]

    features = data.x.numpy()          # (2708, 1433)
    labels = data.y.numpy()            # (2708,)

    edge_index = data.edge_index.numpy()  # (2, num_directed_edges)
    G = nx.Graph()
    G.add_nodes_from(range(features.shape[0]))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)             # duplicates are ignored for Graph

    return features, labels, G


def edge_homophily(G: nx.Graph, labels: np.ndarray) -> float:
    """Fraction of edges whose endpoints share the same class label."""
    same = sum(1 for u, v in G.edges() if labels[u] == labels[v])
    return same / G.number_of_edges()


def rewire_edges(G: nx.Graph, fraction: float, rng: np.random.Generator) -> nx.Graph:
    """Randomly rewire *fraction* of edges, preserving edge count.

    For each selected edge (u, v) we remove it and add a new edge (u, w)
    where w is chosen uniformly at random (avoiding self-loops and existing
    edges).  This is a simplified configuration-model rewire.

    Parameters
    ----------
    G : nx.Graph
        Original graph (not modified in-place).
    fraction : float
        Fraction of edges to rewire, in [0, 1].
    rng : np.random.Generator
        Reproducible random generator.

    Returns
    -------
    H : nx.Graph
        Rewired copy.
    """
    H = G.copy()
    edges = list(H.edges())
    n = H.number_of_nodes()
    num_rewire = int(len(edges) * fraction)
    indices = rng.choice(len(edges), size=num_rewire, replace=False)

    for idx in indices:
        u, v = edges[idx]
        if not H.has_edge(u, v):
            continue  # already removed in a prior step
        H.remove_edge(u, v)
        # add a random replacement edge from u
        for _ in range(100):  # retry limit
            w = rng.integers(0, n)
            if w != u and not H.has_edge(u, w):
                H.add_edge(u, w)
                break

    return H


if __name__ == "__main__":
    features, labels, G = load_cora()
    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape  : {labels.shape}")
    print(f"Classes       : {np.unique(labels)}")
    h = edge_homophily(G, labels)
    print(f"Edge homophily: {h:.4f}")

