# Personalized PageRank for Retrieval Expansion on a Citation Network: When Graph Structure Helps and When It Hurts

**Applied Graph Theory Term Paper Project**

This repository contains the complete, reproducible Python codebase and experimental suite investigating whether re-ranking text-retrieval results with **Personalized PageRank (PPR)** over a citation graph improves retrieval performance over a flat text-only baseline, and analyzing the conditions under which graph structure helps versus when it hurts.

---

## Key Findings Overview

1. **Precision Expansion:** PPR significantly improves Precision@10 from **54.5% to 58.7%** ($p = 5.54 \times 10^{-13}$) and Precision@20 from **49.4% to 54.4%** across 500 evaluation queries.
2. **First-Hit Accuracy (MRR):** Flat TF-IDF slightly outperforms pure PPR on MRR (**0.7815 vs 0.7663**, $p = 0.146$), demonstrating that lexical similarity is superior for the top-1 hit, whereas graph walks excel at deep retrieval expansion.
3. **Hybrid Blending:** A simple linear blend ($\lambda = 0.5$) achieves the highest overall P@5 (**62.1%**) and P@20 (**55.4%**), outperforming both pure text and pure graph ranking.
4. **Seed Sensitivity:** Counter-intuitively, fewer seeds ($n_{\text{seeds}} = 3$) outperform larger seed pools ($n_{\text{seeds}} = 20$), as noisy low-ranked text seeds pollute the PPR personalization vector.
5. **Homophily Boundary:** Graph re-ranking only helps when citation edge homophily exceeds **~60–65%**. Below this threshold (simulated via edge rewiring), PPR actively harms retrieval compared to flat text search.
6. **Hub Bias:** PPR results display a ~3× higher correlation with node degree ($r = 0.1950, p = 1.26 \times 10^{-24}$) than baseline text search ($r = 0.0666$), confirming that random walks are biased toward prominent citation hubs.

*For complete experimental data, statistical tables, and in-depth discussions, see [results_summary.md](results_summary.md).*

---

## Dataset & Graph Formulation

- **Dataset:** [Cora](https://relational.fit.cvut.cz/dataset/CORA) accessed via `torch_geometric.datasets.Planetoid(root="data", name="Cora")`.
- **Nodes:** 2,708 scientific publications across 7 subject classes.
- **Edges:** 5,278 undirected citation links.
- **Node Features:** 1,433-dimensional binary bag-of-words vectors.
- **Edge Homophily:** **0.8100** (81.0% of citations connect papers within the same subject class).
- **Leakage Prevention:** For every query paper $q$, node $q$ is **strictly excised from the citation graph** prior to computing PPR:
  $$\mathcal{G}_{-q} = \mathcal{G} \setminus \{q\}$$
  This guarantees the random walk cannot trivially access $q$'s direct citation neighbors.

---

## Methods

### 1. Flat Text Baseline
- Binary bag-of-words weighted with `sklearn.feature_extraction.text.TfidfTransformer`.
- $L_2$-normalized rows with all-pairs cosine similarity:
  $$S = \tilde{X} \tilde{X}^T \in \mathbb{R}^{N \times N}$$
- Ranks candidate papers $v \neq q$ in descending order of $S_{q, v}$.

### 2. Personalized PageRank (PPR)
- Seeds = top $n_{\text{seeds}}$ baseline results for query $q$ (excluding $q$).
- Personalization vector $\mathbf{p} \in \mathbb{R}^{N-1}$:
  $$p_v = \max(S_{q, v}, \epsilon) \quad \text{for } v \in \text{seeds}(q)$$
  normalized such that $\sum_{v} p_v = 1$.
- Computed via power iteration on the subgraph $\mathcal{G}_{-q}$ with damping $\alpha \in [0, 1]$:
  $$\mathbf{r} = \alpha \mathbf{r} \mathbf{P} + (1 - \alpha) \mathbf{p}$$

### 3. Hybrid Blend
- Min-max normalized combination controlled by parameter $\lambda \in [0, 1]$:
  $$\text{score}(v) = \lambda \cdot \text{norm}(S_{q, v}) + (1 - \lambda) \cdot \text{norm}(r_v)$$

---

## Project Structure

```
├── data.py                  # Cora loader, homophily metric, edge rewiring
├── baseline.py              # TF-IDF cosine similarity matrix & ranking
├── ppr.py                   # Personalized PageRank & hybrid score blending
├── metrics.py               # P@k, MRR, paired Wilcoxon signed-rank test
├── experiments.py           # Experiment suites E1–E6 with plotting & CSV export
├── run_all.py               # Unified CLI runner (--quick or full 500-query run)
├── test_pipeline.py         # Unit test suite verifying components
├── requirements.txt         # Pinned Python dependencies
├── results_summary.md       # Comprehensive results report & academic analysis
├── results/                 # Generated plots and raw CSV data
│   ├── e1_main_comparison.csv
│   ├── e1_main_comparison.png
│   ├── e2_damping_sweep.csv
│   ├── e2_damping_sweep.png
│   ├── e3_seed_sweep.csv
│   ├── e3_seed_sweep.png
│   ├── e4_degree_distribution.png
│   ├── e4_hub_analysis.csv
│   ├── e4_hub_scatter.png
│   ├── e5_edge_noise.csv
│   ├── e5_edge_noise.png
│   ├── e6_per_class.csv
│   └── e6_per_class.png
└── README.md
```

---

## Setup & Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PyTorch and PyTorch Geometric

### Installation
```bash
pip install -r requirements.txt
```

### Run Unit Tests
```bash
python -m unittest test_pipeline.py
```

### Fast Debug Mode (50 queries, ~1.5 minutes)
```bash
python run_all.py --quick
```

### Full Experimental Run (500 queries, ~30 minutes)
```bash
python run_all.py
```

All figures and CSV files are automatically exported into the `results/` directory.

---

## Experimental Suite

| ID | Experiment Description | Core Variable | Primary Output Files |
|:---|:---|:---|:---|
| **E1** | Main Comparison: Baseline vs PPR vs Blend | Method comparison | `e1_main_comparison.{csv,png}` |
| **E2** | Damping Factor Sweep | $\alpha \in \{0.5, 0.7, 0.85, 0.95\}$ | `e2_damping_sweep.{csv,png}` |
| **E3** | Seed Count Sweep | $n_{\text{seeds}} \in \{3, 5, 10, 20\}$ | `e3_seed_sweep.{csv,png}` |
| **E4** | Hub Analysis & Degree Distribution | Node degree vs Top-10 appearances | `e4_degree_distribution.png`, `e4_hub_scatter.png` |
| **E5** | Edge Noise Robustness | Rewiring $0\%, 20\%, 40\%, 60\%$ | `e5_edge_noise.{csv,png}` |
| **E6** | Per-Class Performance Breakdown | 7 Cora subject classes | `e6_per_class.{csv,png}` |

---

## Summary of Results (500 Queries)

| Experiment | Metric | Baseline | PPR | Blend ($\lambda=0.5$) | Wilcoxon $p$-value |
|:---|:---|:---:|:---:|:---:|:---:|
| **E1: Main** | **P@5** | 0.5916 | 0.6172 | **0.6212** | - |
| | **P@10** | 0.5446 | **0.5874** | 0.5734 | $p = 5.54 \times 10^{-13}$ |
| | **P@20** | 0.4938 | 0.5435 | **0.5540** | - |
| | **MRR** | **0.7815** | 0.7663 | 0.7776 | $p = 0.1461$ |
| **E2: Damping** | **Optimal P@10** | - | **0.5874** ($\alpha=0.85$) | - | - |
| **E3: Seeds** | **Optimal P@10** | - | **0.6004** ($n_{\text{seeds}}=3$) | - | - |
| **E4: Hubs** | **Spearman $r$** | 0.0666 | **0.1950** ($p=1.26\times 10^{-24}$) | - | - |
| **E5: Noise** | **Homophily Breakdown** | 0.5446 | **0.5364** (at $h=0.556$) | - | Threshold $\approx 0.60$ |

---

## Citation & Academic Context

Developed for an applied graph theory term paper on retrieval expansion and random walks on homophilous network structures.
