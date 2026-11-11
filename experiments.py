"""
experiments.py – All six experiments (E1–E6).

Each experiment function saves a CSV and a matplotlib figure into results/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr

from data import load_cora, edge_homophily, rewire_edges
from baseline import build_similarity_matrix, baseline_rank, baseline_scores
from ppr import ppr_rank, blend_rank, ppr_rerank
from metrics import evaluate_queries, summary_table, wilcoxon_test

RESULTS_DIR = "results"
KS = (5, 10, 20)


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _sample_queries(n_nodes: int, n_queries: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n_nodes, size=n_queries, replace=False)


# ======================================================================
# E1 – Main comparison: baseline vs PPR vs blend
# ======================================================================

def experiment_e1(features, labels, G, S, queries, damping=0.85, n_seeds=10, lam=0.5):
    """Baseline vs PPR vs Blend at default settings."""
    _ensure_dir()
    print("\n=== E1: Main comparison (baseline vs PPR vs blend) ===")

    methods = {
        "Baseline": lambda q: baseline_rank(S, q),
        "PPR":      lambda q: ppr_rank(G, S, q, n_seeds=n_seeds, damping=damping),
        "Blend":    lambda q: blend_rank(G, S, q, n_seeds=n_seeds, damping=damping, lam=lam),
    }

    all_results = {}
    for name, fn in methods.items():
        print(f"  Evaluating {name}...")
        res = evaluate_queries(
            lambda q, _fn=fn: _fn(q),
            tqdm(queries, desc=f"    {name}", leave=False),
            labels,
            ks=KS,
        )
        all_results[name] = res

    # --- build summary table -------------------------------------------------
    rows = []
    for name, res in all_results.items():
        st = summary_table(res)
        row = {"Method": name}
        for metric, (mean, std) in st.items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "e1_main_comparison.csv"), index=False)
    print(df.to_string(index=False))

    # --- Wilcoxon tests (PPR vs Baseline, Blend vs Baseline) -----------------
    for method in ["PPR", "Blend"]:
        for metric in ["P@10", "MRR"]:
            stat, p = wilcoxon_test(
                all_results[method][metric],
                all_results["Baseline"][metric],
            )
            print(f"  Wilcoxon {method} vs Baseline on {metric}: stat={stat:.1f}, p={p:.4e}")

    # --- bar chart -----------------------------------------------------------
    metrics_to_plot = [f"P@{k}" for k in KS] + ["MRR"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (name, res) in enumerate(all_results.items()):
        st = summary_table(res)
        means = [st[m][0] for m in metrics_to_plot]
        stds = [st[m][1] for m in metrics_to_plot]
        ax.bar(x + i * width, means, width, yerr=stds, label=name, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylabel("Score")
    ax.set_title("E1: Baseline vs PPR vs Blend")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "e1_main_comparison.png"), dpi=150)
    plt.close(fig)

    return all_results


# ======================================================================
# E2 – Damping sweep
# ======================================================================

def experiment_e2(features, labels, G, S, queries, n_seeds=10):
    """Sweep damping factor: 0.5, 0.7, 0.85, 0.95."""
    _ensure_dir()
    print("\n=== E2: Damping sweep ===")
    dampings = [0.5, 0.7, 0.85, 0.95]

    rows = []
    for d in dampings:
        print(f"  damping={d}")
        res = evaluate_queries(
            lambda q, _d=d: ppr_rank(G, S, q, n_seeds=n_seeds, damping=_d),
            tqdm(queries, desc=f"    damping={d}", leave=False),
            labels, ks=KS,
        )
        st = summary_table(res)
        row = {"damping": d}
        for metric, (mean, std) in st.items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "e2_damping_sweep.csv"), index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in [f"P@{k}" for k in KS] + ["MRR"]:
        ax.plot(df["damping"], df[f"{metric}_mean"], marker="o", label=metric)
    ax.set_xlabel("Damping factor (α)")
    ax.set_ylabel("Score")
    ax.set_title("E2: PPR Performance vs Damping")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "e2_damping_sweep.png"), dpi=150)
    plt.close(fig)


# ======================================================================
# E3 – Seed count sweep
# ======================================================================

def experiment_e3(features, labels, G, S, queries, damping=0.85):
    """Sweep seed count: 3, 5, 10, 20."""
    _ensure_dir()
    print("\n=== E3: Seed count sweep ===")
    seed_counts = [3, 5, 10, 20]

    rows = []
    for ns in seed_counts:
        print(f"  n_seeds={ns}")
        res = evaluate_queries(
            lambda q, _ns=ns: ppr_rank(G, S, q, n_seeds=_ns, damping=damping),
            tqdm(queries, desc=f"    seeds={ns}", leave=False),
            labels, ks=KS,
        )
        st = summary_table(res)
        row = {"n_seeds": ns}
        for metric, (mean, std) in st.items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "e3_seed_sweep.csv"), index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in [f"P@{k}" for k in KS] + ["MRR"]:
        ax.plot(df["n_seeds"], df[f"{metric}_mean"], marker="s", label=metric)
    ax.set_xlabel("Number of seeds")
    ax.set_ylabel("Score")
    ax.set_title("E3: PPR Performance vs Seed Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "e3_seed_sweep.png"), dpi=150)
    plt.close(fig)


# ======================================================================
# E4 – Hub analysis
# ======================================================================

def experiment_e4(features, labels, G, S, queries, damping=0.85, n_seeds=10):
    """Degree distribution and hub-bias analysis."""
    _ensure_dir()
    print("\n=== E4: Hub analysis ===")

    degrees = np.array([G.degree(i) for i in range(G.number_of_nodes())])

    # --- log-log degree distribution ----------------------------------------
    unique_deg, counts = np.unique(degrees, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(unique_deg, counts, s=12, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("E4: Degree Distribution (log-log)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "e4_degree_distribution.png"), dpi=150)
    plt.close(fig)

    # --- top-10 appearance counts -------------------------------------------
    n = G.number_of_nodes()
    baseline_appear = np.zeros(n, dtype=int)
    ppr_appear = np.zeros(n, dtype=int)

    for q in tqdm(queries, desc="    Hub counting"):
        b_ranked = baseline_rank(S, q)[:10]
        p_ranked = ppr_rank(G, S, q, n_seeds=n_seeds, damping=damping)[:10]
        for idx in b_ranked:
            baseline_appear[idx] += 1
        for idx in p_ranked:
            ppr_appear[idx] += 1

    # correlation
    corr_b, pval_b = spearmanr(degrees, baseline_appear)
    corr_p, pval_p = spearmanr(degrees, ppr_appear)
    print(f"  Spearman(degree, baseline_appear): r={corr_b:.4f}, p={pval_b:.4e}")
    print(f"  Spearman(degree, ppr_appear):      r={corr_p:.4f}, p={pval_p:.4e}")

    # save CSV
    hub_df = pd.DataFrame({
        "node": np.arange(n),
        "degree": degrees,
        "baseline_top10_count": baseline_appear,
        "ppr_top10_count": ppr_appear,
    })
    hub_df.to_csv(os.path.join(RESULTS_DIR, "e4_hub_analysis.csv"), index=False)

    # scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (appear, title, corr) in zip(
        axes,
        [(baseline_appear, "Baseline", corr_b), (ppr_appear, "PPR", corr_p)],
    ):
        ax.scatter(degrees, appear, s=8, alpha=0.4)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Top-10 appearances")
        ax.set_title(f"{title} (Spearman r={corr:.3f})")
    fig.suptitle("E4: Hub Bias – Appearance Count vs Degree")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "e4_hub_scatter.png"), dpi=150)
    plt.close(fig)

    return corr_b, corr_p


# ======================================================================
# E5 – Edge noise robustness
# ======================================================================

def experiment_e5(features, labels, G, S, queries, damping=0.85, n_seeds=10):
    """Rewire edges at 0%, 20%, 40%, 60%; recompute homophily and P@10."""
    _ensure_dir()
    print("\n=== E5: Edge noise robustness ===")
    fractions = [0.0, 0.2, 0.4, 0.6]
    rng = np.random.default_rng(seed=123)

    # baseline P@10 (fixed)
    base_res = evaluate_queries(
        lambda q: baseline_rank(S, q),
        tqdm(queries, desc="    Baseline ref", leave=False),
        labels, ks=(10,),
    )
    baseline_p10 = summary_table(base_res)["P@10"][0]

    rows = []
    for frac in fractions:
        print(f"  rewire fraction={frac}")
        H = rewire_edges(G, frac, rng) if frac > 0 else G
        h = edge_homophily(H, labels)
        res = evaluate_queries(
            lambda q, _H=H: ppr_rank(_H, S, q, n_seeds=n_seeds, damping=damping),
            tqdm(queries, desc=f"    frac={frac}", leave=False),
            labels, ks=(10,),
        )
        st = summary_table(res)
        rows.append({
            "rewire_frac": frac,
            "homophily": h,
            "PPR_P@10_mean": st["P@10"][0],
            "PPR_P@10_std": st["P@10"][1],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "e5_edge_noise.csv"), index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["homophily"], df["PPR_P@10_mean"], marker="o", label="PPR P@10")
    ax.axhline(baseline_p10, color="red", linestyle="--", label=f"Baseline P@10 = {baseline_p10:.3f}")
    ax.fill_between(
        df["homophily"],
        df["PPR_P@10_mean"] - df["PPR_P@10_std"],
        df["PPR_P@10_mean"] + df["PPR_P@10_std"],
        alpha=0.2,
    )
    ax.set_xlabel("Edge Homophily")
    ax.set_ylabel("P@10")
    ax.set_title("E5: PPR Robustness to Edge Noise")
    ax.legend()
    ax.invert_xaxis()  # higher noise → lower homophily → left side
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "e5_edge_noise.png"), dpi=150)
    plt.close(fig)


# ======================================================================
# E6 – Per-class breakdown (optional)
# ======================================================================

def experiment_e6(features, labels, G, S, queries, damping=0.85, n_seeds=10):
    """Per-class P@10 for baseline vs PPR."""
    _ensure_dir()
    print("\n=== E6: Per-class breakdown ===")

    class_names = {
        0: "Case_Based",
        1: "Genetic_Algorithms",
        2: "Neural_Networks",
        3: "Probabilistic_Methods",
        4: "Reinfortic_Learning",
        5: "Rule_Learning",
        6: "Theory",
    }

    # gather per-query P@10
    base_p10, ppr_p10, query_labels = [], [], []
    for q in tqdm(queries, desc="    Per-class eval"):
        b_ranked = baseline_rank(S, q)
        p_ranked = ppr_rank(G, S, q, n_seeds=n_seeds, damping=damping)
        relevant = set(np.where(labels == labels[q])[0]) - {q}
        base_p10.append(sum(1 for r in b_ranked[:10] if r in relevant) / 10)
        ppr_p10.append(sum(1 for r in p_ranked[:10] if r in relevant) / 10)
        query_labels.append(labels[q])

    base_p10 = np.array(base_p10)
    ppr_p10 = np.array(ppr_p10)
    query_labels = np.array(query_labels)

    rows = []
    for cls in sorted(np.unique(query_labels)):
        mask = query_labels == cls
        rows.append({
            "class": class_names.get(cls, str(cls)),
            "n_queries": mask.sum(),
            "baseline_P@10_mean": base_p10[mask].mean(),
            "PPR_P@10_mean": ppr_p10[mask].mean(),
            "delta": (ppr_p10[mask] - base_p10[mask]).mean(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "e6_per_class.csv"), index=False)
    print(df.to_string(index=False))

    # grouped bar chart
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, df["baseline_P@10_mean"], width, label="Baseline")
    ax.bar(x + width / 2, df["PPR_P@10_mean"], width, label="PPR")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=30, ha="right")
    ax.set_ylabel("P@10")
    ax.set_title("E6: Per-Class P@10 Breakdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "e6_per_class.png"), dpi=150)
    plt.close(fig)

    return df

