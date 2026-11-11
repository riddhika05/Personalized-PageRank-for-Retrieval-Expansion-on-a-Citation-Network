# Results Summary: Personalized PageRank for Retrieval Expansion on Cora

**Term Paper:** *Personalized PageRank for Retrieval Expansion on a Citation Network: When Graph Structure Helps and When It Hurts*  
**Evaluation Scope:** 500 randomly sampled query papers (seed = 42), query node $q$ strictly removed from citation graph during PPR execution to avoid citation leakage. Relevance defined by ground-truth subject class agreement (7 classes total).

---

## 1. Graph and Dataset Summary

| Property | Value | Notes |
|:---|:---|:---|
| **Dataset** | Cora (`torch_geometric.datasets.Planetoid`) | 7 scientific subject classes |
| **Nodes ($N$)** | 2,708 | Scientific publications |
| **Edges ($M$)** | 5,278 | Undirected citation links |
| **Feature Dimension ($D$)** | 1,433 | Binary bag-of-words indicators |
| **Edge Homophily ($h$)** | **0.8100** (81.0%) | Fraction of edges connecting papers with identical class labels |
| **Evaluation Queries** | 500 papers | Fixed seed (42), query vector = paper's BoW |

---

## 2. E1: Main Method Comparison

*Default settings:* Damping $\alpha = 0.85$, $n_{\text{seeds}} = 10$, Blend $\lambda = 0.5$.

| Method | P@5 (mean ± std) | P@10 (mean ± std) | P@20 (mean ± std) | MRR (mean ± std) |
|:---|:---:|:---:|:---:|:---:|
| **Baseline (TF-IDF)** | 0.5916 ± 0.3015 | 0.5446 ± 0.2667 | 0.4938 ± 0.2318 | **0.7815 ± 0.3257** |
| **PPR (Pure Graph)** | 0.6172 ± 0.3389 | **0.5874 ± 0.2964** | 0.5435 ± 0.2753 | 0.7663 ± 0.3488 |
| **Blend ($\lambda = 0.5$)** | **0.6212 ± 0.3184** | 0.5734 ± 0.2783 | **0.5540 ± 0.2608** | 0.7776 ± 0.3293 |

### Statistical Significance (Paired Wilcoxon Signed-Rank Test, $N=500$)
- **PPR vs Baseline on P@10:** $W = 11007.0,\; p = 5.54 \times 10^{-13}$ (**statistically significant improvement**, $p < 0.001$)
- **PPR vs Baseline on MRR:** $W = 8233.5,\; p = 0.1461$ (*not significant*; baseline preserves top-1 rank slightly better)
- **Blend vs Baseline on P@10:** $W = 1378.5,\; p = 8.48 \times 10^{-18}$ (**statistically significant improvement**, $p < 0.001$)
- **Blend vs Baseline on MRR:** $W = 6306.5,\; p = 0.4500$ (*not significant*)

---

## 3. E2: Damping Factor Sweep

*Fixed:* $n_{\text{seeds}} = 10$.

| Damping ($\alpha$) | P@5 (mean ± std) | P@10 (mean ± std) | P@20 (mean ± std) | MRR (mean ± std) |
|:---:|:---:|:---:|:---:|:---:|
| **0.50** | 0.6072 ± 0.3121 | 0.5548 ± 0.2679 | 0.5305 ± 0.2640 | **0.7813 ± 0.3285** |
| **0.70** | 0.6172 ± 0.3283 | 0.5708 ± 0.2837 | 0.5376 ± 0.2687 | 0.7802 ± 0.3361 |
| **0.85** | **0.6172 ± 0.3389** | **0.5874 ± 0.2964** | 0.5435 ± 0.2753 | 0.7663 ± 0.3488 |
| **0.95** | 0.6048 ± 0.3571 | 0.5828 ± 0.3175 | **0.5549 ± 0.2881** | 0.7383 ± 0.3611 |

---

## 4. E3: Seed Count Sweep

*Fixed:* $\alpha = 0.85$.

| $n_{\text{seeds}}$ | P@5 (mean ± std) | P@10 (mean ± std) | P@20 (mean ± std) | MRR (mean ± std) |
|:---:|:---:|:---:|:---:|:---:|
| **3** | **0.6240 ± 0.3499** | **0.6004 ± 0.3375** | **0.5723 ± 0.3284** | 0.7627 ± 0.3515 |
| **5** | 0.6184 ± 0.3354 | 0.5836 ± 0.3066 | 0.5604 ± 0.2931 | 0.7573 ± 0.3488 |
| **10** | 0.6172 ± 0.3389 | 0.5874 ± 0.2964 | 0.5435 ± 0.2753 | 0.7663 ± 0.3488 |
| **20** | 0.6196 ± 0.3387 | 0.5884 ± 0.3006 | 0.5493 ± 0.2704 | **0.7760 ± 0.3451** |

---

## 5. E4: Hub Analysis & Structural Bias

| Metric | Baseline Ranking | PPR Ranking |
|:---|:---:|:---:|
| **Spearman Correlation ($r$) with Node Degree** | **0.0666** ($p = 5.25 \times 10^{-4}$) | **0.1950** ($p = 1.26 \times 10^{-24}$) |
| **Interpretation** | Almost negligible degree dependence | **~3× stronger correlation** with degree; pronounced hub bias |

---

## 6. E5: Edge Noise Robustness & Homophily Boundary

*Baseline P@10 Reference:* **0.5446**

| Rewire Fraction | Resulting Homophily ($h$) | PPR P@10 (mean ± std) | $\Delta$ vs Baseline | Outcome |
|:---:|:---:|:---:|:---:|:---:|
| **0%** (Original) | **0.8100** | **0.5874 ± 0.2964** | **+0.0428** (+7.9%) | **Graph Helps** |
| **20%** | **0.6808** | **0.5544 ± 0.2820** | **+0.0098** (+1.8%) | **Graph Helps** |
| **40%** | **0.5563** | **0.5364 ± 0.2679** | **-0.0082** (-1.5%) | **Graph Hurts** |
| **60%** | **0.4288** | **0.5258 ± 0.2614** | **-0.0188** (-3.5%) | **Graph Hurts** |

---

## 7. E6: Per-Class P@10 Breakdown

| Class Name | Queries ($N_q$) | Baseline P@10 | PPR P@10 | $\Delta$ (PPR - Baseline) |
|:---|:---:|:---:|:---:|:---:|
| **Theory** | 29 | 0.5448 | **0.6310** | **+0.0862** (+15.8%) |
| **Probabilistic_Methods** | 163 | 0.6043 | **0.6663** | **+0.0620** (+10.3%) |
| **Reinforcement_Learning** | 74 | 0.5054 | **0.5500** | **+0.0446** (+8.8%) |
| **Case_Based** | 68 | 0.4691 | **0.4985** | **+0.0294** (+6.3%) |
| **Neural_Networks** | 84 | 0.6060 | **0.6333** | **+0.0274** (+4.5%) |
| **Rule_Learning** | 47 | 0.4660 | **0.4894** | **+0.0234** (+5.0%) |
| **Genetic_Algorithms** | 35 | 0.4543 | **0.4571** | **+0.0029** (+0.6%) |

---

## 8. Plain-Language Observations

### Where Graph Structure (PPR) Helped
1. **Deeper Precision Expansion (P@10 and P@20):**  
   PPR consistently outperforms flat TF-IDF in retrieving relevant papers further down the ranked list (P@10 rises from 54.5% to 58.7%, and P@20 from 49.4% to 54.4%). Because Cora exhibits high edge homophily (81%), walking across citation edges traverses cohesive topical subcommunities that text matching alone missed due to vocabulary mismatch.
2. **Hybrid Blending Dominance:**  
   The hybrid blend ($\lambda = 0.5$) achieved the highest P@5 (62.1%) and P@20 (55.4%), marrying the high-confidence direct lexical matches from TF-IDF with the cluster-expansion coverage of PPR.
3. **High-Homophily Subdisciplines:**  
   Classes like `Theory` (+8.6%) and `Probabilistic_Methods` (+6.2%) experienced substantial boosts from PPR, demonstrating that well-connected, structurally cohesive citation clusters provide strong contextual expansion.

### Where Graph Structure (PPR) Did Not Help (or Hurt)
1. **Top-1 Rank (MRR):**  
   PPR did **not** beat the baseline on Mean Reciprocal Rank (Baseline MRR = 0.7815 vs PPR MRR = 0.7663). Flat TF-IDF is superior at identifying the single best matching paper. PPR diffuses probability across multiple neighbors, sometimes demoting the best textual match below a high-degree structural neighbor.
2. **Noisy / Degraded Graphs ($h < 0.60$):**  
   When random rewiring degraded edge homophily below ~60% (at 40% rewiring, homophily dropped to 55.6%), PPR retrieval fell below the text-only baseline (0.5364 vs 0.5446). At 60% rewiring, PPR deteriorated further to 0.5258. Random walks on non-homophilous citation networks actively spread noise.
3. **Sparsely Connected or Diffuse Classes:**  
   `Genetic_Algorithms` saw essentially zero improvement ($\Delta = +0.0029$). In classes where terminology is heterogeneous or citation connectivity is sparse, PPR cannot find dense communities to exploit.

### Surprising and Non-Obvious Insights
1. **Fewer Seeds Won ($n_{\text{seeds}} = 3$ outperformed $n_{\text{seeds}} = 20$):**  
   Counter-intuitively, initializing PPR with only the top 3 baseline results yielded higher P@10 (60.0%) and P@20 (57.2%) than using 10 or 20 seeds. Expanding the seed set introduced lower-ranked, off-topic papers into the personalization vector, which 'poisoned' the random walk by propagating into unrelated citation neighborhoods.
2. **Quantifiable Hub Bias:**  
   PPR exhibited nearly three times stronger correlation with node degrees ($r = 0.1950$ vs $r = 0.0666$, $p = 1.26 \times 10^{-24}$). Prominent, highly cited hub papers frequently appeared in top-10 lists across diverse queries simply due to their graph centrality, even when their query-specific textual relevance was secondary.

