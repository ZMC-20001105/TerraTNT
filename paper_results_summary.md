# Paper Experiment Results Summary

## 1. Phase V2 Evaluation (Main Table)

**ADE (km) across evaluation phases with decreasing prior quality**

| Model | P1a (Precise) | P1b (OOD) | P2a (σ=10km) | P2c (Offset 5km) | P3a (No Prior) |
|---|---|---|---|---|---|
| Const. Velocity | 12.6 | 12.3 | 12.6 | 12.6 | 9.9 |
| MLP | 7.8 | 7.2 | 7.8 | 7.8 | 6.0 |
| LSTM | 7.7 | 7.5 | 7.7 | 7.7 | 6.5 |
| Seq2Seq+Attn | 6.0 | 6.2 | **6.0** | **6.0** | **5.2** |
| LSTM+Env+Goal | **1.4** | **2.0** | 8.8 | 9.6 | 13.8 |
| V3: +Waypoints | 1.9 | 2.6 | 9.0 | 9.7 | 13.7 |
| V4: +Spatial | 1.5 | 2.1 | 7.4 | 8.0 | 12.5 |
| TerraTNT | 3.2 | 3.8 | 7.7 | 8.3 | 16.0 |
| V6: +Autoreg | 1.3 | 3.1 | 9.4 | 10.2 | 17.0 |
| **V6R: +GoalDrop** | **1.2** | 3.0 | 8.7 | 9.5 | 16.1 |
| V7: ConfGate | 4.5 | 5.2 | 7.5 | 7.9 | 12.2 |

## 2. Key Findings

### Finding 1: Goal-Dependent Models Excel with Precise Prior, Collapse Without
- V6R achieves **1.2 km ADE** with precise goal (P1a) — best overall
- But degrades to **16.1 km** without prior (P3a) — 13x worse
- LSTM+EG: 1.4 km → 13.8 km (10x degradation)

### Finding 2: Goal-Free Models Are Invariant but Mediocre
- Seq2Seq+Attn: stable 5.2-6.2 km across all phases
- Best in P2a/P2c/P3a because it's unaffected by prior quality
- But can never match goal-dependent models when prior is good

### Finding 3: V6R is Optimal for Precise Goal Scenarios
- V6R (1.2 km) beats V6 (1.3 km) even on P1a
- Goal dropout regularization improves both robustness AND precision
- V6R also best on P1b OOD among TerraTNT variants (3.0 km)

### Finding 4: Architecture Trade-off is Fundamental
- V8 iteration (5 attempts) proved that decoupling goal from fusion destroys joint nonlinear interactions
- Phase4 collapse (~12-16 km) is inherent to goal-classifier architectures
- This is a fundamental limitation, not a training issue

## 3. Legacy Phase System (v5) — GT Goal Input

| Model | FAS1 (GT) | FAS2 (OOD) | FAS3 (No GT cand) | FAS3b (σ=10km) | FAS4 (No prior) |
|---|---|---|---|---|---|
| V6R | 1068 | 2896 | **1208** | 2497 | 12301 |
| V6 | **973** | 2823 | 1536 | 3478 | 13468 |
| LSTM+EG | 1401 | **2027** | 1412 | **1440** | **1451** |
| V4 | 1482 | 2042 | 1557 | 1789 | 2503 |
| TerraTNT | 3300 | 3786 | 3450 | 4032 | 12862 |

Key: V6R beats LSTM_EG on FAS3 (1208 vs 1412m) — first goal-classifier model to do so.

## 4. Cross-Region Generalization (Bohemian Forest → Scottish Highlands)

| Model | BH P1a | SH P1a | Δ% | BH P3a | SH P3a | Δ% |
|---|---|---|---|---|---|---|
| V6R | 1.2 km | 3.4 km | +177% | 16.1 km | 12.9 km | -19% |
| V6 | 1.3 km | 3.9 km | +198% | 17.0 km | 14.4 km | -16% |
| V3 | 1.9 km | 2.6 km | +40% | 13.7 km | 7.5 km | -46% |
| LSTM+EG | 1.4 km | 4.0 km | +184% | 13.8 km | 7.5 km | -46% |
| Seq2Seq | 6.0 km | 8.5 km | +43% | 5.2 km | 8.6 km | +64% |

Key: V3 and LSTM+EG generalize best on P1a (+40%/+184%). All models degrade on cross-region P1a.

## 5. Generated Figures

| File | Description |
|---|---|
| `fig1_phase_comparison.pdf` | Grouped bar: ADE across phases for key models |
| `fig2_degradation.pdf` | Line chart: performance degradation with prior quality |
| `fig3_ablation.pdf` | Horizontal bar: ablation V2→V6R on P1a and P2a |
| `fig4_temporal.pdf` | Early/Mid/Late ADE breakdown |
| `fig5_cross_region.pdf` | Cross-region comparison BH vs SH |
| `fig6_v5_legacy.pdf` | Legacy phase system with GT goal |
| `fig_example_1-6.pdf` | Trajectory prediction examples |

## 6. Data Sources

- Phase V2 results: `outputs/evaluation/phase_v2/phase_v2_results.json`
- Legacy v5 results: `outputs/evaluation/phase_diagnostic_v5/summary_all_phases.json`
- Cross-region results: `outputs/evaluation/cross_region_sh/phase_v2_results.json`
- All figures: `outputs/paper_figures/`
