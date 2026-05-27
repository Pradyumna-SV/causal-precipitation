# Identification-first causal inference for Pacific Northwest winter precipitation extremes

## Abstract

Extreme precipitation in the western United States is shaped by coupled ocean-atmosphere variability, but translating teleconnection patterns into causal claims is difficult because discovery graphs, confounding adjustment, and robustness checks are often treated separately. We present an auditable causal workflow for ERA5 monthly reanalysis panels over three western US regions from 1979-2023. The pipeline combines time-series causal discovery (PCMCI+ and VARLiNGAM), a documented projection to a lag-0 acyclic panel graph, domain backbone edges for identification, inverse-probability weighting, doubly robust estimation, falsification checks, temporal holdout validation, logistic baselines, and targeted sensitivity analyses. The primary estimand is the average causal effect of warm regional sea-surface temperature anomalies (`sst_warm`) on 90th-percentile precipitation extremes (`tp_extreme`) during ONDJFM in the Pacific Northwest. In the full sample, IPW estimates a risk difference of 0.146 (95% CI 0.026, 0.237), and DR estimates 0.160 (95% CI 0.040, 0.245). In a temporal holdout that fits nuisance models on 1979-1999 and evaluates 2000-2023, the effect remains positive and statistically separated from zero: IPW 0.180 (95% CI 0.016, 0.361) and DR 0.066 (95% CI 0.045, 0.085). The effect does not generalize cleanly to California, the Intermountain West, pooled western-US stacking, or the NDJF-only seasonal variant, and broad observed-covariate adjustment remains a sensitivity concern. The contribution is therefore a reproducible, identification-audited causal climate workflow for making narrow claims inspectable, including where they fail.

## 1. Introduction

Western US precipitation extremes are high-impact events for flood risk, water resources, infrastructure, and ecosystem stress. Large-scale climate variability, including sea-surface temperature anomalies and ENSO-related circulation, is known to modulate precipitation patterns, but there is a gap between detecting associations and making transparent causal claims about extreme precipitation risk.

This paper asks a deliberately narrow question: under an explicit identification graph and reproducible adjustment strategy, does warm regional SST increase cool-season extreme precipitation risk in the Pacific Northwest?

The analysis is designed around three constraints. First, the causal estimand must be stated before searching over many specifications. Second, the graph used for identification must be documented rather than silently equating a discovered time-series graph with ground truth. Third, any headline effect must survive both estimator agreement and temporal holdout validation.

Our main finding is that the Pacific Northwest ONDJFM estimand satisfies these requirements. In contrast, parallel estimates for California, the Intermountain West, and a pooled western-US stack are weak or null. This is not evidence for a broad western-US causal effect. It is evidence for a narrower, season-specific Pacific Northwest effect under the stated assumptions.

## 2. Contributions

1. We implement a reproducible ERA5-to-causal-inference pipeline covering preprocessing, causal discovery, graph projection, backdoor identification, ACE estimation, robustness checks, temporal holdout validation, logistic baselines, targeted sensitivity checks, figures, and verification.

2. We report a pre-specified Pacific Northwest ONDJFM causal estimand for warm regional SST anomalies and 90th-percentile precipitation extremes. The effect is positive under both IPW and DR and remains positive in a held-out post-2000 evaluation window.

3. We explicitly separate primary, holdout-confirmed claims from exploratory, negative, or assumption-sensitive evidence: California, the Intermountain West, pooled western-US stacking, ENSO-threshold alternatives, broad observed adjustment, and NDJF-only season variants are not promoted to headline claims.

## 3. Data and preprocessing

The analysis uses ERA5 monthly reanalysis data for 1979-01 through 2023-12 over a western-US domain. The processed panels cover three regions. Regional SST is the SST anomaly averaged over the same region box; for the Pacific Northwest this is 45-50 N and 125-115 W, not a separate named coastal SST index.

| Region | Inference rows (ONDJFM) | Extreme fraction | Notes |
|---|---:|---:|---|
| Pacific Northwest | 270 | 0.10 | Primary gate-B region |
| California | 270 | 0.10 | Secondary regional estimate |
| Intermountain West | 270 | 0.10 | Secondary regional estimate |

Monthly anomalies are computed relative to the configured baseline period, and `tp_extreme` is a binary indicator for the regional 90th percentile of precipitation. The primary season mask is ONDJFM: October, November, December, January, February, and March. The verification artifact confirms three processed region panels, 540 monthly observations per region, expected extreme fractions, raw ERA5 files, figures, robustness outputs, and temporal holdout outputs.

## 4. Causal design

### 4.1 Primary estimand

The primary estimand is:

> The average causal effect, as a risk difference, of warm regional SST anomaly (`sst_warm`, defined as regional `sst >= 0.5 K`) on 90th-percentile precipitation extremes (`tp_extreme`) for Pacific Northwest ONDJFM months from 1979-2023.

The primary adjustment set selected from the identification graph is `z500`. The full-sample inference uses 270 ONDJFM rows. Larger sets such as `{z500, nino34}`, `{z500, t2m}`, and `{z500, nino34, t2m, swvl1}` are sensitivity checks because added variables can have different causal roles under different graph assumptions.

### 4.2 Discovery-to-identification graph

The pipeline uses PCMCI+ and VARLiNGAM for time-series causal discovery. Their consensus records are not treated as a true data-generating DAG. Instead, the pipeline projects the time-series discovery output into a lag-0 acyclic panel DAG by:

1. dropping self-edges,
2. retaining lag-0 links only,
3. greedily adding edges in order of PCMCI p-value while preserving acyclicity.

This structural panel DAG is merged with a domain identification backbone:

```text
nino34 -> z500
nino34 -> sst
z500   -> tp
sst    -> tp
tp     -> tp_extreme
```

Backdoor adjustment sets are then identified on the merged identification DAG. This makes the identification assumptions inspectable and reproducible, while avoiding the overclaim that discovery algorithms recover the true atmospheric causal graph.

### 4.3 Estimators and uncertainty

The primary effect is estimated using:

- inverse probability weighting (IPW),
- a doubly robust estimator (DR),
- 2,000 block bootstrap replicates,
- 12-month block size.

Robustness checks include propensity overlap, effective sample size, placebo month-shuffle IPW, IPW/DR agreement, and an alternate NDJF seasonal artifact check. Temporal holdout validation fits nuisance models on 1979-1999 and evaluates the ACE on 2000-2023.

## 5. Results

### 5.1 Primary Pacific Northwest estimate

The primary Pacific Northwest effect is positive under both estimators:

| Estimator | Risk difference | 95% CI |
|---|---:|---:|
| IPW | 0.146 | [0.026, 0.237] |
| DR | 0.160 | [0.040, 0.245] |

The IPW overlap diagnostics are acceptable: ESS/n = 0.701, propensity range [0.116, 0.408], and 99th-percentile weight = 6.217. The adjusted odds-ratio E-value summary is 5.483 for the point estimate, but this should be treated as a sensitivity summary rather than proof of unconfoundedness.

### 5.2 Temporal holdout confirmation

The temporal holdout is the strongest evidence added beyond full-sample estimation. Nuisance models are fit on 1979-1999 ONDJFM rows (126 rows) and evaluated on 2000-2023 ONDJFM rows (144 rows).

| Holdout estimator | Risk difference | 95% CI |
|---|---:|---:|
| IPW | 0.180 | [0.016, 0.361] |
| DR | 0.066 | [0.045, 0.085] |

Both held-out estimates are positive and both confidence intervals exclude zero. The holdout artifact therefore reports both `directional_replication: true` and `strict_statistical_replication: true`. The DR holdout interval is much tighter than IPW: the audit reports an IPW/DR width ratio of 8.7. This reflects fixed train-fitted nuisance models during evaluation bootstrap resampling and the stabilizing role of the DR outcome model, not a free robustness guarantee.

### 5.3 Logistic baseline comparison

| Model | Risk difference | 95% CI | Interpretation |
|---|---:|---:|---|
| Crude logistic | 0.094 | [-0.008, 0.205] | Positive but not separated from zero |
| Graph-adjusted logistic (`z500`) | 0.147 | [0.051, 0.237] | Close to IPW; simple adjusted signal is visible |
| Expanded logistic (`z500`, `nino34`, `t2m`, `swvl1`) | -0.007 | [-0.105, 0.100] | Broad observed adjustment changes the estimand |

### 5.4 Regional and pooled estimates

The headline effect is not western-US-wide:

| Region / stack | IPW risk difference | IPW 95% CI | DR risk difference | DR 95% CI |
|---|---:|---:|---:|---:|
| Pacific Northwest | 0.146 | [0.026, 0.237] | 0.160 | [0.040, 0.245] |
| California | 0.001 | [-0.109, 0.152] | -0.020 | [-0.119, 0.146] |
| Intermountain West | 0.014 | [-0.081, 0.135] | 0.023 | [-0.075, 0.147] |
| Pooled western-US stack | -0.007 | [-0.062, 0.054] | -0.015 | [-0.093, 0.062] |

These results support a narrow Pacific Northwest claim and argue against presenting the analysis as a unified western-US effect.

### 5.5 Robustness and falsification

The robustness bundle passes the pre-specified Gate C threshold by satisfying three checks:

| Check | Result |
|---|---|
| Overlap / ESS | Pass |
| Month-shuffle placebo IPW CI includes zero | Pass |
| IPW and DR same sign with overlapping CIs | Pass |
| NDJF-only seasonal variant sign-stable | Fail |

Backbone sensitivity preserves the positive sign under several graph perturbations, but broad observed adjustment is null and has poor overlap. Lag sensitivity preserves the positive sign under lagged covariates, although `sst_lag1` is imprecise because overlap degrades. The NDJF-only result is also important. ONDJFM is the frozen current-release season and is climatologically motivated as a broad PNW cool-season window, but the repository does not prove fully prospective seasonal pre-registration. Temporal holdout reduces overfitting concern; it does not erase all seasonal-selection risk.

### 5.5 Counterfactual outputs

The pipeline also computes a linear SCM-style `do(ENSO=0)` counterfactual for precipitation. These outputs are useful as a diagnostic and visualization, but they should be interpreted conservatively. The manuscript should not use them as the central causal effect because the primary validated estimand is warm regional SST on precipitation-extreme risk.

## 6. Figures

The current figure set can support the manuscript as follows:

| Figure artifact | Role in paper |
|---|---|
| `figures/fig1_domain_map.pdf` | Study domain and regional masks |
| `figures/fig2_enso_precip_ts.pdf` | Time-series context for ENSO and precipitation |
| `figures/fig3_pcmci_graphs.pdf` | Discovery graph evidence, framed as exploratory input |
| `figures/fig4_varlingam_heatmaps.pdf` | Complementary discovery / dependence structure |
| `figures/fig5_ace_comparison.pdf` | Supporting ACE comparison across regions and exploratory Niño contrast |
| `figures/fig6_counterfactual_cdfs.pdf` | Supplementary counterfactual diagnostic |
| `figures/fig7_primary_holdout_summary.pdf` | Main results figure: full-sample regional/pooled context plus temporal holdout |

Recommended main text figures:

1. Domain and causal estimand schematic.
2. `fig7_primary_holdout_summary.pdf`: ACE forest plot with full-sample PNW, CA, IMW, pooled stack, and PNW temporal holdout.
3. Robustness and holdout panel: overlap, placebo, full-sample PNW, holdout PNW.

The primary paper result should use `fig7_primary_holdout_summary.pdf`; discovery graphs and counterfactual CDFs are better suited for methods or supplementary material if space is tight.

## 7. Discussion

The central result is that warm regional SST anomalies are associated with a higher probability of ONDJFM extreme precipitation in the Pacific Northwest under the specified identification graph. The effect is not only present in the full sample but also survives a temporal holdout evaluation. This is stronger than a purely exploratory discovery result because the estimand, adjustment set, falsification checks, and holdout rule are all explicit.

At the same time, the result is geographically and seasonally specific. California and the Intermountain West do not show comparable effects, and pooling the three regions washes out the signal. The NDJF-only variant also weakens or reverses the primary result. These negatives are not failures of the study; they are constraints on the claim.

The likely interpretation is that the Pacific Northwest ONDJFM relationship captures a seasonally broad cool-season pathway in which regional SST anomalies and mid-tropospheric circulation (`z500`) align with extreme precipitation conditions. The pipeline does not prove this physical mechanism. It provides an identification-audited statistical estimate consistent with it.

## 8. Limitations

1. The causal interpretation depends on the identification DAG, especially the domain backbone and the projected lag-0 graph. Discovery edges are inputs to identification, not proof of the true graph.

2. ERA5 is a reanalysis product, not direct observational truth. Model assimilation choices may affect regional SST, circulation, and precipitation relationships.

3. Monthly aggregation smooths submonthly storm dynamics. Atmospheric rivers are likely mechanisms behind many Pacific Northwest precipitation extremes, but this monthly panel does not include atmospheric-river covariates and should not be read as storm-scale AR causality.

4. The temporal holdout validates the post-2000 period but does not replace independent external validation in another reanalysis product or observational dataset. The current release scopes MERRA-2/JRA-55 validation as future work and does not claim cross-product replication.

5. The crude gamma sensitivity block is not a Rosenbaum bounds analysis and should not be described as such.

6. The pooled western-US estimate is null, so the study should avoid western-US-wide causal language.

## 9. Reproducibility statement

The full primary pipeline is run by:

```bash
make pipeline
```

This regenerates preprocessing, discovery, inference, figures, robustness, temporal holdout validation, pooled ACE, verification, and the manuscript brief. The current verification status is `PASS`, with `error_count: 0`. The key machine-readable artifacts are:

- `results/ace_all_regions.json`
- `results/holdout_validation.json`
- `results/robustness_report.json`
- `results/pooled_ace.json`
- `results/verification_report.json`
- `results/manuscript_brief.json`

## 10. Claim language for final paper

Recommended:

> Under a frozen identification graph combining a domain backbone with a lag-0 acyclic projection of PCMCI+/VARLiNGAM discovery output, warm regional SST anomalies increase ONDJFM 90th-percentile precipitation-extreme risk in the Pacific Northwest. The full-sample estimate is positive under IPW and DR, and the effect strictly replicates in a 2000-2023 temporal holdout after fitting nuisance models on 1979-1999.

Avoid:

> Warm SST causes western-US precipitation extremes.

Avoid:

> PCMCI+ and VARLiNGAM recover the true climate causal graph.

Avoid:

> ENSO intervention counterfactuals prove no ENSO effect.

## 11. Current readiness

The project is now suitable for a workshop-style applied causal climate paper if the manuscript stays narrow and transparent. The strongest framing is:

- primary empirical claim: Pacific Northwest ONDJFM warm-SST effect on extreme precipitation risk;
- methodological contribution: reproducible identification-first pipeline;
- integrity contribution: explicit negative results, temporal holdout, and exploratory/primary separation.

The next paper step is to convert this draft into the target venue format and add citations while keeping `fig7_primary_holdout_summary.pdf` as the main quantitative result figure.
