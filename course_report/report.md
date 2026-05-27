# Identification-First Causal Inference for Pacific Northwest Extreme Precipitation

## Executive Summary

This report expands the NeurIPS CCAI-style paper into a course-project version.
The core question is narrow: under an explicit identification graph, does warm
regional SST increase ONDJFM 90th-percentile precipitation-extreme risk in the
Pacific Northwest?

The answer is cautiously positive for ERA5. In the full Pacific Northwest
ONDJFM sample, IPW estimates a risk difference of 0.146 and DR estimates 0.160.
Both intervals exclude zero. In a temporal holdout where nuisance models are fit
on 1979-1999 and evaluated on 2000-2023, IPW estimates 0.180 and DR estimates
0.066, again with intervals excluding zero. The claim does not generalize to
California, the Intermountain West, pooled western-US stacking, or the NDJF-only
seasonal variant.

The important course-project contribution is not a new estimator. It is the
discipline of the causal workflow: discovery is separated from identification,
graph assumptions are explicit, adjustment sets are auditable, and negative or
assumption-sensitive results are reported instead of hidden.

## Data and Panel Construction

The project uses ERA5 monthly reanalysis from 1979-01 through 2023-12. The
western-US domain is aggregated into three regional panels: Pacific Northwest,
California, and Intermountain West. The Pacific Northwest box is 45-50 N and
125-115 W. Regional SST means the monthly SST anomaly averaged over that same
box.

The outcome `tp_extreme` is one when regional monthly precipitation is at or
above the region-specific 90th percentile. The treatment `sst_warm` is one when
regional SST anomaly is at least 0.5 K. The primary season is ONDJFM, defined as
October through March.

## Projection Algorithm

The causal discovery stage uses PCMCI+ and VARLiNGAM to summarize time-series
dependence. Those outputs are not treated as the true atmospheric causal graph.
They are converted into a monthly panel graph through a documented projection:

1. Drop self-edges because they do not define cross-variable panel adjustment.
2. Retain lag-0 links only for the panel DAG used by DoWhy and ACE estimation.
3. Sort candidate lag-0 edges by discovery strength.
4. Greedily add edges while preserving acyclicity.
5. Merge the resulting structural panel DAG with a domain identification
   backbone.

The domain backbone is:

```text
nino34 -> z500
nino34 -> sst
z500   -> tp
sst    -> tp
tp     -> tp_extreme
```

This is the key modeling choice. It makes the assumptions visible, but it also
means the causal interpretation depends on the backbone and the projection rule.

## Identification and Adjustment

Under the merged identification graph, the primary backdoor adjustment set for
`sst -> tp_extreme` in the Pacific Northwest is `{z500}`. This is intentionally
minimal. Adding variables is not automatically better: `nino34`, `t2m`, and
`swvl1` can play different roles under different graphs, including ancestors,
mediators, descendants, or precision variables.

The project therefore treats `{z500}` as the primary graph-identified set and
uses larger observed sets as sensitivity checks.

## Primary Results

| Estimator | Risk Difference | 95% CI |
|---|---:|---:|
| IPW | 0.146 | [0.026, 0.237] |
| DR | 0.160 | [0.040, 0.245] |

Overlap diagnostics are acceptable for the primary IPW model: ESS/n = 0.701,
propensity range [0.116, 0.408], and 99th-percentile weight = 6.217.

## Temporal Holdout

The holdout fits nuisance models on 1979-1999 and evaluates on 2000-2023.

| Estimator | Risk Difference | 95% CI |
|---|---:|---:|
| IPW holdout | 0.180 | [0.016, 0.361] |
| DR holdout | 0.066 | [0.045, 0.085] |

The DR interval is much tighter than IPW. The audit shows the IPW interval is
8.7 times wider. This is not magic: both holdout estimators keep train-fitted
nuisance models fixed during evaluation bootstrap resampling, and the DR
estimate is stabilized by the outcome-effect model. The narrow DR interval is
therefore useful but model-dependent.

## Logistic Baseline Comparison

| Model | Covariates | Risk Difference | 95% CI |
|---|---|---:|---:|
| Crude logistic | `sst_warm` | 0.094 | [-0.008, 0.205] |
| Graph-adjusted logistic | `sst_warm + z500` | 0.147 | [0.051, 0.237] |
| Expanded logistic | `sst_warm + z500 + nino34 + t2m + swvl1` | -0.007 | [-0.105, 0.100] |

The adjusted logistic result is close to IPW, which means the signal is visible
in a simple model. The value of the causal workflow is that it formalizes the
estimand, records the graph assumptions, checks overlap, runs falsifications,
and performs temporal validation.

## Backbone and Adjustment Sensitivity

Graph perturbations that add `nino34 -> tp`, add `z500 -> sst`, or remove
`nino34 -> sst` preserve a positive primary estimate. However, broad observed
adjustment is null and has poor overlap: ESS/n falls to 0.198. This is the
strongest remaining weakness in the identification story. It should be framed as
graph dependence, not swept away.

## Lag and Projection Sensitivity

Lag sensitivity checks whether the lag-0 projection is driving the result.

| Variant | IPW Risk Difference | 95% CI | Note |
|---|---:|---:|---|
| Base `{z500}` | 0.145 | [0.036, 0.246] | Positive |
| Add `sst_lag1` | 0.166 | [-0.056, 0.441] | Positive but imprecise; overlap degrades |
| Add `z500_lag1, nino34_lag1` | 0.227 | [0.047, 0.449] | Positive |
| Add `tp_lag1` | 0.156 | [0.037, 0.282] | Positive; autoregressive stress check |

The result is not solely an artifact of dropping lagged variables, although some
lagged specifications are less stable.

## Seasonal Provenance and NDJF

ONDJFM is the frozen primary season for the current analysis release and is
climatologically motivated as a broad Pacific Northwest cool-season window. The
repository does not prove fully prospective seasonal pre-registration, so the
honest language is that temporal holdout validates the current frozen claim but
does not remove all seasonal-selection concern.

The NDJF-only failure is therefore important. It means the claim is
ONDJFM-specific and should not be sold as a generic winter effect.

## Climate and Decision Relevance

The practical decision context is seasonal risk screening for Pacific Northwest
water managers, flood-risk planners, reservoir operators, and emergency
preparedness offices. The estimate is not an operational forecast. It is a
scenario and monitoring input: when regional SST conditions are warm, planners
may want to prioritize closer monitoring, stress tests, and risk communication
for ONDJFM extreme precipitation.

## Atmospheric Rivers and Monthly Aggregation

Atmospheric rivers are likely submonthly mechanisms behind many Pacific
Northwest precipitation extremes. This project does not include an atmospheric
river covariate and does not resolve storm-scale event pathways. The monthly
panel estimates a regional risk-difference effect under a causal abstraction,
not atmospheric-river event causality.

## External Validation

External validation in MERRA-2 or JRA-55 remains future work. The current
release documents the minimum viable validation scope but does not claim
cross-product replication.

## Final Claim

The strongest defensible claim is:

> Under a frozen identification graph combining a domain backbone with a lag-0
> acyclic projection of causal-discovery output, warm regional SST anomalies
> increase ONDJFM 90th-percentile precipitation-extreme risk in the Pacific
> Northwest in ERA5. The effect is positive under IPW and DR and strictly
> replicates in a post-2000 temporal holdout, but it is geographically,
> seasonally, and graph-assumption specific.
