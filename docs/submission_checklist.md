# Submission checklist

Use this checklist to turn `docs/manuscript_draft.md` into a workshop submission.

## Claim discipline

- [ ] Lead with the Pacific Northwest ONDJFM claim only.
- [ ] State the exact estimand: `sst_warm -> tp_extreme`, Pacific Northwest, ONDJFM, 1979-2023, 90th-percentile precipitation extremes.
- [ ] Report temporal holdout as confirmatory: train 1979-1999, evaluate 2000-2023.
- [ ] State that California, Intermountain West, pooled western-US stack, and NDJF-only variants are weak/null/mixed.
- [ ] Avoid "western-US-wide causal effect" language.
- [ ] Avoid saying PCMCI+/VARLiNGAM recover the true causal graph.

## Numbers to carry into the paper

Primary full-sample Pacific Northwest:

- IPW risk difference: `0.146`, 95% CI `[0.026, 0.237]`
- DR risk difference: `0.160`, 95% CI `[0.040, 0.245]`
- Overlap ESS/n: `0.701`

Temporal holdout:

- IPW risk difference: `0.180`, 95% CI `[0.016, 0.361]`
- DR risk difference: `0.066`, 95% CI `[0.045, 0.085]`
- `strict_statistical_replication: true`

Pooled western-US stack:

- IPW risk difference: `-0.007`, 95% CI `[-0.062, 0.054]`
- DR risk difference: `-0.015`, 95% CI `[-0.093, 0.062]`

## Figures

- [ ] Keep `figures/fig1_domain_map.pdf` as domain / region figure.
- [ ] Use `figures/fig7_primary_holdout_summary.pdf` as the main quantitative results figure.
- [ ] Keep `figures/fig5_ace_comparison.pdf` as supporting/legacy regional ACE context.
- [ ] Confirm `figures/fig7_primary_holdout_summary.pdf` shows full-sample PNW IPW/DR, holdout PNW IPW/DR, and California / Intermountain / pooled null estimates.
- [ ] Move discovery graphs to supplement unless space allows a compact methods figure.

## Citations to add

- [ ] ERA5 reanalysis.
- [ ] PCMCI+ / Tigramite.
- [ ] VARLiNGAM / LiNGAM.
- [ ] Pearl / do-calculus / backdoor criterion.
- [ ] IPW / propensity score weighting.
- [ ] Doubly robust / AIPW / EconML if cited.
- [ ] Block bootstrap for dependent time series.
- [ ] E-values: VanderWeele and Ding.
- [ ] Regional hydroclimate / ENSO teleconnection background for western US / Pacific Northwest.

## Reproducibility

- [ ] Run `make pipeline` before final submission.
- [ ] Confirm `results/verification_report.json` has `status: PASS`.
- [ ] Confirm `results/holdout_validation.json` has `strict_statistical_replication: true`.
- [ ] Confirm `results/manuscript_brief.json` matches the numbers in the paper.
- [ ] Do not cite stale values from `results/experiment_log.jsonl`; use current artifacts under `results/`.

## Final wording guardrails

Recommended title shape:

> Identification-first causal inference for Pacific Northwest winter precipitation extremes in ERA5

Recommended main claim:

> Under a frozen identification graph and temporal holdout validation, warm regional SST anomalies increase ONDJFM extreme-precipitation risk in the Pacific Northwest.

Do not write:

> Warm SST causes extreme precipitation across the western United States.

Do not write:

> Discovery recovered the true climate DAG.
