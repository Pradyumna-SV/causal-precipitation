# Paper spine template (use after gates A–D pass)

Fill this in only when [`docs/analysis_plan.md`](analysis_plan.md) gates **A–D** are satisfied for the **primary** estimand (see [`config/analysis_registry.yaml`](../config/analysis_registry.yaml)). The narrative should lead with **one** strong, identification-credible effect—not with cross-region heterogeneity as the hook.

## Title (sketch)

Identification-first panel inference for [treatment] → [outcome] on ERA5 western US [season / annual] windows.

## Abstract (150–200 words)

- **Problem:** [One sentence on extremes / teleconnections.]
- **Method:** [Backbone ∪ lag-0 projected DAG; IPW/DR; robustness checks in `results/robustness_report.json`; temporal holdout in `results/holdout_validation.json`; closed preset grid.]
- **Result:** [Primary risk difference + CI; DR agreement; holdout replication status; optional pooled Western US estimate from `results/pooled_ace.json`.]
- **Limitation:** [Reanalysis, monthly scale, linear SCM as sensitivity only, discovery not ground truth.]

## Three contributions (max)

1. **Pipeline:** Reproducible ERA5 → discovery projection → identification DAG → ACE + falsification bundle.
2. **Empirical:** Quantified [estimand] with uncertainty (bootstrap) and overlap/placebo gates passed.
3. **Integrity:** Closed preset list + documented multiplicity / exploratory vs primary + temporal holdout status (see `manuscript_brief.json` `analysis` / `multiplicity` / `temporal_holdout_validation` blocks).

## Main text figure strategy

- **Figure 1:** Estimand + identification schematic (DAG).
- **Figure 2:** Primary ACE + CIs (IPW/DR); supplementary forest across **presets** if needed.
- **Figure 3:** Robustness diagnostics (ESS, placebo, optional seasonal alternate).

## If gates A–D do not pass

Do not use this template as the submission spine; keep iterating within `config/presets/` or document honest negative evidence with methods emphasis.
