# CCAI submission readiness audit

This audit defines what is still needed before generating a NeurIPS Climate Change AI (CCAI)-style paper. It is not the final paper. It is the acceptance-focused checklist and scaffold plan for turning the current verified analysis into a concise workshop submission.

## Target venue assumptions

- **Venue target:** NeurIPS Workshop: Tackling Climate Change with Machine Learning (Climate Change AI / CCAI).
- **Submission style assumption:** Use the current NeurIPS workshop LaTeX style unless CCAI publishes a different year-specific template. For a 2025-targeted draft, assume `neurips_2025.sty` and anonymized submission mode.
- **Length assumption:** Plan for a compact workshop paper: approximately 4 main-content pages plus references and optional appendix. If the active CCAI call allows a longer page limit, use the extra space for appendix material rather than broadening the claim.
- **Submission type assumption:** Position as an applied paper, not a proposal and not a methodological novelty paper.
- **Primary reviewer expectation:** A CCAI reviewer should see clear climate relevance, reproducible ML/causal workflow, a defensible claim, and honest limitations within the first two pages.

## Current evidence status

| Requirement | Current artifact | Status |
|---|---|---|
| Reproducible end-to-end pipeline | `make pipeline`, `results/verification_report.json` | Ready: `PASS`, `error_count: 0` |
| Primary causal effect | `results/ace_all_regions.json` | Ready for PNW ONDJFM only |
| Temporal holdout | `results/holdout_validation.json` | Ready: `strict_statistical_replication: true` |
| Robustness / falsification | `results/robustness_report.json` | Ready: Gate C pass, 3 checks passed |
| Main quantitative figure | `figures/fig7_primary_holdout_summary.pdf` | Ready |
| Main narrative draft | `docs/manuscript_draft.md` | Drafted, not template-formatted |
| Submission checklist | `docs/submission_checklist.md` | Drafted |
| LaTeX submission scaffold | `paper/` | Missing |
| Bibliography | `paper/references.bib` | Missing |

## Submission claim

Recommended claim:

> Under a frozen identification graph and temporal holdout validation, warm regional SST anomalies increase ONDJFM 90th-percentile precipitation-extreme risk in the Pacific Northwest.

Do not broaden this to a western-US-wide claim. The pooled western-US stack and two of the three individual regions are weak or null.

## Numeric claims and source artifacts

| Claim | Value | Source |
|---|---:|---|
| PNW full-sample IPW risk difference | `0.146 [0.026, 0.237]` | `results/ace_all_regions.json` |
| PNW full-sample DR risk difference | `0.160 [0.040, 0.245]` | `results/ace_all_regions.json` |
| PNW holdout IPW risk difference | `0.180 [0.016, 0.361]` | `results/holdout_validation.json` |
| PNW holdout DR risk difference | `0.066 [0.045, 0.085]` | `results/holdout_validation.json` |
| Full-sample overlap ESS/n | `0.701` | `results/robustness_report.json` / `results/ace_all_regions.json` |
| Pooled western-US IPW risk difference | `-0.007 [-0.062, 0.054]` | `results/pooled_ace.json` |
| Pooled western-US DR risk difference | `-0.015 [-0.093, 0.062]` | `results/pooled_ace.json` |
| Verification status | `PASS` | `results/verification_report.json` |

## Paper skeleton and page budget

Target a short paper that reviewers can understand quickly.

| Section | Target length | Purpose |
|---|---:|---|
| Abstract | 150-200 words | Claim, data, method, holdout, limitation |
| Introduction | 0.6 page | Climate motivation and why causal framing matters |
| Contributions | 0.25 page | Three bullets: pipeline, empirical claim, integrity |
| Data and estimand | 0.5 page | ERA5, regions, ONDJFM, `sst_warm -> tp_extreme` |
| Identification-first method | 0.9 page | Discovery-to-DAG projection, backbone, adjustment, IPW/DR |
| Results | 0.9 page | Figure 7, primary effect, holdout, null contexts |
| Robustness and limitations | 0.6 page | Gate C, NDJF failure, assumptions, reanalysis caveats |
| Reproducibility | 0.25 page | `make pipeline`, artifact list |
| Appendix | As allowed | Discovery graphs, counterfactual CDFs, experiment matrix |

## Main figures and tables

Main paper:

| Item | Artifact or content | Role |
|---|---|---|
| Figure 1 | `figures/fig1_domain_map.pdf` | Study domain and regional masks |
| Figure 2 | `figures/fig7_primary_holdout_summary.pdf` | Main quantitative result |
| Table 1 | New table in LaTeX | Estimand, rows, adjustment set, train/eval split, robustness status |

Appendix or supplement:

| Item | Artifact | Role |
|---|---|---|
| Figure A1 | `figures/fig3_pcmci_graphs.pdf` | PCMCI+ discovery context |
| Figure A2 | `figures/fig4_varlingam_heatmaps.pdf` | VARLiNGAM context |
| Figure A3 | `figures/fig5_ace_comparison.pdf` | Legacy ACE and exploratory Niño contrast |
| Figure A4 | `figures/fig6_counterfactual_cdfs.pdf` | Counterfactual diagnostic only |
| Table A1 | `results/experiments_index.json` summary | Exploratory matrix / null contexts |

## Required citation pack

Add these citations before final prose generation.

| Topic | Citation need | Why it matters |
|---|---|---|
| ERA5 | Hersbach et al. ERA5 | Establish data provenance |
| PCMCI+ / Tigramite | Runge / PCMCI+ / Tigramite references | Justify causal discovery component |
| VARLiNGAM / LiNGAM | Hyvarinen / Shimizu / VARLiNGAM references | Justify complementary discovery |
| Causal identification | Pearl; backdoor criterion | Explain identification graph and adjustment |
| Propensity/IPW | Rosenbaum and Rubin; Hirano/Imbens if needed | Ground IPW estimator |
| Doubly robust estimation | Robins / AIPW / DR learner reference | Ground DR estimator |
| EconML | EconML package citation if named | Implementation transparency |
| Block bootstrap | Kunsch or time-series block bootstrap reference | Justify dependent-time uncertainty |
| E-values | VanderWeele and Ding | Sensitivity summary |
| ENSO / PNW hydroclimate | Regional hydroclimate teleconnection references | Climate relevance |
| Causal ML for climate | Recent CCAI causal papers | Venue fit and positioning |

## Related-work positioning

The paper should not claim to invent a new causal estimator. The novelty is a reproducible applied workflow and an identification-audited empirical finding.

Position against three literatures:

1. **Climate teleconnections and hydroclimate extremes:** Existing work studies ENSO/SST/circulation links to western-US precipitation, often associational or process-focused.
2. **Causal discovery for climate:** Existing work applies causal graphs to climate systems, but graph discovery alone is not sufficient for a backdoor estimand.
3. **Applied causal ML:** Existing work estimates treatment effects with observational data; this paper contributes a climate-specific, reproducible, holdout-validated case study.

The strongest framing is: "We do not replace physical climate analysis; we make a narrow causal claim auditable end to end."

## Future LaTeX scaffold

Create this only when ready to generate the paper:

```text
paper/
  README.md
  main.tex
  references.bib
  neurips_2025.sty
  figures/
    fig1_domain_map.pdf
    fig7_primary_holdout_summary.pdf
    fig3_pcmci_graphs.pdf
    fig4_varlingam_heatmaps.pdf
    fig5_ace_comparison.pdf
    fig6_counterfactual_cdfs.pdf
```

Build commands:

```bash
cd paper
latexmk -pdf main.tex
```

Fallback if `latexmk` is unavailable:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Do not modify the NeurIPS style file. Do not use `final` or `preprint` options for anonymized review unless the workshop explicitly instructs otherwise.

## Acceptance-risk matrix

| Risk | Severity | Current status | Mitigation |
|---|---|---|---|
| Claim sounds too broad | High | Controlled in draft | Keep title/abstract PNW-specific; state pooled null |
| Low algorithmic novelty | High | Acknowledged | Frame as applied causal climate workflow + holdout validation |
| Causal assumptions challenged | High | Partially mitigated | Put identification DAG/backbone assumptions in main text; avoid graph-recovery claims |
| Reviewer doubts robustness | Medium | Gate C and holdout strong | Lead with holdout and placebo; disclose NDJF failure |
| Climate relevance unclear | Medium | Needs citations | Add PNW hydroclimate / extremes motivation and references |
| Too many artifacts / unfocused story | Medium | Current repo is broad | Main paper uses only Figure 1, Figure 2, Table 1; move rest to appendix |
| Template/page mismatch | Medium | No LaTeX scaffold yet | Create `paper/` scaffold with active CCAI/NeurIPS template |
| Numeric mismatch between paper and results | Medium | Avoidable | Generate numbers from `results/manuscript_brief.json` or copy only from current artifacts |
| Counterfactual overinterpretation | Medium | Guardrails present | Keep `do(ENSO=0)` as appendix diagnostic |
| No external dataset validation | Low-medium | Known limitation | State explicitly; temporal holdout is internal validation, not external replication |

## Must-have items before final paper generation

- [ ] Confirm the exact CCAI year/template/page limit.
- [ ] Create `paper/` scaffold with the correct style file.
- [ ] Populate `references.bib` with the citation pack.
- [ ] Convert `docs/manuscript_draft.md` into a 4-page LaTeX outline, not a long report.
- [ ] Use `figures/fig1_domain_map.pdf` and `figures/fig7_primary_holdout_summary.pdf` as the only required main figures.
- [ ] Add one compact Table 1 for estimand/data/holdout/robustness.
- [ ] Run `make pipeline` immediately before freezing final numbers.
- [ ] Confirm `results/verification_report.json` is `PASS`.
- [ ] Confirm `results/holdout_validation.json` has `strict_statistical_replication: true`.
- [ ] Check every numeric claim against `results/` artifacts.

## Optional improvements

- [ ] Add a concise appendix table summarizing exploratory experiments from `results/experiments_index.json`.
- [ ] Add a small methods schematic for the discovery-to-identification pipeline.
- [ ] Add external validation in a second reanalysis product if time permits.
- [ ] Add a better calibrated sensitivity analysis than the current crude gamma block.
- [ ] Add a short author contribution / data availability statement if required by the venue.

## Go / no-go gates

Proceed to final paper generation only if all go gates pass:

| Gate | Requirement | Source |
|---|---|---|
| G1: Pipeline | `make pipeline` completes | terminal output / current artifacts |
| G2: Verification | `status: PASS`, `error_count: 0` | `results/verification_report.json` |
| G3: Primary result | PNW full-sample IPW and DR CIs exclude zero | `results/ace_all_regions.json` |
| G4: Holdout | `strict_statistical_replication: true` | `results/holdout_validation.json` |
| G5: Claim discipline | Abstract/title do not imply western-US-wide effect | manual review |
| G6: Figure package | Main figures are `fig1` and `fig7` | `figures/` |
| G7: Citations | All required citation groups represented | `paper/references.bib` |
| G8: Template | Correct CCAI/NeurIPS style and page budget | `paper/main.tex` |

If G1-G4 fail, do not submit a positive causal-claim paper. If G5-G8 fail, do not submit until packaging is fixed.

## Confidence estimate

With the current artifacts and a disciplined 4-page writeup, this can plausibly be a competitive CCAI workshop paper. The strongest acceptance factors are:

- climate adaptation relevance,
- transparent causal claim,
- temporal holdout replication,
- explicit null results,
- reproducible pipeline.

The weakest acceptance factors are:

- limited algorithmic novelty,
- reliance on identification assumptions,
- single-reanalysis validation,
- monthly aggregation rather than storm-scale dynamics.

The path to a high-confidence submission is not more specification search. It is a concise, honest, well-cited paper that makes the PNW claim easy to evaluate.
