#!/usr/bin/env python3
"""
Build results/manuscript_brief.json — a machine-readable summary aligned with a
NeurIPS CCAI-style paper: estimand, headline numbers, integrity, and guardrails.

Run after preprocess → discovery → inference → figures → verify (verify should PASS).

Usage:  python scripts/manuscript_brief.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from causal_precip import load_config, results_path  # noqa: E402


def _ci_crosses_zero(lo: float, hi: float) -> bool:
    return lo <= 0.0 <= hi


def main() -> int:
    cfg = load_config()
    reg_path = _REPO / "config" / "analysis_registry.yaml"
    registry: dict = {}
    if reg_path.exists():
        with open(reg_path) as f:
            registry = yaml.safe_load(f) or {}

    primary_preset = (registry.get("primary") or {}).get("preset_id", "primary")
    preset_name = cfg.get("_preset")
    if preset_name is None or preset_name == primary_preset:
        analysis_tier = "primary"
    else:
        analysis_tier = "exploratory"

    rdir = results_path("", cfg).resolve()
    vpath = rdir / "verification_report.json"
    apath = rdir / "ace_all_regions.json"
    robust_path = rdir / "robustness_report.json"
    pooled_path = rdir / "pooled_ace.json"
    holdout_path = rdir / "holdout_validation.json"
    baseline_path = rdir / "baseline_logistic.json"
    backbone_path = rdir / "backbone_sensitivity.json"
    lag_path = rdir / "projection_lag_sensitivity.json"
    holdout_audit_path = rdir / "holdout_interval_audit.json"
    external_scope_path = rdir / "external_validation_scope.json"

    verification = {}
    if vpath.exists():
        with open(vpath) as f:
            verification = json.load(f)

    if not apath.exists():
        print(f"Missing {apath}; run 04 first.", file=sys.stderr)
        return 1

    with open(apath) as f:
        ace_all = json.load(f)

    regions = list(ace_all.keys())
    headline = []
    for region in regions:
        a = ace_all[region]
        ipw, dr = a.get("ipw", {}), a.get("dr", {})
        ev = a.get("e_value", {})
        alt = (a.get("alternative_estimands") or {}).get("ipw_warm_nino34_vs_not", {})
        headline.append(
            {
                "region": region,
                "n_inference_rows": (a.get("inference_meta") or {}).get("n_rows"),
                "adjustment_set": a.get("adjustment_set"),
                "ipw_risk_difference": {
                    "ate": ipw.get("ate"),
                    "ci95": [ipw.get("ci_low"), ipw.get("ci_high")],
                    "ci_crosses_zero": _ci_crosses_zero(
                        float(ipw.get("ci_low", -1.0)), float(ipw.get("ci_high", 1.0))
                    ),
                },
                "dr_risk_difference": {
                    "ate": dr.get("ate"),
                    "ci95": [dr.get("ci_low"), dr.get("ci_high")],
                    "ci_crosses_zero": _ci_crosses_zero(
                        float(dr.get("ci_low", -1.0)), float(dr.get("ci_high", 1.0))
                    ),
                },
                "e_value_adjusted_or_min": ev.get("adjusted_or_e_value_min"),
                "nino_hot_ipw": {
                    "ate": alt.get("ate"),
                    "ci95": [alt.get("ci_low"), alt.get("ci_high")],
                    "ci_crosses_zero": _ci_crosses_zero(
                        float(alt.get("ci_low", -1.0)), float(alt.get("ci_high", 1.0))
                    ),
                    "adjustment_set": alt.get("adjustment_set"),
                },
            }
        )

    inf = cfg.get("inference") or {}
    disc = cfg.get("discovery") or {}

    het_notes = []
    for h in headline:
        ex = not h["ipw_risk_difference"]["ci_crosses_zero"]
        het_notes.append(
            f"{h['region']}: primary IPW 95% CI {'excludes' if ex else 'includes'} zero"
        )

    mult = registry.get("multiplicity") or {}
    robustness_summary = None
    if robust_path.exists():
        with open(robust_path) as f:
            robustness_summary = json.load(f)
    pooled_summary = None
    if pooled_path.exists():
        with open(pooled_path) as f:
            pooled_summary = json.load(f)
    holdout_summary = None
    if holdout_path.exists():
        with open(holdout_path) as f:
            holdout_summary = json.load(f)
    baseline_summary = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_summary = json.load(f)
    backbone_summary = None
    if backbone_path.exists():
        with open(backbone_path) as f:
            backbone_summary = json.load(f)
    lag_summary = None
    if lag_path.exists():
        with open(lag_path) as f:
            lag_summary = json.load(f)
    holdout_audit_summary = None
    if holdout_audit_path.exists():
        with open(holdout_audit_path) as f:
            holdout_audit_summary = json.load(f)
    external_scope_summary = None
    if external_scope_path.exists():
        with open(external_scope_path) as f:
            external_scope_summary = json.load(f)

    brief = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "venue_target": "NeurIPS Workshop on Climate Change AI (CCAI)",
        "analysis": {
            "tier": analysis_tier,
            "active_preset": preset_name,
            "primary_preset_id": primary_preset,
            "registry_path": str(reg_path) if reg_path.exists() else None,
            "analysis_plan_doc": "docs/analysis_plan.md",
        },
        "multiplicity": {
            "strategy": mult.get("strategy"),
            "bh_fdr_target_q": mult.get("bh_fdr_target_q"),
            "exploratory_preset_ids": [e.get("id") for e in (registry.get("exploratory_presets") or [])],
        },
        "goal_one_sentence": (
            "Seek a single identification-credible estimand (warm SST → extreme precip) on ERA5 "
            "western US panels with backbone + projected discovery DAG, IPW/DR, and falsification/robustness "
            "checks—iterate only over a closed preset list (see docs/analysis_plan.md) rather than "
            "leading with spatial heterogeneity as the main claim."
        ),
        "estimand": {
            "primary_treatment": ace_all[regions[0]].get("treatment", "sst_warm"),
            "primary_outcome": ace_all[regions[0]].get("outcome", "tp_extreme"),
            "calendar_window": f'{cfg["date_range"]["start"]} — {cfg["date_range"]["end"]}',
            "discovery_months": disc.get("season_months"),
            "inference_months": inf.get("season_months"),
            "n_bootstrap": inf.get("n_bootstrap") or inf.get("n_boot"),
            "block_size_months": inf.get("block_size"),
            "identification": (
                "Pearl backdoor adjustment sets from DoWhy on identification_dag = "
                "domain backbone merged with lag-0 acyclic consensus DAG."
            ),
            "secondary": (
                "Exploratory IPW contrast for warm Niño 3.4 vs not (binary); empty "
                "backdoor uses marginal propensity—not a fully mediated causal "
                "effect without a front-door estimator."
            ),
        },
        "empirical_summary": headline,
        "spatial_variation_notes": het_notes,
        "robustness_report": robustness_summary,
        "temporal_holdout_validation": holdout_summary,
        "feedback_response_diagnostics": {
            "baseline_logistic": baseline_summary,
            "backbone_sensitivity": backbone_summary,
            "projection_lag_sensitivity": lag_summary,
            "holdout_interval_audit": holdout_audit_summary,
        },
        "external_validation_scope": external_scope_summary,
        "pooled_ace_artifact": pooled_summary,
        "panel_diagnostics": verification.get("panels"),
        "counterfactual_metrics": verification.get("counterfactual"),
        "integrity": {
            "verification_status": verification.get("status", "unknown"),
            "verification_report": str(vpath) if vpath.exists() else None,
            "expected_panel_months": verification.get("expected_months"),
        },
        "contribution_framing": verification.get("contribution_assessment", {}),
        "guardrails_for_claims": [
            "Do not assert PCMCI+/VARLiNGAM recover the true data-generating DAG.",
            "Primary causal interpretation rides on identification_assumptions (DAG + positivity), not discovery p-values.",
            "Gamma sensitivity block is crude stress on unadjusted contrasts, not Rosenbaum-style bounds.",
            "Linear SCM counterfactual is a stylized sensitivity analysis; small |Δtp| implies weak linear propagation, not absence of all ENSO effects.",
            "Treat exploratory CAUSAL_PRESET runs as multiplicity-budget items (see analysis.multiplicity); do not upgrade them to main claims without holdout or FDR plan.",
            "Temporal holdout validation is confirmatory only when strict_statistical_replication is true; directional replication alone is supporting but not decisive.",
            "Broad observed-covariate adjustment is a stress check, not the primary estimand; when it induces poor overlap or adjusts potential mediators, interpret null estimates as sensitivity evidence rather than automatic refutation.",
            "Do not claim MERRA-2/JRA-55 replication unless an external_validation_<dataset>.json artifact exists.",
        ],
    }

    out = results_path("manuscript_brief.json", cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(brief, f, indent=2)
    print(json.dumps({"wrote": str(out), "verification": brief["integrity"]["verification_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
