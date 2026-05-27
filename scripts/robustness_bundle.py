#!/usr/bin/env python3
"""
Build results/robustness_report.json — overlap / placebo / estimator agreement /
optional alternate-season artifact checks (gate C helpers for docs/analysis_plan.md).

Does NOT hard-fail the pipeline on weak evidence; records pass/fail booleans.

Usage:  .venv/bin/python scripts/robustness_bundle.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from causal_precip import load_config, processed_path, results_path  # noqa: E402
from causal_precip.inference import (  # noqa: E402
    estimate_ace_ipw,
    filter_panel_dataframe_by_month,
    trim_by_propensity_quantiles,
)


def load_panel(region: str, cfg: dict) -> pd.DataFrame:
    path = processed_path(f"panel_{region}.nc", cfg)
    ds = xr.open_dataset(path)
    try:
        return ds.to_dataframe().dropna()
    finally:
        ds.close()


def _placebo_shuffle_within_month(
    df: pd.DataFrame,
    treatment: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Permute binary treatment within calendar month (destroys SST–outcome link)."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        months = out.index.month.values
    else:
        months = pd.to_datetime(out.index).month.values
    t = out[treatment].values.astype(int).copy()
    for m in np.unique(months):
        idx = np.where(months == m)[0]
        t[idx] = t[idx][rng.permutation(len(idx))]
    out[treatment] = t
    return out


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return not (lo <= 0.0 <= hi)


def _jsonify(obj: Any) -> Any:
    """Convert numpy scalars for json.dump."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> int:
    cfg = load_config()
    reg_path = _REPO / "config" / "analysis_registry.yaml"
    with open(reg_path) as f:
        registry = yaml.safe_load(f) or {}
    gate_region = (registry.get("primary") or {}).get("gate_b_region", "pacific_northwest")

    inf = cfg.get("inference") or {}
    n_boot = int(inf.get("robustness_n_boot", 500))
    block_size = int(inf.get("block_size", 12))
    season_months = inf.get("season_months")

    rdir = results_path("", cfg).resolve()
    ace_path = rdir / "ace_all_regions.json"
    if not ace_path.exists():
        print(f"Missing {ace_path}; run inference first.", file=sys.stderr)
        return 1
    with open(ace_path) as f:
        ace_all = json.load(f)
    if gate_region not in ace_all:
        print(f"Region {gate_region!r} not in ace_all_regions.json", file=sys.stderr)
        return 1
    ace = ace_all[gate_region]
    adj_set = list(ace.get("adjustment_set") or [])
    sst_bin = ace.get("treatment") or "sst_warm"
    outcome = ace.get("outcome") or "tp_extreme"

    df0 = load_panel(gate_region, cfg)
    df = filter_panel_dataframe_by_month(df0, season_months)
    if sst_bin not in df.columns:
        df[sst_bin] = (df["sst"] >= 0.5).astype(int)

    trim_meta = (ace.get("inference_meta") or {}).get("propensity_trim")
    if trim_meta and trim_meta.get("quantiles"):
        q = (float(trim_meta["quantiles"][0]), float(trim_meta["quantiles"][1]))
        df_work, _trim = trim_by_propensity_quantiles(df, sst_bin, adj_set, q)
    else:
        df_work = df

    ipw = ace.get("ipw") or {}
    dr = ace.get("dr") or {}
    overlap = ipw.get("overlap") or {}
    ess_n = float(overlap.get("ess_over_n", 0.0) or 0.0)
    overlap_pass = ess_n >= 0.12

    rng = np.random.default_rng(42)
    df_ph = _placebo_shuffle_within_month(df_work, sst_bin, rng)
    ph = estimate_ace_ipw(
        df_ph,
        treatment=sst_bin,
        outcome=outcome,
        covariates=adj_set,
        n_boot=n_boot,
        block_size=block_size,
    )
    placebo_pass = not _ci_excludes_zero(float(ph["ci_low"]), float(ph["ci_high"]))

    sign_agree = np.sign(float(ipw.get("ate", 0.0))) == np.sign(float(dr.get("ate", 0.0)))
    ci_overlap = not (
        float(ipw.get("ci_high", -1.0)) < float(dr.get("ci_low", 0.0))
        or float(dr.get("ci_high", -1.0)) < float(ipw.get("ci_low", 0.0))
    )
    agree_pass = bool(sign_agree and ci_overlap)

    season_path = rdir / "grid" / "season_ndjf" / "ace_all_regions.json"
    season_check: dict[str, Any] = {"status": "skipped_no_artifact", "pass": True}
    if season_path.is_file():
        with open(season_path) as f:
            alt_ace = json.load(f)
        if gate_region in alt_ace:
            a0 = float(ipw.get("ate", 0.0))
            a1 = float((alt_ace[gate_region].get("ipw") or {}).get("ate", 0.0))
            season_check = {
                "status": "compared",
                "pass": np.sign(a0) == np.sign(a1) or abs(a0 * a1) < 1e-8,
                "baseline_ipw_ate": a0,
                "season_ndjf_ipw_ate": a1,
                "artifact": str(season_path),
            }
        else:
            season_check = {
                "status": "artifact_missing_region",
                "pass": False,
                "artifact": str(season_path),
            }

    optional_checks = [
        ("overlap_ess", overlap_pass),
        ("placebo_month_shuffle_ipw_ci_includes_zero", placebo_pass),
        ("dr_ipw_ci_overlap_same_sign", agree_pass),
    ]
    if season_check.get("status") not in ("skipped_no_artifact",):
        optional_checks.append(("season_ndjf_sign_stable", bool(season_check.get("pass"))))

    n_pass = sum(1 for _n, p in optional_checks if p)
    gate_c_pass = n_pass >= 2

    gate_b_ipw = _ci_excludes_zero(float(ipw.get("ci_low")), float(ipw.get("ci_high")))
    gate_b_dr = _ci_excludes_zero(float(dr.get("ci_low")), float(dr.get("ci_high")))

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gate_b_region": gate_region,
        "gates": {
            "B_statistical_strength_informational": {
                "ipw_ci_excludes_zero": gate_b_ipw,
                "dr_ci_excludes_zero": gate_b_dr,
                "note": "Not used by verify_pipeline; informational for analysis_plan gate B.",
            },
            "C_identification_stress": {
                "pass": gate_c_pass,
                "checks_passed_count": n_pass,
                "checks_required_min": 2,
            },
            "D_multiplicity": {
                "note": "See config/analysis_registry.yaml and manuscript_brief multiplicity block.",
            },
        },
        "checks": {
            "overlap": {"pass": overlap_pass, "ess_over_n": ess_n, **overlap},
            "placebo_ipw_month_shuffle": {
                "pass": placebo_pass,
                "ate": ph.get("ate"),
                "ci95": [ph.get("ci_low"), ph.get("ci_high")],
                "n_boot": n_boot,
            },
            "dr_ipw_agreement": {
                "pass": agree_pass,
                "sign_agree": sign_agree,
                "ci_overlap": ci_overlap,
            },
            "season_ndjf_alternate_artifact": season_check,
        },
        "optional_checks_list": [{"name": n, "pass": p} for n, p in optional_checks],
    }
    report = _jsonify(report)

    out = results_path("robustness_report.json", cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"wrote": str(out), "gate_c_pass": gate_c_pass, "checks_passed": n_pass}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
