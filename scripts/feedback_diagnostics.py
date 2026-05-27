#!/usr/bin/env python3
"""
Targeted diagnostics responding to reviewer/course feedback.

Outputs:
  results/baseline_logistic.json
  results/backbone_sensitivity.json
  results/projection_lag_sensitivity.json
  results/holdout_interval_audit.json

These are PNW-only stress checks for the frozen ONDJFM primary estimand. They are
not a new search grid.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from causal_precip import load_config, processed_path, results_path  # noqa: E402
from causal_precip.inference import (  # noqa: E402
    block_bootstrap_ci,
    consensus_records_to_panel_dag_edges,
    estimate_ace_dr,
    estimate_ace_ipw,
    filter_panel_dataframe_by_month,
    identification_dag_edges,
    parse_backbone_edges_yaml,
)


REGION = "pacific_northwest"
TREATMENT = "sst_warm"
OUTCOME = "tp_extreme"
SENS_N_BOOT = 300


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return not (lo <= 0.0 <= hi)


def _load_panel(cfg: dict, seasonal: bool = True) -> pd.DataFrame:
    ds = xr.open_dataset(processed_path(f"panel_{REGION}.nc", cfg))
    try:
        df = ds.to_dataframe().dropna()
    finally:
        ds.close()
    if TREATMENT not in df.columns:
        df[TREATMENT] = (df["sst"] >= 0.5).astype(int)
    if seasonal:
        df = filter_panel_dataframe_by_month(df, (cfg.get("inference") or {}).get("season_months"))
    return df.copy()


def _time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        if "time" in idx.names:
            return pd.DatetimeIndex(pd.to_datetime(idx.get_level_values("time")))
        return pd.DatetimeIndex(pd.to_datetime(idx.get_level_values(0)))
    return pd.DatetimeIndex(pd.to_datetime(idx))


def _fit_logit_rd(df: pd.DataFrame, covariates: list[str], n_boot: int, block_size: int) -> dict[str, Any]:
    import statsmodels.api as sm

    cols = [TREATMENT] + covariates
    work = df[[OUTCOME] + cols].dropna().copy()

    def _fit(d: pd.DataFrame) -> tuple[float, float]:
        x = sm.add_constant(d[cols].astype(float), has_constant="add")
        y = d[OUTCOME].astype(float)
        res = sm.Logit(y, x).fit(disp=False)
        x1 = x.copy()
        x0 = x.copy()
        x1[TREATMENT] = 1.0
        x0[TREATMENT] = 0.0
        rd = float(np.mean(res.predict(x1) - res.predict(x0)))
        odds_ratio = float(np.exp(res.params[TREATMENT]))
        return rd, odds_ratio

    rd, odds_ratio = _fit(work)

    def _rd_only(d: pd.DataFrame) -> float:
        try:
            return _fit(d)[0]
        except Exception:
            return float("nan")

    ci_low, ci_high = block_bootstrap_ci(_rd_only, work, n_boot=n_boot, block_size=block_size)
    valid = [OUTCOME, TREATMENT] + covariates
    crude_p1 = float(work.loc[work[TREATMENT] == 1, OUTCOME].mean())
    crude_p0 = float(work.loc[work[TREATMENT] == 0, OUTCOME].mean())
    return {
        "covariates": covariates,
        "n_rows": int(len(work)),
        "risk_difference": rd,
        "ci95": [ci_low, ci_high],
        "ci_excludes_zero": _ci_excludes_zero(ci_low, ci_high),
        "odds_ratio": odds_ratio,
        "crude_treated_rate": crude_p1,
        "crude_control_rate": crude_p0,
        "columns_used": valid,
    }


def _load_structural_edges(cfg: dict) -> list[tuple[str, str]]:
    path = results_path(f"consensus_dag_{REGION}.json", cfg)
    with open(path) as f:
        dag = json.load(f)
    return consensus_records_to_panel_dag_edges(dag["consensus_edges"])


def _identify_adjustment_set(dag_edges: list[tuple[str, str]], all_vars: list[str]) -> list[str]:
    path = _REPO / "scripts" / "04_causal_inference.py"
    spec = importlib.util.spec_from_file_location("causal_inference_04", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    raw = mod.identify_adjustment_set(dag_edges, "sst", OUTCOME, all_vars)
    drop = {"sst", OUTCOME, TREATMENT, "nino_hot"}
    adj = [v for v in raw if v in all_vars and v not in drop]
    return adj or [v for v in ["z500", "t2m", "swvl1"] if v in all_vars]


def _estimate_sensitivity(df: pd.DataFrame, covariates: list[str], n_boot: int, block_size: int) -> dict[str, Any]:
    ipw = estimate_ace_ipw(
        df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        covariates=covariates,
        n_boot=n_boot,
        block_size=block_size,
    )
    dr = estimate_ace_dr(
        df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        covariates=covariates,
        n_boot=n_boot,
        block_size=block_size,
    )
    return {
        "adjustment_set": covariates,
        "n_rows": int(len(df)),
        "ipw": {
            **ipw,
            "ci_excludes_zero": _ci_excludes_zero(float(ipw["ci_low"]), float(ipw["ci_high"])),
        },
        "dr": {
            **dr,
            "ci_excludes_zero": _ci_excludes_zero(float(dr["ci_low"]), float(dr["ci_high"])),
        },
    }


def baseline_logistic(cfg: dict) -> dict[str, Any]:
    df = _load_panel(cfg)
    block_size = int((cfg.get("inference") or {}).get("block_size", 12))
    specs = {
        "crude": [],
        "graph_adjusted": ["z500"],
        "expanded_observed": ["z500", "nino34", "t2m", "swvl1"],
    }
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "estimand": _estimand_meta(cfg),
        "models": {
            name: _fit_logit_rd(df, [c for c in covs if c in df.columns], SENS_N_BOOT, block_size)
            for name, covs in specs.items()
        },
        "interpretation": "Baseline logistic models compare the causal workflow against simple associational models; they do not replace the identification analysis.",
    }


def backbone_sensitivity(cfg: dict) -> dict[str, Any]:
    df = _load_panel(cfg)
    block_size = int((cfg.get("inference") or {}).get("block_size", 12))
    structural = _load_structural_edges(cfg)
    base_backbone = parse_backbone_edges_yaml((cfg.get("identification_backbone") or {}).get("edges"))
    all_vars = list(df.columns)
    graph_variants = {
        "base_graph_identified": base_backbone,
        "add_nino34_to_tp": [*base_backbone, ("nino34", "tp")],
        "add_z500_to_sst": [*base_backbone, ("z500", "sst")],
        "remove_nino34_to_sst": [e for e in base_backbone if e != ("nino34", "sst")],
    }
    adjustment_overrides = {
        "override_z500_nino34": ["z500", "nino34"],
        "override_z500_t2m": ["z500", "t2m"],
        "override_broad_observed": ["z500", "nino34", "t2m", "swvl1"],
    }
    variants: dict[str, Any] = {}
    for name, backbone in graph_variants.items():
        id_edges = identification_dag_edges(structural, backbone)
        adj = _identify_adjustment_set(id_edges, all_vars)
        variants[name] = {
            "type": "graph_variant",
            "backbone_edges": [list(e) for e in backbone],
            **_estimate_sensitivity(df, adj, SENS_N_BOOT, block_size),
        }
    for name, covs in adjustment_overrides.items():
        adj = [c for c in covs if c in df.columns]
        variants[name] = {
            "type": "adjustment_override",
            **_estimate_sensitivity(df, adj, SENS_N_BOOT, block_size),
        }

    base = variants["base_graph_identified"]["ipw"]["ate"]
    for v in variants.values():
        v["same_ipw_sign_as_base"] = bool(np.sign(float(v["ipw"]["ate"])) == np.sign(float(base)))
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "estimand": _estimand_meta(cfg),
        "n_boot": SENS_N_BOOT,
        "variants": variants,
        "interpretation": "Backbone sensitivity probes whether the PNW result survives plausible graph and adjustment-set perturbations.",
    }


def lag_sensitivity(cfg: dict) -> dict[str, Any]:
    df0 = _load_panel(cfg, seasonal=False)
    for col in ("sst", "z500", "nino34", "tp"):
        if col in df0.columns:
            df0[f"{col}_lag1"] = df0[col].shift(1)
    df = filter_panel_dataframe_by_month(df0, (cfg.get("inference") or {}).get("season_months")).dropna()
    block_size = int((cfg.get("inference") or {}).get("block_size", 12))
    specs = {
        "base_z500": ["z500"],
        "add_sst_lag1": ["z500", "sst_lag1"],
        "lagged_circulation_enso": ["z500", "z500_lag1", "nino34_lag1"],
        "autoregressive_tp_lag1": ["z500", "tp_lag1"],
    }
    variants = {
        name: _estimate_sensitivity(df, [c for c in covs if c in df.columns], SENS_N_BOOT, block_size)
        for name, covs in specs.items()
    }
    base = variants["base_z500"]["ipw"]["ate"]
    for v in variants.values():
        v["same_ipw_sign_as_base"] = bool(np.sign(float(v["ipw"]["ate"])) == np.sign(float(base)))
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "estimand": _estimand_meta(cfg),
        "n_boot": SENS_N_BOOT,
        "variants": variants,
        "interpretation": "Lag sensitivity checks whether the PNW signal depends on the lag-0 panel projection. Lagged outcome adjustment changes the estimand and is treated as a stress check.",
    }


def holdout_audit(cfg: dict) -> dict[str, Any]:
    path = results_path("holdout_validation.json", cfg)
    with open(path) as f:
        hv = json.load(f)
    ipw = hv["holdout"]["ipw"]
    dr = hv["holdout"]["dr"]
    ipw_width = float(ipw["ci_high"]) - float(ipw["ci_low"])
    dr_width = float(dr["ci_high"]) - float(dr["ci_low"])
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "source": str(path),
        "ipw_ci_width": ipw_width,
        "dr_ci_width": dr_width,
        "width_ratio_ipw_over_dr": float(ipw_width / dr_width) if dr_width else None,
        "ipw_overlap": ipw.get("overlap"),
        "dr_policy": "LinearDRLearner nuisance models are fit on the train window only; the reported CI bootstraps evaluation rows with nuisance models fixed.",
        "ipw_policy": "Propensity scores are fit on the train window and held fixed for evaluation-row bootstrap resamples.",
        "interpretation": "The DR interval is much tighter because its held-out bootstrap varies evaluation rows through a fixed outcome-effect model, while IPW remains sensitive to inverse-propensity weights. This supports reporting both estimators and describing DR as model-dependent stabilization, not a free robustness guarantee.",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _estimand_meta(cfg: dict) -> dict[str, Any]:
    return {
        "region": REGION,
        "season_months": (cfg.get("inference") or {}).get("season_months"),
        "treatment": TREATMENT,
        "outcome": OUTCOME,
        "primary_adjustment": ["z500"],
        "sensitivity_n_boot": SENS_N_BOOT,
    }


def _write(name: str, data: dict[str, Any], cfg: dict) -> None:
    path = results_path(name, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonify(data), f, indent=2)
    print(json.dumps({"wrote": str(path)}, indent=2))


def main() -> int:
    cfg = load_config()
    _write("baseline_logistic.json", baseline_logistic(cfg), cfg)
    _write("backbone_sensitivity.json", backbone_sensitivity(cfg), cfg)
    _write("projection_lag_sensitivity.json", lag_sensitivity(cfg), cfg)
    _write("holdout_interval_audit.json", holdout_audit(cfg), cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
