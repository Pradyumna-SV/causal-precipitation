#!/usr/bin/env python3
"""
Stack regional monthly panels into one western-US dataset and estimate ACE of
binary warm SST (≥ 0.5 K anomaly) on tp_extreme with IPW + DR.

- Covariates: z500, t2m, swvl1, nino34 (when present) plus one-hot region dummies.
- Uncertainty: block bootstrap resamples **calendar months** (cluster_time_col)
  holding all regions for each sampled month.

Output: results/pooled_ace.json (under cfg paths.results).

Usage:  python scripts/pooled_ace.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from causal_precip import load_config, processed_path, results_path  # noqa: E402
from causal_precip.inference import (  # noqa: E402
    estimate_ace_dr,
    estimate_ace_ipw,
    filter_panel_dataframe_by_month,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TREATMENT_BIN = "sst_warm"
OUTCOME = "tp_extreme"


def load_panel(region: str, cfg: dict) -> pd.DataFrame:
    path = processed_path(f"panel_{region}.nc", cfg)
    ds = xr.open_dataset(path)
    try:
        return ds.to_dataframe().dropna()
    finally:
        ds.close()


def main() -> int:
    cfg = load_config()
    inf = cfg.get("inference") or {}
    n_boot = int(inf.get("n_boot", 2000))
    block_size = int(inf.get("block_size", 12))
    season_months = inf.get("season_months")
    regions = list(cfg["regions"].keys())

    frames: list[pd.DataFrame] = []
    for region in regions:
        df0 = load_panel(region, cfg)
        df = filter_panel_dataframe_by_month(df0, season_months)
        df = df.reset_index()
        if "time" not in df.columns:
            raise ValueError(f"{region}: expected 'time' column after reset_index()")
        df["_time"] = pd.to_datetime(df["time"])
        df["region"] = region
        df[TREATMENT_BIN] = (df["sst"] >= 0.5).astype(int)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    ref = "california"
    if ref not in regions:
        ref = regions[0]
    for r in regions:
        if r != ref:
            df_all[f"reg__{r}"] = (df_all["region"] == r).astype(int)
    reg_cols = [f"reg__{r}" for r in regions if r != ref]
    base_cov = ["z500", "t2m", "swvl1", "nino34"]
    covariates = [c for c in base_cov if c in df_all.columns] + reg_cols
    log.info("Pooled rows=%d, covariates=%s", len(df_all), covariates)

    ace_ipw = estimate_ace_ipw(
        df_all,
        treatment=TREATMENT_BIN,
        outcome=OUTCOME,
        covariates=covariates,
        n_boot=n_boot,
        block_size=block_size,
        cluster_time_col="_time",
    )
    ace_dr = estimate_ace_dr(
        df_all,
        treatment=TREATMENT_BIN,
        outcome=OUTCOME,
        covariates=covariates,
        n_boot=n_boot,
        block_size=block_size,
        cluster_time_col="_time",
    )
    log.info("Pooled IPW ATE=%.4f [%.4f, %.4f]", ace_ipw["ate"], ace_ipw["ci_low"], ace_ipw["ci_high"])
    log.info("Pooled DR  ATE=%.4f [%.4f, %.4f]", ace_dr["ate"], ace_dr["ci_low"], ace_dr["ci_high"])

    out = {
        "schema_version": 1,
        "regions_stacked": regions,
        "reference_region_dummy_omitted": ref,
        "treatment": TREATMENT_BIN,
        "outcome": OUTCOME,
        "covariates": covariates,
        "inference_meta": {
            "season_months": season_months,
            "n_boot": n_boot,
            "block_size": block_size,
            "n_rows": len(df_all),
            "cluster_bootstrap_time_col": "_time",
        },
        "ipw": ace_ipw,
        "dr": ace_dr,
    }
    path = results_path("pooled_ace.json", cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"wrote": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
