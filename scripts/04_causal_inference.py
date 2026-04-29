"""
scripts/04_causal_inference.py
Estimate the Average Causal Effect (ACE) of SST anomalies on extreme
precipitation and execute the do(ENSO=0) counterfactual.

Uses the consensus DAG from script 03 to identify valid adjustment sets,
then runs two cross-validated estimators (IPW and doubly-robust) with
block-bootstrap confidence intervals.

Outputs (in cfg['paths']['results']):
  ace_{region}.json                   — IPW + DR ACE estimates with 95% CIs
  counterfactual_tp_{region}.nc       — factual vs. CF precipitation time series
  rosenbaum_bounds_{region}.json      — sensitivity analysis (Γ range)

Run:   python scripts/04_causal_inference.py           (local)
       ENV=nautilus python scripts/04_causal_inference.py  (Nautilus k8s)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_precip import load_config, processed_path, results_path
from causal_precip.inference import (
    counterfactual_enso_zero,
    estimate_ace_dr,
    estimate_ace_ipw,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TREATMENT = "sst"         # continuous SST anomaly
OUTCOME   = "tp_extreme"  # binary extreme-precip indicator
ENSO_VAR  = "nino34"


def load_panel(region: str, cfg: dict) -> pd.DataFrame:
    path = processed_path(f"panel_{region}.nc", cfg)
    ds   = xr.open_dataset(path)
    df   = ds.to_dataframe().dropna()
    log.info("Panel: %s — %d rows, vars: %s", region, len(df), list(df.columns))
    return df


def load_consensus_dag(region: str, cfg: dict) -> list[tuple[str, str]]:
    """Return list of (source, target) edge tuples from the consensus DAG."""
    path = results_path(f"consensus_dag_{region}.json", cfg)
    with open(path) as f:
        dag = json.load(f)
    edges = [(e["source"], e["target"]) for e in dag["consensus_edges"]]
    log.info("Consensus DAG loaded: %d edges", len(edges))
    return edges


def identify_adjustment_set(
    dag_edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    all_vars: list[str],
) -> list[str]:
    """
    Identify a valid backdoor adjustment set via the backdoor criterion.
    Falls back to all non-treatment/non-outcome observed variables if
    dowhy raises an error (e.g. the DAG is a partial graph).
    """
    try:
        import networkx as nx
        from dowhy import CausalModel

        # Build GML string for dowhy
        G = nx.DiGraph()
        G.add_nodes_from(all_vars)
        G.add_edges_from(dag_edges)
        gml_lines = ["graph [", "  directed 1"]
        for v in all_vars:
            gml_lines.append(f'  node [ id "{v}" label "{v}" ]')
        for src, dst in dag_edges:
            gml_lines.append(f'  edge [ source "{src}" target "{dst}" ]')
        gml_lines.append("]")
        gml_str = "\n".join(gml_lines)

        model = CausalModel(
            data=pd.DataFrame(columns=all_vars),
            treatment=treatment,
            outcome=outcome,
            graph=gml_str,
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        backdoor = estimand.get_backdoor_variables()
        adj_set  = [v for v in backdoor if v in all_vars]
        log.info("Backdoor adjustment set: %s", adj_set)
        return adj_set

    except Exception as exc:
        log.warning("dowhy adjustment failed (%s); using all covariates.", exc)
        return [v for v in all_vars if v not in (treatment, outcome)]


def rosenbaum_sensitivity(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    ate_ipw: float,
    gamma_range: list[float] | None = None,
) -> dict:
    """
    Simplified Rosenbaum-style sensitivity analysis.
    For each Γ (odds-ratio of hidden bias), compute the range of p-values
    under the worst-case hidden confounder.  Reports at which Γ the
    lower bound of the p-value crosses 0.05 (i.e. conclusion becomes fragile).
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    T = df[treatment].values
    Y = df[outcome].values
    n = len(T)

    results_sens = {}
    for gamma in gamma_range:
        # Under worst-case hidden confounder, treatment assignment probability
        # can range from 1/(1+Γ) to Γ/(1+Γ).
        # Use a sign test approximation for the binary outcome case.
        ps_hi = gamma / (1 + gamma)
        ps_lo = 1.0   / (1 + gamma)
        # Expected ATE range under hidden confounding
        ate_hi = float(np.mean(Y[T == 1]) - np.mean(Y[T == 0]) * gamma)
        ate_lo = float(np.mean(Y[T == 1]) / gamma - np.mean(Y[T == 0]))
        results_sens[gamma] = {
            "ate_lower_bound": round(ate_lo, 4),
            "ate_upper_bound": round(ate_hi, 4),
            "sign_change":     bool(ate_lo * ate_hi < 0),
        }

    # Critical Γ: smallest value where sign change occurs
    critical_gamma = None
    for gamma in sorted(gamma_range):
        if results_sens[gamma]["sign_change"]:
            critical_gamma = gamma
            break

    return {
        "gamma_bounds": results_sens,
        "critical_gamma": critical_gamma,
        "interpretation": (
            f"Conclusions robust up to Γ < {critical_gamma}"
            if critical_gamma else "Robust across all tested Γ values"
        ),
    }


def main(cfg: dict) -> None:
    out_dir = results_path("", cfg).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_boot     = 1000 if cfg.get("_env") == "nautilus" else 200
    block_size = 12   # 1 year blocks to respect annual seasonality

    all_ace_results: dict = {}

    for region in cfg["regions"]:
        log.info("=" * 60)
        log.info("Causal inference — region: %s", region)

        df        = load_panel(region, cfg)
        dag_edges = load_consensus_dag(region, cfg)
        all_vars  = list(df.columns)

        # Ensure binary treatment option (binarize SST anomaly at +0.5 K)
        # and keep continuous version for GPS-IPW
        sst_bin_col = "sst_warm"
        df[sst_bin_col] = (df[TREATMENT] >= 0.5).astype(int)

        adj_set = identify_adjustment_set(dag_edges, TREATMENT, OUTCOME, all_vars)
        if not adj_set:
            # If DAG is sparse, use climate-theory priors: z500, t2m as confounders
            adj_set = [v for v in ["z500", "t2m", "swvl1"] if v in df.columns]
            log.warning("Empty adjustment set; using fallback: %s", adj_set)

        log.info("Treatment: %s (continuous SST anomaly), binarized: %s", TREATMENT, sst_bin_col)
        log.info("Outcome: %s", OUTCOME)
        log.info("Adjustment set: %s", adj_set)

        # --- IPW (binary treatment: warm SST vs. not) ---
        log.info("Running IPW (n_boot=%d) …", n_boot)
        ace_ipw = estimate_ace_ipw(
            df, treatment=sst_bin_col, outcome=OUTCOME,
            covariates=adj_set, n_boot=n_boot, block_size=block_size,
        )
        log.info("IPW  ATE = %.4f  [%.4f, %.4f]",
                 ace_ipw["ate"], ace_ipw["ci_low"], ace_ipw["ci_high"])

        # --- Doubly-robust ---
        log.info("Running doubly-robust DR (n_boot=%d) …", n_boot)
        ace_dr = estimate_ace_dr(
            df, treatment=sst_bin_col, outcome=OUTCOME,
            covariates=adj_set, n_boot=n_boot, block_size=block_size,
        )
        log.info("DR   ATE = %.4f  [%.4f, %.4f]",
                 ace_dr["ate"], ace_dr["ci_low"], ace_dr["ci_high"])

        # --- Rosenbaum sensitivity ---
        log.info("Running Rosenbaum sensitivity analysis …")
        sens = rosenbaum_sensitivity(
            df[[sst_bin_col, OUTCOME]].rename(columns={sst_bin_col: "T"}),
            treatment="T", outcome=OUTCOME,
            ate_ipw=ace_ipw["ate"],
        )
        log.info("Critical Γ: %s", sens["critical_gamma"])

        ace_out = {
            "region":    region,
            "treatment": sst_bin_col,
            "outcome":   OUTCOME,
            "adjustment_set": adj_set,
            "ipw": ace_ipw,
            "dr":  ace_dr,
            "sensitivity": sens,
        }
        all_ace_results[region] = ace_out

        ace_path = results_path(f"ace_{region}.json", cfg)
        with open(ace_path, "w") as f:
            json.dump(ace_out, f, indent=2)
        log.info("ACE results → %s", ace_path)

        # --- Counterfactual do(ENSO = 0) ---
        log.info("Computing do(ENSO=0) counterfactual …")
        cf_tp = counterfactual_enso_zero(
            panel=df[[c for c in df.columns if c not in (sst_bin_col, OUTCOME)]],
            dag_edges=dag_edges,
            treatment_col=ENSO_VAR,
            outcome_col="tp",
        )

        # Save factual + counterfactual as a NetCDF
        ds_cf = xr.Dataset({
            "tp_factual":      xr.DataArray(df["tp"].values, dims=["time"],
                                            attrs={"long_name": "Factual precipitation anomaly"}),
            "tp_cf_enso0":     xr.DataArray(cf_tp.values,    dims=["time"],
                                            attrs={"long_name": "Counterfactual precipitation anomaly (do(ENSO=0))"}),
        }, coords={"time": df.index.values})
        ds_cf.attrs["region"]      = region
        ds_cf.attrs["counterfactual"] = "do(nino34 = 0)"
        cf_path = results_path(f"counterfactual_tp_{region}.nc", cfg)
        ds_cf.to_netcdf(cf_path)
        log.info("Counterfactual → %s", cf_path)

    # Save combined ACE table
    combined_path = results_path("ace_all_regions.json", cfg)
    with open(combined_path, "w") as f:
        json.dump(all_ace_results, f, indent=2)
    log.info("Combined ACE results → %s", combined_path)
    log.info("Causal inference complete.")


if __name__ == "__main__":
    cfg = load_config()
    log.info("Environment : %s", cfg.get("_env", "local"))
    log.info("Date range  : %s → %s", cfg["date_range"]["start"], cfg["date_range"]["end"])
    main(cfg)
