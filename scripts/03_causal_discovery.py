"""
scripts/03_causal_discovery.py
Run PCMCI+ (tigramite) and VARLiNGAM (lingam) on the region panels.
Reconcile the two recovered graphs into a consensus DAG.

Outputs (in cfg['paths']['results']):
  pcmci_{region}.pkl          — full tigramite PCMCI+ results dict
  pcmci_{region}_summary.json — adjacency, p-values, val_matrix (lag 1 only)
  varlingam_{region}.pkl      — fitted VARLiNGAM model
  varlingam_{region}_summary.json
  consensus_dag_{region}.json — edges present in both methods (high-confidence)

Run:   python scripts/03_causal_discovery.py           (local, ParCorr CI test)
       ENV=nautilus python scripts/03_causal_discovery.py  (Nautilus, GPDC CI test)
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_precip import load_config, processed_path, results_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Variables in the panel that enter causal discovery
# (tp_extreme is for inference; raw tp anomaly for discovery)
DISCOVERY_VARS = ["tp", "sst", "t2m", "swvl1", "z500", "u850", "v850", "nino34"]

# PCMCI+ hyperparameters
TAU_MAX   = 6       # max lag in months
PC_ALPHA  = 0.05    # significance threshold for MCI test


def load_panel(region: str, cfg: dict) -> pd.DataFrame:
    """Load the preprocessed panel NetCDF and return as a DataFrame."""
    path = processed_path(f"panel_{region}.nc", cfg)
    ds   = xr.open_dataset(path)
    available = [v for v in DISCOVERY_VARS if v in ds]
    df = ds[available].to_dataframe().dropna()
    log.info("Panel loaded: %s — %d time steps, vars: %s", region, len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# PCMCI+
# ---------------------------------------------------------------------------

def run_pcmciplus(
    df: pd.DataFrame,
    tau_max: int = TAU_MAX,
    pc_alpha: float = PC_ALPHA,
    nonlinear: bool = False,
) -> dict:
    """
    Run PCMCI+ on the panel DataFrame.
    Uses ParCorr (fast, linear) locally; GPDC (nonlinear kernel) on Nautilus.
    Returns the tigramite results dict.
    """
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    data_array = df.values.astype(float)
    var_names  = list(df.columns)

    dataframe = pp.DataFrame(data_array, var_names=var_names)

    if nonlinear:
        from tigramite.independence_tests.gpdc import GPDC
        cond_ind_test = GPDC(significance="analytic")
        log.info("Using GPDC conditional independence test (nonlinear)")
    else:
        from tigramite.independence_tests.parcorr import ParCorr
        cond_ind_test = ParCorr(significance="analytic")
        log.info("Using ParCorr conditional independence test (linear)")

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    results["var_names"] = var_names
    return results


def summarise_pcmci(results: dict, alpha: float = PC_ALPHA) -> dict:
    """Extract significant edges from PCMCI+ results."""
    var_names  = results["var_names"]
    val_matrix = results["val_matrix"]    # (n, n, tau_max+1)
    p_matrix   = results["p_matrix"]
    n_vars     = len(var_names)

    edges = []
    for j in range(n_vars):
        for i in range(n_vars):
            for tau in range(0, val_matrix.shape[2]):
                if p_matrix[i, j, tau] < alpha and abs(val_matrix[i, j, tau]) > 1e-6:
                    if i == j and tau == 0:
                        continue
                    edges.append({
                        "source": var_names[i],
                        "target": var_names[j],
                        "lag":    tau,
                        "mci":    float(val_matrix[i, j, tau]),
                        "pvalue": float(p_matrix[i, j, tau]),
                    })

    return {
        "n_significant_edges": len(edges),
        "edges": edges,
        "var_names": var_names,
        "tau_max": int(val_matrix.shape[2] - 1),
    }


# ---------------------------------------------------------------------------
# VARLiNGAM
# ---------------------------------------------------------------------------

def run_varlingam(df: pd.DataFrame, lags: int = TAU_MAX) -> object:
    """Fit a VARLiNGAM model and return the fitted model object."""
    import lingam

    data_array = df.values.astype(float)
    model      = lingam.VARLiNGAM(lags=lags, criterion="bic", prune=True)
    model.fit(data_array)
    log.info("VARLiNGAM fitted. Causal order: %s",
             [df.columns[i] for i in model.causal_order_])
    return model


def summarise_varlingam(model: object, var_names: list[str], alpha: float = PC_ALPHA) -> dict:
    """
    Extract significant directed edges from VARLiNGAM adjacency matrices.
    model.adjacency_matrices_ : list of (n_vars × n_vars) arrays, one per lag.
    Threshold by absolute coefficient > 0 (pruned model).
    """
    edges = []
    n_vars = len(var_names)
    for lag_idx, mat in enumerate(model.adjacency_matrices_):
        tau = lag_idx  # lag 0 = contemporaneous
        for j in range(n_vars):
            for i in range(n_vars):
                coef = mat[j, i]  # mat[effect, cause]
                if abs(coef) > 1e-6 and not (i == j and tau == 0):
                    edges.append({
                        "source": var_names[i],
                        "target": var_names[j],
                        "lag":    tau,
                        "coef":   float(coef),
                    })
    return {
        "n_significant_edges": len(edges),
        "edges": edges,
        "var_names": var_names,
        "causal_order": [var_names[i] for i in model.causal_order_],
    }


# ---------------------------------------------------------------------------
# Graph reconciliation
# ---------------------------------------------------------------------------

def reconcile_dags(pcmci_summary: dict, varlingam_summary: dict) -> dict:
    """
    Build a consensus DAG: edges (source → target at any lag) that appear
    in **both** PCMCI+ and VARLiNGAM are considered high-confidence.

    Also records edges unique to each method for transparency in the paper.
    """
    def _edge_set(summary: dict) -> set[tuple[str, str]]:
        return {(e["source"], e["target"]) for e in summary["edges"]}

    pcmci_edges   = _edge_set(pcmci_summary)
    varling_edges = _edge_set(varlingam_summary)

    consensus   = pcmci_edges & varling_edges
    pcmci_only  = pcmci_edges - varling_edges
    vl_only     = varling_edges - pcmci_edges

    log.info("Edge agreement: %d consensus, %d PCMCI+ only, %d VARLiNGAM only",
             len(consensus), len(pcmci_only), len(vl_only))

    # For the consensus edges, take MCI coefficient and p-value from PCMCI+
    # (better calibrated for climate data)
    pcmci_edge_info = {(e["source"], e["target"]): e for e in pcmci_summary["edges"]}
    consensus_edges = []
    for src, dst in sorted(consensus):
        info = pcmci_edge_info.get((src, dst), {"source": src, "target": dst})
        consensus_edges.append(info)

    return {
        "consensus_edges":    consensus_edges,
        "pcmci_only_edges":   [{"source": s, "target": t} for s, t in sorted(pcmci_only)],
        "varlingam_only_edges": [{"source": s, "target": t} for s, t in sorted(vl_only)],
        "n_consensus":        len(consensus),
        "n_pcmci_only":       len(pcmci_only),
        "n_varlingam_only":   len(vl_only),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    use_nonlinear = cfg.get("_env", "local") == "nautilus"
    regions       = list(cfg["regions"].keys())
    out_dir       = results_path("", cfg).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        log.info("=" * 60)
        log.info("Causal discovery — region: %s", region)

        df = load_panel(region, cfg)
        var_names = list(df.columns)

        # --- PCMCI+ ---
        log.info("Running PCMCI+ (tau_max=%d, alpha=%.2f) …", TAU_MAX, PC_ALPHA)
        pcmci_results = run_pcmciplus(df, tau_max=TAU_MAX, pc_alpha=PC_ALPHA,
                                      nonlinear=use_nonlinear)
        pcmci_summary = summarise_pcmci(pcmci_results)

        pkl_path = results_path(f"pcmci_{region}.pkl", cfg)
        with open(pkl_path, "wb") as f:
            pickle.dump(pcmci_results, f)
        json_path = results_path(f"pcmci_{region}_summary.json", cfg)
        with open(json_path, "w") as f:
            json.dump(pcmci_summary, f, indent=2)
        log.info("PCMCI+: %d significant edges → %s", pcmci_summary["n_significant_edges"], json_path)

        # --- VARLiNGAM ---
        log.info("Running VARLiNGAM (lags=%d) …", TAU_MAX)
        vl_model   = run_varlingam(df, lags=TAU_MAX)
        vl_summary = summarise_varlingam(vl_model, var_names)

        pkl_path = results_path(f"varlingam_{region}.pkl", cfg)
        with open(pkl_path, "wb") as f:
            pickle.dump(vl_model, f)
        json_path = results_path(f"varlingam_{region}_summary.json", cfg)
        with open(json_path, "w") as f:
            json.dump(vl_summary, f, indent=2)
        log.info("VARLiNGAM: %d edges → %s", vl_summary["n_significant_edges"], json_path)

        # --- Reconciliation ---
        consensus = reconcile_dags(pcmci_summary, vl_summary)
        con_path  = results_path(f"consensus_dag_{region}.json", cfg)
        with open(con_path, "w") as f:
            json.dump(consensus, f, indent=2)
        log.info("Consensus DAG: %d edges → %s", consensus["n_consensus"], con_path)

    log.info("Causal discovery complete for all regions.")


if __name__ == "__main__":
    cfg = load_config()
    log.info("Environment : %s", cfg.get("_env", "local"))
    log.info("Date range  : %s → %s", cfg["date_range"]["start"], cfg["date_range"]["end"])
    main(cfg)
