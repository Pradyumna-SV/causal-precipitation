"""
scripts/04_causal_inference.py
Estimate the Average Causal Effect (ACE) of SST anomalies on extreme
precipitation and execute the do(ENSO=0) counterfactual.

Uses the consensus DAG from script 03 plus an optional **identification backbone**
(config) to build an identification graph for adjustment / SCM. Runs IPW and
doubly-robust estimators with block-bootstrap CIs; reports crude/adjusted ratio
E-values (VanderWeele & Ding point-estimate formulae).

Outputs (in cfg['paths']['results']):
  ace_{region}.json                — IPW + DR ACE; sensitivity; e_value block
  structural_dag_{region}.json    — lag-0 acyclic edges from discovery only
  identification_dag_{region}.json — backbone ∪ structural (for ID + CF)
  counterfactual_tp_{region}.nc  — factual vs. CF precipitation time series

Run:   python scripts/04_causal_inference.py
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
    adjusted_or_logit,
    augment_edges_for_treatment_counterfactual,
    consensus_records_to_panel_dag_edges,
    counterfactual_enso_zero,
    crude_risk_ratio_binary,
    estimate_ace_dr,
    estimate_ace_ipw,
    filter_panel_dataframe_by_month,
    identification_dag_edges,
    minimum_e_value_rr_or_or,
    parse_backbone_edges_yaml,
    trim_by_propensity_quantiles,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TREATMENT = "sst"         # continuous SST anomaly (used for backdoor ID)
OUTCOME   = "tp_extreme"  # binary extreme-precip indicator
ENSO_VAR  = "nino34"


def load_panel(region: str, cfg: dict) -> pd.DataFrame:
    path = processed_path(f"panel_{region}.nc", cfg)
    ds   = xr.open_dataset(path)
    df   = ds.to_dataframe().dropna()
    log.info("Panel: %s — %d rows, vars: %s", region, len(df), list(df.columns))
    return df


def load_consensus_dag(region: str, cfg: dict) -> list[tuple[str, str]]:
    """Return lag-0 acyclic structural edges derived from the consensus discovery JSON."""
    path = results_path(f"consensus_dag_{region}.json", cfg)
    with open(path) as f:
        dag = json.load(f)
    raw = dag["consensus_edges"]
    edges = consensus_records_to_panel_dag_edges(raw)
    log.info(
        "Panel structural DAG: %d lag-0 acyclic edges (from %d consensus records)",
        len(edges),
        len(raw),
    )
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


def hidden_confounding_crude_gamma_scaling(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    ate_ipw: float,
    gamma_range: list[float] | None = None,
) -> dict:
    """
    Exploratory stress on the **crude** subgroup difference E[Y|T=1]-E[Y|T=0],
    applying a multiplicative Γ rescaling to group means. This is **not**
    Rosenbaum's RBAR / matched-pairs sensitivity; do **not** describe it as such
    in a paper without replacing it with a standard method.

    Included to show how hidden scaling moves **unadjusted** contrasts; compare
    ``ate_ipw`` (adjusted) against ``crude_risk_difference``.
    """
    if gamma_range is None:
        gamma_range = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    T = df[treatment].values.astype(int)
    Y = df[outcome].values.astype(float)
    m1 = float(np.mean(Y[T == 1])) if np.any(T == 1) else float("nan")
    m0 = float(np.mean(Y[T == 0])) if np.any(T == 0) else float("nan")
    crude = float(m1 - m0)

    results_sens: dict = {}
    for gamma in gamma_range:
        ate_hi = float(m1 - m0 * gamma)
        ate_lo = float(m1 / gamma - m0) if gamma else crude
        results_sens[str(gamma)] = {
            "crude_contrast_lower": round(ate_lo, 4),
            "crude_contrast_upper": round(ate_hi, 4),
            "sign_change": bool(ate_lo * ate_hi < 0),
        }

    critical_gamma = None
    for gamma in sorted(gamma_range):
        if results_sens[str(gamma)]["sign_change"]:
            critical_gamma = float(gamma)
            break

    return {
        "method_id": "crude_gamma_scaling_subgroup_means",
        "is_rosenbaum_bounds": False,
        "description": (
            "Gamma perturbs crude group means only; not calibrated to IPW ATE."
        ),
        "ipw_ate": float(ate_ipw),
        "crude_risk_difference": crude,
        "gamma_bounds": results_sens,
        "critical_gamma": critical_gamma,
        "interpretation": (
            f"Crude subgroup contrast flips sign by Γ = {critical_gamma}"
            if critical_gamma
            else "Crude scaling bounds do not cross zero for tested Γ"
        ),
    }


def main(cfg: dict) -> None:
    out_dir = results_path("", cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    inf_cfg = cfg.get("inference") or {}
    n_boot = int(inf_cfg.get("n_boot", 2000))
    block_size = int(inf_cfg.get("block_size", 12))
    season_months = inf_cfg.get("season_months")

    backbone = parse_backbone_edges_yaml(
        (cfg.get("identification_backbone") or {}).get("edges"),
    )
    log.info(
        "Inference settings: n_boot=%d block_size=%d season_months=%s backbone_edges=%d",
        n_boot,
        block_size,
        season_months,
        len(backbone),
    )

    all_ace_results: dict = {}

    for region in cfg["regions"]:
        log.info("=" * 60)
        log.info("Causal inference — region: %s", region)

        df0       = load_panel(region, cfg)
        df        = filter_panel_dataframe_by_month(df0, season_months)
        if season_months and len(df) < len(df0):
            log.info(
                "Seasonal subset: %d rows (from %d full panel)",
                len(df),
                len(df0),
            )
        if len(df) < 24:
            log.warning("Very few rows after filtering (%d); CIs may be unstable.", len(df))

        dag_edges = load_consensus_dag(region, cfg)
        struct_path = results_path(f"structural_dag_{region}.json", cfg)
        with open(struct_path, "w") as f:
            json.dump(
                {"region": region, "edges": [{"source": s, "target": t} for s, t in dag_edges]},
                f,
                indent=2,
            )
        log.info("Structural DAG → %s", struct_path)

        id_edges = identification_dag_edges(dag_edges, backbone)
        id_path = results_path(f"identification_dag_{region}.json", cfg)
        with open(id_path, "w") as f:
            json.dump(
                {
                    "region": region,
                    "edges": [{"source": s, "target": t} for s, t in id_edges],
                    "n_backbone": len(backbone),
                    "n_structural": len(dag_edges),
                    "n_identification": len(id_edges),
                },
                f,
                indent=2,
            )
        log.info("Identification DAG (%d edges) → %s", len(id_edges), id_path)

        all_vars = list(df.columns)

        sst_bin_col = "sst_warm"
        df[sst_bin_col] = (df[TREATMENT] >= 0.5).astype(int)

        profile = (inf_cfg.get("ace_spec") or {}).get("profile", "warm_sst_tp_extreme")
        _valid_profiles = {
            "warm_sst_tp_extreme",
            "warm_sst_tp_continuous",
            "nino_tp_extreme",
            "continuous_sst_tp_extreme",
        }
        if profile not in _valid_profiles:
            log.warning("Unknown ace_spec.profile %r — using warm_sst_tp_extreme", profile)
            profile = "warm_sst_tp_extreme"

        if profile == "warm_sst_tp_extreme":
            id_treat, id_outcome = TREATMENT, OUTCOME
            treat_col, out_col = sst_bin_col, OUTCOME
            treat_binary = True
            run_nino_alternative = bool(inf_cfg.get("report_alternative_estimands", False))
        elif profile == "warm_sst_tp_continuous":
            id_treat, id_outcome = TREATMENT, "tp"
            treat_col, out_col = sst_bin_col, "tp"
            treat_binary = True
            run_nino_alternative = False
        elif profile == "nino_tp_extreme":
            thr = float(inf_cfg.get("nino34_warm_threshold", 0.5))
            df["nino_hot"] = (df["nino34"] >= thr).astype(int)
            id_treat, id_outcome = ENSO_VAR, OUTCOME
            treat_col, out_col = "nino_hot", OUTCOME
            treat_binary = True
            run_nino_alternative = False
        else:  # continuous_sst_tp_extreme
            id_treat, id_outcome = TREATMENT, OUTCOME
            treat_col, out_col = "sst", OUTCOME
            treat_binary = False
            run_nino_alternative = False

        adj_set = identify_adjustment_set(id_edges, id_treat, id_outcome, all_vars)
        _drop_adj = {
            id_treat,
            id_outcome,
            treat_col,
            sst_bin_col,
            "nino_hot",
        }
        adj_set = [v for v in adj_set if v not in _drop_adj]
        if not adj_set:
            adj_set = [v for v in ["z500", "t2m", "swvl1"] if v in df.columns]
            log.warning("Empty adjustment set; using fallback: %s", adj_set)

        log.info(
            "ACE profile=%s treat_id=%s outcome_id=%s est_treat_col=%s est_out_col=%s binary_T=%s",
            profile,
            id_treat,
            id_outcome,
            treat_col,
            out_col,
            treat_binary,
        )
        log.info("Adjustment set: %s", adj_set)

        df_work = df
        trim_meta = None
        trim_q = inf_cfg.get("ipw_ps_trim_quantiles")
        if treat_binary and trim_q:
            tpair = (float(trim_q[0]), float(trim_q[1]))
            df_work, trim_meta = trim_by_propensity_quantiles(df, treat_col, adj_set, tpair)
            log.info(
                "Propensity trimming %s kept %d / %d rows",
                tpair,
                trim_meta["n_after"],
                trim_meta["n_before"],
            )
            if trim_meta["n_after"] < 24:
                log.warning("Very few rows after propensity trim (%d)", trim_meta["n_after"])

        log.info("Running IPW (n_boot=%d) …", n_boot)
        ace_ipw = estimate_ace_ipw(
            df_work,
            treatment=treat_col,
            outcome=out_col,
            covariates=adj_set,
            n_boot=n_boot,
            block_size=block_size,
        )
        log.info(
            "IPW  ATE = %.4f  [%.4f, %.4f]",
            ace_ipw["ate"],
            ace_ipw["ci_low"],
            ace_ipw["ci_high"],
        )

        if treat_binary:
            log.info("Running doubly-robust DR (n_boot=%d) …", n_boot)
            ace_dr = estimate_ace_dr(
                df_work,
                treatment=treat_col,
                outcome=out_col,
                covariates=adj_set,
                n_boot=n_boot,
                block_size=block_size,
            )
            log.info(
                "DR   ATE = %.4f  [%.4f, %.4f]",
                ace_dr["ate"],
                ace_dr["ci_low"],
                ace_dr["ci_high"],
            )
        else:
            ace_dr = {
                "ate": None,
                "ci_low": None,
                "ci_high": None,
                "note": "DR skipped for continuous SST (GPS IPW only; use discrete-T DR if discretizing).",
            }
            log.info("DR skipped (continuous treatment path).")

        if treat_binary:
            log.info("Running crude Γ scaling stress check …")
            sens = hidden_confounding_crude_gamma_scaling(
                df_work[[treat_col, out_col]].rename(columns={treat_col: "T"}),
                treatment="T",
                outcome=out_col,
                ate_ipw=float(ace_ipw["ate"]),
            )
            log.info("Critical Γ (crude contrast): %s", sens["critical_gamma"])
        else:
            sens = {
                "method_id": "not_applicable",
                "is_rosenbaum_bounds": False,
                "note": "Gamma scaling branch uses binary treatment; skipped for continuous SST.",
            }

        if out_col == OUTCOME and treat_binary:
            crude_rr = crude_risk_ratio_binary(df_work, treat_col, out_col)
            try:
                adj_or = adjusted_or_logit(df_work, treat_col, out_col, adj_set)
            except Exception as exc:
                log.warning("Adjusted OR / logit failed: %s", exc)
                adj_or = {"odds_ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
            e_value_block = {
                "reference": "VanderWeele_Ding_2017_minimum_e_value_point_estimate",
                "crude_risk_ratio": crude_rr,
                "crude_e_value_min": minimum_e_value_rr_or_or(crude_rr),
                "adjusted_odds_ratio": adj_or["odds_ratio"],
                "adjusted_or_ci95_low": adj_or["ci_low"],
                "adjusted_or_ci95_high": adj_or["ci_high"],
                "adjusted_or_e_value_min": minimum_e_value_rr_or_or(adj_or["odds_ratio"]),
            }
        else:
            e_value_block = {
                "reference": "not_applicable",
                "note": "E-values / OR path requires binary tp_extreme outcome.",
            }

        alt: dict = {}
        if run_nino_alternative:
            thr = float(inf_cfg.get("nino34_warm_threshold", 0.5))
            df_work["nino_hot"] = (df_work["nino34"] >= thr).astype(int)
            adj_nino = identify_adjustment_set(id_edges, ENSO_VAR, OUTCOME, all_vars)
            adj_nino = [
                v
                for v in adj_nino
                if v not in (ENSO_VAR, OUTCOME, sst_bin_col, "nino_hot", TREATMENT)
            ]
            if not adj_nino:
                log.info(
                    "Empty backdoor for nino34 → tp_extreme; IPW uses marginal P(nino_hot)"
                )
            log.info(
                "Alternative IPW: warm Niño 3.4 (≥%.2f K) vs not — adj %s",
                thr,
                adj_nino,
            )
            ace_nino = estimate_ace_ipw(
                df_work,
                treatment="nino_hot",
                outcome=OUTCOME,
                covariates=adj_nino,
                n_boot=n_boot,
                block_size=block_size,
            )
            alt["ipw_warm_nino34_vs_not"] = {
                "description": (
                    "IPW risk difference for tp_extreme: warm Niño 3.4 vs not; "
                    "backdoor adjustment sets identified with treatment node nino34."
                ),
                "treatment_column": "nino_hot",
                "nino34_threshold_K": thr,
                "adjustment_set": adj_nino,
                **ace_nino,
            }

        ace_out = {
            "region": region,
            "ace_profile": profile,
            "treatment": treat_col,
            "outcome": out_col,
            "adjustment_set": adj_set,
            "identification_graph": "identification_dag (backbone ∪ structural)",
            "ipw": ace_ipw,
            "dr": ace_dr,
            "sensitivity": sens,
            "e_value": e_value_block,
            "inference_meta": {
                "season_months": season_months,
                "n_boot": n_boot,
                "block_size": block_size,
                "n_rows": len(df_work),
                "propensity_trim": trim_meta,
                "id_treatment_graph": id_treat,
                "id_outcome_graph": id_outcome,
            },
        }
        if alt:
            ace_out["alternative_estimands"] = alt

        ace_path = results_path(f"ace_{region}.json", cfg)
        with open(ace_path, "w") as f:
            json.dump(ace_out, f, indent=2)
        log.info("ACE results → %s", ace_path)
        all_ace_results[region] = ace_out

        # --- Counterfactual do(ENSO = 0) ---
        log.info("Computing do(ENSO=0) counterfactual …")
        cf_edges, cf_augmented = augment_edges_for_treatment_counterfactual(
            id_edges, ENSO_VAR, "tp",
        )
        if cf_augmented:
            log.warning(
                "SCM augmented with prior edge %s → tp (no %s→…→tp path in identification DAG).",
                ENSO_VAR,
                ENSO_VAR,
            )
        _drop_cf = {c for c in (sst_bin_col, "nino_hot") if c in df_work.columns}
        if out_col != "tp":
            _drop_cf.add(out_col)
        cf_tp = counterfactual_enso_zero(
            panel=df_work[[c for c in df_work.columns if c not in _drop_cf]],
            dag_edges=cf_edges,
            treatment_col=ENSO_VAR,
            outcome_col="tp",
        )

        # Save factual + counterfactual as a NetCDF
        ds_cf = xr.Dataset({
            "tp_factual":      xr.DataArray(df_work["tp"].values, dims=["time"],
                                            attrs={"long_name": "Factual precipitation anomaly"}),
            "tp_cf_enso0":     xr.DataArray(cf_tp.values,    dims=["time"],
                                            attrs={"long_name": "Counterfactual precipitation anomaly (do(ENSO=0))"}),
        }, coords={"time": df_work.index.values})
        ds_cf.attrs["region"]      = region
        ds_cf.attrs["counterfactual"] = "do(nino34 = 0)"
        ds_cf.attrs["scm_used_prior_edge_nino34_tp"] = "true" if cf_augmented else "false"
        ds_cf.attrs["inference_n_times"] = str(len(df_work))
        if season_months is not None:
            ds_cf.attrs["inference_season_months"] = json.dumps(season_months)
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
