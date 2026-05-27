"""
scripts/05_figures.py
Generate all six publication-quality figures for the NeurIPS CCAI paper.

Figure inventory:
  fig1_domain_map.pdf         — Western US domain + sub-regions (cartopy)
  fig2_enso_precip_ts.pdf     — Niño 3.4 index + regional precip time series
  fig3_pcmci_graphs.pdf       — PCMCI+ causal graph (one panel per region)
  fig4_varlingam_heatmaps.pdf — VARLiNGAM coefficient matrices
  fig5_ace_comparison.pdf     — IPW vs. DR ACE bar chart with 95% CIs
  fig6_counterfactual_cdfs.pdf — Factual vs. do(ENSO=0) CDFs
  fig7_primary_holdout_summary.pdf — Primary/regional/holdout risk-difference forest plot

Run:   python scripts/05_figures.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_precip import load_config, nino34_index, processed_path, results_path
from causal_precip.data import figures_path
from causal_precip.viz import (
    FIGURE_STYLE,
    plot_ace_comparison,
    plot_counterfactual_cdfs,
    plot_domain_map,
    plot_enso_precip_timeseries,
    plot_pcmci_graph,
    plot_varlingam_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REGIONS = ["pacific_northwest", "california", "intermountain_west"]
COL2 = 7.0
COL1 = 3.5


def save_fig(fig: plt.Figure, name: str, cfg: dict) -> None:
    out = figures_path(name, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    log.info("Saved → %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Domain map
# ---------------------------------------------------------------------------

def fig1_domain_map(cfg: dict) -> None:
    plt.rcParams.update(FIGURE_STYLE)
    fig, ax = plt.subplots(1, 1, figsize=(COL2 * 0.55, COL2 * 0.45),
                           subplot_kw={"projection": __import__("cartopy.crs", fromlist=["PlateCarree"]).PlateCarree()})
    plot_domain_map(cfg, ax=ax)
    fig.tight_layout()
    save_fig(fig, "fig1_domain_map.pdf", cfg)


# ---------------------------------------------------------------------------
# Figure 2: ENSO + precip time series
# ---------------------------------------------------------------------------

def fig2_enso_precip_ts(cfg: dict) -> None:
    plt.rcParams.update(FIGURE_STYLE)

    nino34 = nino34_index(cfg)

    precip_ts: dict = {}
    for region in REGIONS:
        ds = xr.open_dataset(processed_path(f"panel_{region}.nc", cfg))
        try:
            if "tp" in ds:
                precip_ts[region] = ds["tp"]
        finally:
            ds.close()

    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    plot_enso_precip_timeseries(nino34, precip_ts, ax=ax, cfg=cfg)
    fig.tight_layout()
    save_fig(fig, "fig2_enso_precip_ts.pdf", cfg)


# ---------------------------------------------------------------------------
# Figure 3: PCMCI+ causal graphs
# ---------------------------------------------------------------------------

def fig3_pcmci_graphs(cfg: dict) -> None:
    plt.rcParams.update(FIGURE_STYLE)
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(COL2, COL2 * 0.45))

    for ax, region in zip(axes, REGIONS):
        pkl_path = results_path(f"pcmci_{region}.pkl", cfg)
        if not pkl_path.exists():
            log.warning("PCMCI+ results not found for %s, skipping.", region)
            ax.set_title(f"{region}\n(no results)")
            ax.axis("off")
            continue
        with open(pkl_path, "rb") as f:
            pcmci_results = pickle.load(f)
        plot_pcmci_graph(
            pcmci_results["val_matrix"],
            pcmci_results["p_matrix"],
            pcmci_results["var_names"],
            ax=ax,
            title=region.replace("_", " ").title(),
        )

    fig.suptitle("PCMCI+ (ParCorr): exploratory time-series graph", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3_pcmci_graphs.pdf", cfg)


# ---------------------------------------------------------------------------
# Figure 4: VARLiNGAM coefficient heatmaps
# ---------------------------------------------------------------------------

def fig4_varlingam_heatmaps(cfg: dict) -> None:
    plt.rcParams.update(FIGURE_STYLE)
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(COL2, COL1 * 0.8))

    for ax, region in zip(axes, REGIONS):
        pkl_path = results_path(f"varlingam_{region}.pkl", cfg)
        if not pkl_path.exists():
            log.warning("VARLiNGAM results not found for %s, skipping.", region)
            ax.set_title(region)
            ax.axis("off")
            continue
        with open(pkl_path, "rb") as f:
            blob = pickle.load(f)
        if isinstance(blob, dict) and "model" in blob:
            model = blob["model"]
        else:
            model = blob

        json_path = results_path(f"varlingam_{region}_summary.json", cfg)
        with open(json_path) as f:
            summary = json.load(f)
        var_names = summary["var_names"]

        plot_varlingam_heatmap(
            model.adjacency_matrices_,
            var_names,
            ax=ax,
            title=region.replace("_", " ").title(),
        )

    fig.suptitle("VARLiNGAM coefficient matrices (predictors scaled to 1σ)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig4_varlingam_heatmaps.pdf", cfg)


# ---------------------------------------------------------------------------
# Figure 5: ACE bar chart
# ---------------------------------------------------------------------------

def fig5_ace_comparison(cfg: dict) -> None:
    plt.rcParams.update(FIGURE_STYLE)

    ace_combined_path = results_path("ace_all_regions.json", cfg)
    if not ace_combined_path.exists():
        log.warning("Combined ACE results not found; trying per-region files.")
        ace_results: dict = {}
        for region in REGIONS:
            p = results_path(f"ace_{region}.json", cfg)
            if p.exists():
                with open(p) as f:
                    ace_results[region] = json.load(f)
    else:
        with open(ace_combined_path) as f:
            ace_results = json.load(f)

    if not ace_results:
        log.warning("No ACE results found; skipping fig5.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL2, 4.9), sharex=True)
    plot_ace_comparison(ace_results, ax=ax1, ax_nino=ax2)
    fig.suptitle(
        "Risk differences for P(extreme precip): primary estimand (SST) and exploratory Niño IPW",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig5_ace_comparison.pdf", cfg)


# ---------------------------------------------------------------------------
# Figure 6: Counterfactual CDFs
# ---------------------------------------------------------------------------

def fig6_counterfactual_cdfs(cfg: dict) -> None:
    plt.rcParams.update(FIGURE_STYLE)

    available = [r for r in REGIONS
                 if results_path(f"counterfactual_tp_{r}.nc", cfg).exists()]
    if not available:
        log.warning("No counterfactual results found; skipping fig6.")
        return

    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(COL1 * ncols, COL1 * 0.9))
    if ncols == 1:
        axes = [axes]

    for ax, region in zip(axes, available):
        cf_path = results_path(f"counterfactual_tp_{region}.nc", cfg)
        ds_cf   = xr.open_dataset(cf_path)
        try:
            factual = pd.Series(ds_cf["tp_factual"].values)
            counterfactual = pd.Series(ds_cf["tp_cf_enso0"].values)
            plot_counterfactual_cdfs(factual, counterfactual, region, ax=ax, n_boot=300)
        finally:
            ds_cf.close()

    fig.suptitle("Precipitation distribution under do(ENSO = 0)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig6_counterfactual_cdfs.pdf", cfg)


# ---------------------------------------------------------------------------
# Figure 7: Primary + holdout forest plot
# ---------------------------------------------------------------------------

def _forest_rows_from_ace(ace_results: dict, pooled: dict | None) -> list[dict]:
    """Collect full-sample regional and pooled risk-difference rows."""
    label_map = {
        "pacific_northwest": "Pacific Northwest",
        "california": "California",
        "intermountain_west": "Intermountain West",
    }
    rows: list[dict] = []
    for region in REGIONS:
        if region not in ace_results:
            continue
        entry = ace_results[region]
        for est_name, pretty in (("ipw", "IPW"), ("dr", "DR")):
            est = entry.get(est_name) or {}
            if est.get("ate") is None:
                continue
            rows.append(
                {
                    "label": f"{label_map.get(region, region)} {pretty}",
                    "ate": float(est["ate"]),
                    "ci_low": float(est["ci_low"]),
                    "ci_high": float(est["ci_high"]),
                    "primary": region == "pacific_northwest",
                }
            )

    if pooled:
        for est_name, pretty in (("ipw", "IPW"), ("dr", "DR")):
            est = pooled.get(est_name) or {}
            if est.get("ate") is None:
                continue
            rows.append(
                {
                    "label": f"Pooled western US {pretty}",
                    "ate": float(est["ate"]),
                    "ci_low": float(est["ci_low"]),
                    "ci_high": float(est["ci_high"]),
                    "primary": False,
                }
            )
    return rows


def _forest_rows_from_holdout(holdout: dict) -> list[dict]:
    """Collect temporal-holdout rows for the frozen primary estimand."""
    rows: list[dict] = []
    for est_name, pretty in (("ipw", "IPW"), ("dr", "DR")):
        est = ((holdout.get("holdout") or {}).get(est_name) or {})
        if est.get("ate") is None:
            continue
        rows.append(
            {
                "label": f"Holdout Pacific Northwest {pretty}",
                "ate": float(est["ate"]),
                "ci_low": float(est["ci_low"]),
                "ci_high": float(est["ci_high"]),
                "primary": True,
            }
        )
    return rows


def _plot_forest_rows(ax: plt.Axes, rows: list[dict], title: str) -> None:
    y = np.arange(len(rows))
    ax.axvline(0.0, color="0.25", lw=0.9, ls="--", zorder=0)
    for yi, row in zip(y, rows):
        color = "#1f77b4" if row.get("primary") else "0.45"
        x = row["ate"]
        xerr = np.array([[x - row["ci_low"]], [row["ci_high"] - x]])
        ax.errorbar(
            x,
            yi,
            xerr=xerr,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=4.5,
            lw=1.3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="0.9", lw=0.8)


def fig7_primary_holdout_summary(cfg: dict) -> None:
    plt.rcParams.update(FIGURE_STYLE)

    ace_path = results_path("ace_all_regions.json", cfg)
    holdout_path = results_path("holdout_validation.json", cfg)
    pooled_path = results_path("pooled_ace.json", cfg)
    missing = [p for p in (ace_path, holdout_path, pooled_path) if not p.exists()]
    if missing:
        log.warning("Missing inputs for fig7, skipping: %s", [str(p) for p in missing])
        return

    with open(ace_path) as f:
        ace_results = json.load(f)
    with open(holdout_path) as f:
        holdout = json.load(f)
    with open(pooled_path) as f:
        pooled = json.load(f)

    full_rows = _forest_rows_from_ace(ace_results, pooled)
    holdout_rows = _forest_rows_from_holdout(holdout)
    if not full_rows or not holdout_rows:
        log.warning("Insufficient rows for fig7; skipping.")
        return
    holdout_rows += [{'ate':np.nan, 'ci_low':np.nan, 'ci_high':np.nan, 'label':None}] * 6
    all_rows = full_rows + holdout_rows
    xmin = min(r["ci_low"] for r in all_rows)
    xmax = max(r["ci_high"] for r in all_rows)
    pad = max(0.02, 0.08 * (xmax - xmin))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(COL2, 3.8),
        sharex=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    _plot_forest_rows(ax1, full_rows, "A. Full-sample regional and pooled estimates")
    _plot_forest_rows(ax2, holdout_rows, "B. Temporal holdout")
    ax1.set_xlabel("Risk difference for P(extreme precipitation)")
    ax2.set_xlabel("Risk difference for P(extreme precipitation)")
    ax1.set_xlim(xmin - pad, xmax + pad)

    strict = bool((holdout.get("confirmation") or {}).get("strict_statistical_replication"))
    split = holdout.get("split") or {}
    ax2.text(
        0.02,
        0.03,
        (
            f"Strict replication: {'yes' if strict else 'no'}\n"
            f"Train {split.get('train_start')}–{split.get('train_end')}; "
            f"eval {split.get('eval_start')}–{split.get('eval_end')}"
        ),
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "lw": 0.8},
    )

    fig.suptitle(
        "Warm SST → ONDJFM extreme-precipitation risk: validated PNW signal, null wider context",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig7_primary_holdout_summary.pdf", cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    log.info("Generating Figure 1: domain map …")
    try:
        fig1_domain_map(cfg)
    except Exception as e:
        log.warning("fig1 failed: %s", e)

    log.info("Generating Figure 2: ENSO + precip time series …")
    try:
        fig2_enso_precip_ts(cfg)
    except Exception as e:
        log.warning("fig2 failed: %s", e)

    log.info("Generating Figure 3: PCMCI+ graphs …")
    try:
        fig3_pcmci_graphs(cfg)
    except Exception as e:
        log.warning("fig3 failed: %s", e)

    log.info("Generating Figure 4: VARLiNGAM heatmaps …")
    try:
        fig4_varlingam_heatmaps(cfg)
    except Exception as e:
        log.warning("fig4 failed: %s", e)

    log.info("Generating Figure 5: ACE comparison …")
    try:
        fig5_ace_comparison(cfg)
    except Exception as e:
        log.warning("fig5 failed: %s", e)

    log.info("Generating Figure 6: counterfactual CDFs …")
    try:
        fig6_counterfactual_cdfs(cfg)
    except Exception as e:
        log.warning("fig6 failed: %s", e)

    log.info("Generating Figure 7: primary + holdout summary …")
    try:
        fig7_primary_holdout_summary(cfg)
    except Exception as e:
        log.warning("fig7 failed: %s", e)

    log.info("All figures complete.")


if __name__ == "__main__":
    cfg = load_config()
    log.info("Environment : %s", cfg.get("_env", "local"))
    main(cfg)
