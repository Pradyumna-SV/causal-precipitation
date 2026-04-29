"""
src/causal_precip/viz.py
Reusable publication-quality plotting functions for the NeurIPS CCAI paper.
All figures target 300 dpi, Nature/NeurIPS single- or double-column widths.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

FIGURE_STYLE: dict = {
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
}

# NeurIPS column widths in inches
COL1 = 3.5
COL2 = 7.0

REGION_COLORS = {
    "pacific_northwest": "#2196F3",
    "california":        "#F44336",
    "intermountain_west": "#4CAF50",
}

REGION_LABELS = {
    "pacific_northwest": "Pacific Northwest",
    "california":        "California",
    "intermountain_west": "Intermountain West",
}


def _apply_style() -> None:
    mpl.rcParams.update(FIGURE_STYLE)


# ---------------------------------------------------------------------------
# Figure 1: Domain map with sub-regions
# ---------------------------------------------------------------------------

def plot_domain_map(
    cfg: dict,
    ax: Optional[object] = None,
    show_nino34: bool = True,
) -> object:
    """
    Cartopy map of the Western US domain with sub-region bounding boxes.
    Returns the matplotlib Axes.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle

    _apply_style()
    proj = ccrs.PlateCarree()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(COL1, COL1 * 0.8),
                               subplot_kw={"projection": proj})

    d = cfg["domain"]
    ax.set_extent([d["lon_min"] - 2, d["lon_max"] + 2,
                   d["lat_min"] - 2, d["lat_max"] + 2], crs=proj)

    ax.add_feature(cfeature.LAND,       facecolor="#F5F5F0", zorder=0)
    ax.add_feature(cfeature.OCEAN,      facecolor="#DDEEFF", zorder=0)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.4, linestyle="--")
    ax.add_feature(cfeature.STATES,     linewidth=0.3, edgecolor="gray")

    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.5, linestyle=":")
    gl.top_labels    = False
    gl.right_labels  = False
    gl.xlocator = mticker.FixedLocator(range(-130, -95, 5))
    gl.ylocator = mticker.FixedLocator(range(25, 55, 5))

    # Study domain outline
    ax.add_patch(Rectangle(
        (d["lon_min"], d["lat_min"]),
        d["lon_max"] - d["lon_min"],
        d["lat_max"] - d["lat_min"],
        linewidth=1.5, edgecolor="black", facecolor="none",
        transform=proj, zorder=3,
    ))

    for rname, rcfg in cfg["regions"].items():
        ax.add_patch(Rectangle(
            (rcfg["lon_min"], rcfg["lat_min"]),
            rcfg["lon_max"] - rcfg["lon_min"],
            rcfg["lat_max"] - rcfg["lat_min"],
            linewidth=1.2,
            edgecolor=REGION_COLORS[rname],
            facecolor=mpl.colors.to_rgba(REGION_COLORS[rname], alpha=0.15),
            transform=proj, zorder=4, label=REGION_LABELS[rname],
        ))

    ax.legend(loc="lower left", framealpha=0.85)
    ax.set_title("Study domain and sub-regions")
    return ax


# ---------------------------------------------------------------------------
# Figure 2: Niño 3.4 + regional precipitation time series
# ---------------------------------------------------------------------------

def plot_enso_precip_timeseries(
    nino34: "xr.DataArray",  # noqa: F821
    precip_ts: Dict[str, "xr.DataArray"],  # noqa: F821
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Dual-axis plot: Niño 3.4 index (shaded) and regional tp anomalies (lines).
    """
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(COL2, 2.2))

    times  = pd.to_datetime(nino34.time.values)
    nino34_vals = nino34.values

    ax2 = ax.twinx()
    ax.fill_between(times, nino34_vals, 0,
                    where=nino34_vals > 0, color="#D32F2F", alpha=0.35, label="El Niño")
    ax.fill_between(times, nino34_vals, 0,
                    where=nino34_vals < 0, color="#1565C0", alpha=0.35, label="La Niña")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Niño 3.4 SST anomaly (K)")

    for rname, da in precip_ts.items():
        ax2.plot(times, da.values,
                 color=REGION_COLORS[rname],
                 label=REGION_LABELS[rname],
                 linewidth=0.9, alpha=0.85)

    ax2.set_ylabel("Precipitation anomaly (mm/day)")
    ax.set_xlabel("Year")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", ncol=2, framealpha=0.85)
    ax.set_title("Niño 3.4 index and regional precipitation anomalies (1979–2023)")
    return ax


# ---------------------------------------------------------------------------
# Figure 3: PCMCI+ causal graph
# ---------------------------------------------------------------------------

def plot_pcmci_graph(
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    var_names: List[str],
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
    title: str = "PCMCI+ causal graph",
) -> plt.Axes:
    """
    Visualise the PCMCI+ result as a directed graph.

    val_matrix : (n_vars, n_vars, tau_max+1) — MCI coefficients
    p_matrix   : (n_vars, n_vars, tau_max+1) — p-values
    Significant edges (p < alpha, tau > 0) are drawn with width ∝ |MCI|.
    Contemporaneous edges (tau=0) drawn dashed.
    """
    import networkx as nx

    _apply_style()
    n_vars = len(var_names)

    if ax is None:
        _, ax = plt.subplots(figsize=(COL2 * 0.6, COL2 * 0.5))

    G   = nx.DiGraph()
    G.add_nodes_from(var_names)
    edges, widths, styles = [], [], []

    tau_max = val_matrix.shape[2] - 1
    for j in range(n_vars):          # target
        for i in range(n_vars):      # source
            for tau in range(0, tau_max + 1):
                if p_matrix[i, j, tau] < alpha and abs(val_matrix[i, j, tau]) > 1e-6:
                    if i == j and tau == 0:
                        continue     # skip self-loops at lag-0
                    edges.append((var_names[i], var_names[j]))
                    widths.append(max(0.5, abs(val_matrix[i, j, tau]) * 6))
                    styles.append("dashed" if tau == 0 else "solid")

    pos = nx.circular_layout(G)
    node_colors = plt.cm.tab10(np.linspace(0, 0.9, n_vars))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600,
                           node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    for (src, dst), w, style in zip(edges, widths, styles):
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(src, dst)],
            width=w, style=style, alpha=0.8,
            arrows=True, arrowsize=12,
            connectionstyle="arc3,rad=0.1",
        )

    ax.set_title(title)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Figure 4: VARLiNGAM coefficient heatmap
# ---------------------------------------------------------------------------

def plot_varlingam_heatmap(
    coef_matrices: List[np.ndarray],
    var_names: List[str],
    ax: Optional[plt.Axes] = None,
    title: str = "VARLiNGAM coefficients",
) -> plt.Axes:
    """
    Heatmap of VARLiNGAM causal coefficient matrices (stacked across lags).
    coef_matrices: list of (n_vars × n_vars) arrays, one per lag.
    """
    _apply_style()
    stacked = np.concatenate([m for m in coef_matrices], axis=1)
    n_vars  = len(var_names)
    n_lags  = len(coef_matrices)

    if ax is None:
        _, ax = plt.subplots(figsize=(COL2, COL1 * 0.7))

    vmax = np.abs(stacked).max()
    im   = ax.imshow(stacked, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(var_names)
    xtick_pos    = [i * n_vars + n_vars // 2 for i in range(n_lags)]
    xtick_labels = [f"lag {i+1}" for i in range(n_lags)]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels)

    for lag_idx in range(1, n_lags):
        ax.axvline(lag_idx * n_vars - 0.5, color="white", linewidth=0.8)

    plt.colorbar(im, ax=ax, shrink=0.7, label="Causal coefficient")
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Figure 5: ACE comparison bar chart
# ---------------------------------------------------------------------------

def plot_ace_comparison(
    ace_results: Dict[str, Dict[str, Dict[str, float]]],
    ax: Optional[plt.Axes] = None,
    title: str = "ACE of SST anomaly on P(extreme precipitation)",
) -> plt.Axes:
    """
    Grouped bar chart comparing IPW vs. DR ACE estimates across regions.

    ace_results: {region: {estimator: {ate, ci_low, ci_high}}}
    """
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(COL2, 2.4))

    regions    = list(ace_results.keys())
    estimators = ["ipw", "dr"]
    labels     = {"ipw": "IPW", "dr": "Doubly-robust"}
    est_colors = {"ipw": "#1976D2", "dr": "#388E3C"}

    n   = len(regions)
    x   = np.arange(n)
    w   = 0.35
    off = {"ipw": -w / 2, "dr": w / 2}

    for est in estimators:
        ates   = [ace_results[r][est]["ate"]    for r in regions]
        ci_lo  = [ace_results[r][est]["ate"] - ace_results[r][est]["ci_low"]  for r in regions]
        ci_hi  = [ace_results[r][est]["ci_high"] - ace_results[r][est]["ate"] for r in regions]
        bars   = ax.bar(
            x + off[est], ates, width=w,
            color=est_colors[est], alpha=0.85, label=labels[est],
        )
        ax.errorbar(
            x + off[est], ates,
            yerr=[ci_lo, ci_hi],
            fmt="none", color="black", capsize=3, linewidth=1.0,
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions], rotation=15, ha="right")
    ax.set_ylabel("ΔP(extreme precip)")
    ax.legend()
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Figure 6: Counterfactual CDF comparison
# ---------------------------------------------------------------------------

def plot_counterfactual_cdfs(
    factual: pd.Series,
    counterfactual: pd.Series,
    region: str,
    ax: Optional[plt.Axes] = None,
    n_boot: int = 500,
) -> plt.Axes:
    """
    Empirical CDF of factual vs. do(ENSO=0) counterfactual precipitation,
    with bootstrap uncertainty band around the counterfactual CDF.
    """
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(COL1, COL1 * 0.85))

    def _ecdf(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.sort(s)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    fvals   = factual.dropna().values
    cfvals  = counterfactual.dropna().values
    xs_f, ys_f   = _ecdf(fvals)
    xs_cf, ys_cf = _ecdf(cfvals)

    # Bootstrap bands on counterfactual
    rng = np.random.default_rng(0)
    boot_cdfs = []
    x_grid = np.linspace(min(fvals.min(), cfvals.min()),
                         max(fvals.max(), cfvals.max()), 200)
    for _ in range(n_boot):
        samp = rng.choice(cfvals, size=len(cfvals), replace=True)
        boot_cdfs.append(np.searchsorted(np.sort(samp), x_grid) / len(samp))
    band_lo = np.percentile(boot_cdfs, 2.5, axis=0)
    band_hi = np.percentile(boot_cdfs, 97.5, axis=0)

    ax.plot(xs_f, ys_f, color="#D32F2F", linewidth=1.4,
            label="Factual")
    ax.plot(xs_cf, ys_cf, color="#1565C0", linewidth=1.4,
            label="do(ENSO = 0)")
    ax.fill_between(x_grid, band_lo, band_hi,
                    color="#1565C0", alpha=0.2, label="95% bootstrap CI")

    ax.set_xlabel("Monthly precipitation anomaly (mm/day)")
    ax.set_ylabel("Empirical CDF")
    ax.legend(framealpha=0.85)
    ax.set_title(f"Counterfactual precipitation — {REGION_LABELS.get(region, region)}")
    return ax
