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
    cfg: Optional[dict] = None,
    highlight_inference_months: bool = True,
) -> plt.Axes:
    """
    Dual-axis plot: Niño 3.4 index (shaded) and regional tp anomalies (lines).

    If ``cfg`` is passed and ``highlight_inference_months`` is True, lightly shades
    months where ``cfg['inference']['season_months']`` matches (links plot to ACE estimand).
    """
    from pandas.tseries.offsets import MonthEnd

    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(COL2, 2.2))

    times  = pd.to_datetime(nino34.time.values)
    tidx   = pd.DatetimeIndex(times)
    nino34_vals = nino34.values

    ax2 = ax.twinx()
    if cfg and highlight_inference_months:
        months = (cfg.get("inference") or {}).get("season_months")
        if months:
            mset = set(int(m) for m in months)
            mask = np.array([dt.month in mset for dt in tidx], dtype=bool)
            i0: Optional[int] = None
            for i, ok in enumerate(mask):
                if ok and i0 is None:
                    i0 = i
                elif not ok and i0 is not None:
                    ax.axvspan(
                        tidx[i0], tidx[i - 1] + MonthEnd(0),
                        facecolor="#9E9E9E", alpha=0.12, linewidth=0, zorder=0,
                    )
                    i0 = None
            if i0 is not None:
                ax.axvspan(
                    tidx[i0], tidx[-1] + MonthEnd(0),
                    facecolor="#9E9E9E", alpha=0.12, linewidth=0, zorder=0,
                )
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

    ax2.set_ylabel("Precipitation anomaly (ERA5 tp, m)")
    ax.set_xlabel("Year")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper right", ncol=2, framealpha=0.85)
    if cfg and cfg.get("date_range"):
        dr = cfg["date_range"]
        ax.set_title(
            f"Niño 3.4 and regional tp anomalies ({dr['start']} — {dr['end']}); "
            "shading = inference months"
        )
    else:
        ax.set_title("Niño 3.4 index and regional precipitation anomalies")
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

    for src, dst in dict.fromkeys(edges):
        G.add_edge(src, dst)
    pos = nx.spring_layout(G, seed=42, k=1.8 / max(1, np.sqrt(n_vars)))
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

    ax.set_title(f"{title}\n(solid: lag ≥1; dashed: τ=0)")
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
    vmax_percentile: float = 98.0,
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

    flat = np.abs(stacked.ravel())
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) and vmax_percentile < 100.0:
        vmax = float(np.percentile(flat, vmax_percentile))
    else:
        vmax = float(np.abs(stacked).max())
    if vmax < 1e-12:
        vmax = 1.0
    im   = ax.imshow(stacked, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(var_names)
    xtick_pos    = [i * n_vars + n_vars // 2 for i in range(n_lags)]
    xtick_labels = [f"lag {i+1}" for i in range(n_lags)]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels)

    for lag_idx in range(1, n_lags):
        ax.axvline(lag_idx * n_vars - 0.5, color="white", linewidth=0.8)

    plt.colorbar(im, ax=ax, shrink=0.7, label="Coefficient (1σ units)")
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Figure 5: ACE comparison bar chart
# ---------------------------------------------------------------------------

def plot_ace_comparison(
    ace_results: Dict[str, dict],
    ax: Optional[plt.Axes] = None,
    ax_nino: Optional[plt.Axes] = None,
    title: str = "ACE: warm SST vs not on P(extreme precip)",
    title_nino: str = "Exploratory: warm Niño 3.4 vs not (IPW)",
) -> plt.Axes:
    """
    Grouped bar chart comparing IPW vs. DR ACE estimates across regions.

    If ``ax_nino`` is set, draws a second panel with the exploratory Niño IPW
    estimand (when present under ``alternative_estimands``).

    ace_results: {region: {ipw: {...}, dr: {...}, alternative_estimands: {...}}}
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
    ax.set_ylabel("Risk difference (probability scale)")
    ax.legend()
    ax.set_title(title)

    if ax_nino is not None:
        ates_n, lo_n, hi_n = [], [], []
        for r in regions:
            alt = (ace_results[r].get("alternative_estimands") or {}).get("ipw_warm_nino34_vs_not")
            if not alt or "ate" not in alt:
                ates_n.append(float("nan"))
                lo_n.append(float("nan"))
                hi_n.append(float("nan"))
                continue
            ate = float(alt["ate"])
            ates_n.append(ate)
            lo_n.append(ate - float(alt["ci_low"]))
            hi_n.append(float(alt["ci_high"]) - ate)
        ax_nino.bar(x, ates_n, width=0.55, color="#7B1FA2", alpha=0.88, label="IPW (marginal propensity)")
        ax_nino.errorbar(
            x, ates_n,
            yerr=[lo_n, hi_n],
            fmt="none", color="black", capsize=3, linewidth=1.0,
        )
        ax_nino.axhline(0, color="black", linewidth=0.6)
        ax_nino.set_xticks(x)
        ax_nino.set_xticklabels([REGION_LABELS.get(r, r) for r in regions], rotation=15, ha="right")
        ax_nino.set_ylabel("Risk difference (probability scale)")
        ax_nino.set_title(title_nino)
        ax_nino.legend(loc="upper right", fontsize=7)

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

    fvals   = factual.dropna().values.astype(float)
    cfvals  = counterfactual.dropna().values.astype(float)
    # Display in mm water-equivalent for readability (ERA5 tp monthly sum in m).
    fvals_mm = fvals * 1000.0
    cfvals_mm = cfvals * 1000.0
    xs_f, ys_f   = _ecdf(fvals_mm)
    xs_cf, ys_cf = _ecdf(cfvals_mm)

    # Bootstrap bands on counterfactual
    rng = np.random.default_rng(0)
    boot_cdfs = []
    x_grid = np.linspace(min(fvals_mm.min(), cfvals_mm.min()),
                         max(fvals_mm.max(), cfvals_mm.max()), 200)
    for _ in range(n_boot):
        samp = rng.choice(cfvals_mm, size=len(cfvals_mm), replace=True)
        boot_cdfs.append(np.searchsorted(np.sort(samp), x_grid) / len(samp))
    band_lo = np.percentile(boot_cdfs, 2.5, axis=0)
    band_hi = np.percentile(boot_cdfs, 97.5, axis=0)

    ax.plot(xs_f, ys_f, color="#D32F2F", linewidth=1.4,
            label="Factual")
    ax.plot(xs_cf, ys_cf, color="#1565C0", linewidth=1.4,
            label="do(ENSO = 0)")
    ax.fill_between(x_grid, band_lo, band_hi,
                    color="#1565C0", alpha=0.2, label="95% bootstrap CI")

    ax.set_xlabel("Monthly precipitation anomaly (mm)")
    ax.set_ylabel("Empirical CDF")
    ax.legend(framealpha=0.85)
    ax.set_title(f"Counterfactual precipitation — {REGION_LABELS.get(region, region)}")
    return ax
