"""
src/causal_precip/inference.py
Causal inference estimators: IPW, doubly-robust ACE, block bootstrap CIs,
and the do(ENSO=0) counterfactual.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Consensus graph → simultaneous DAG for panel SCM / backdoor
# ---------------------------------------------------------------------------

def consensus_records_to_panel_dag_edges(
    consensus_edges: List[dict],
) -> List[tuple[str, str]]:
    """
    Turn PCMCI+ consensus *records* into edges for a *monthly panel* with no
    explicit lag columns.

    - Drops self-edges (lagged autocorrelation is not a simultaneous parent).
    - Keeps only **lag == 0** links (contemporaneous; matches wide data format).
    - **Greedy DAG:** add edges in order of increasing PCMCI *p-value* (most
      significant first); discard an edge if it would create a directed cycle.

    This avoids NaN-filled SCM propagation when instantaneous cycles or self-loops
    are present in the raw discovery output.
    """
    import networkx as nx

    scored: list[tuple[float, str, str]] = []
    for e in consensus_edges:
        s, t = e["source"], e["target"]
        if s == t:
            continue
        lag = int(e.get("lag", 0))
        if lag != 0:
            continue
        pval = float(e.get("pvalue", 1.0))
        scored.append((pval, s, t))

    scored.sort(key=lambda x: x[0])

    G = nx.DiGraph()
    for _p, s, t in scored:
        G.add_edge(s, t)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(s, t)

    # Deterministic tie-break: sorted order
    return sorted(G.edges())


def augment_edges_for_treatment_counterfactual(
    dag_edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
) -> tuple[list[tuple[str, str]], bool]:
    """
    If ``treatment`` has no directed path to ``outcome`` in ``dag_edges``,
    append ``(treatment, outcome)`` when that keeps the graph acyclic.

    Used for ``do(ENSO=0)``: lag-0 discovery graphs often omit Niño teleconnections;
    this edge is a **documented domain prior** for propagating the intervention,
    not an additional claim that the consensus graph already contained it.
    """
    import networkx as nx

    aug_candidate = list(dict.fromkeys([*dag_edges, (treatment, outcome)]))
    G = nx.DiGraph()
    G.add_edges_from(dag_edges)
    if treatment in G and outcome in G and nx.has_path(G, treatment, outcome):
        return dag_edges, False
    G2 = nx.DiGraph()
    G2.add_edges_from(aug_candidate)
    if not nx.is_directed_acyclic_graph(G2):
        return dag_edges, False
    return aug_candidate, True


def merge_ordered_edges_acyclic(*edge_sequences: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Sequentially add edges from one or more ordered lists, skipping any edge
    that would introduce a directed cycle or duplicate an existing edge.
    """
    import networkx as nx

    G = nx.DiGraph()
    accepted: list[tuple[str, str]] = []
    for edges in edge_sequences:
        for s, t in edges:
            if s == t:
                continue
            if G.has_edge(s, t):
                continue
            G.add_edge(s, t)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(s, t)
            else:
                accepted.append((s, t))
    return accepted


def _edges_topo_sort_order(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Order edges so that, for an acyclic graph, greedy addition keeps every edge.
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(G) or not edges:
        return sorted(edges)
    topo_rank = {n: i for i, n in enumerate(nx.topological_sort(G))}
    return sorted(edges, key=lambda e: (topo_rank[e[0]], topo_rank[e[1]], e[0], e[1]))


def identification_dag_edges(
    structural_edges: list[tuple[str, str]],
    backbone_edges: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Merge a literature / domain **identification backbone** with the data-derived
    lag-0 structural DAG. Backbone edges are tried first (in list order), then
    structural edges in an order compatible with the structural DAG topology.
    """
    struct_ordered = _edges_topo_sort_order(structural_edges)
    return merge_ordered_edges_acyclic(backbone_edges, struct_ordered)


def minimum_e_value_rr_or_or(effect: float) -> float:
    """
    Minimum E-value (VanderWeele & Ding 2017) for a **point estimate** risk ratio
    or odds ratio on the ratio scale; use the larger of ``effect`` and ``1/effect``
    so the summary is two-sided.
    """
    x = float(effect)
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    x = max(x, 1.0 / x)
    return float(x + np.sqrt(x * (x - 1.0)))


def crude_risk_ratio_binary(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """P(Y=1|T=1) / P(Y=1|T=0) for binary T and Y."""
    t = df[treatment].values
    y = df[outcome].values.astype(float)
    if not np.any(t == 1) or not np.any(t == 0):
        return float("nan")
    p1 = float(np.mean(y[t == 1]))
    p0 = float(np.mean(y[t == 0]))
    if p0 <= 0:
        return float("nan")
    return p1 / p0


def adjusted_or_logit(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
) -> dict[str, float]:
    """
    Maximum-likelihood logistic regression OR for ``treatment`` on ``outcome``,
    adjusting for ``covariates``. Returns point OR and 95% Wald CI.
    """
    import statsmodels.api as sm

    cols = [treatment] + [c for c in covariates if c != treatment]
    X = sm.add_constant(df[cols].astype(float), has_constant="add")
    y = df[outcome].values.astype(float)
    res = sm.Logit(y, X).fit(disp=False)
    beta = float(res.params[treatment])
    se = float(res.bse[treatment])
    lo = np.exp(beta - 1.96 * se)
    hi = np.exp(beta + 1.96 * se)
    return {"odds_ratio": float(np.exp(beta)), "ci_low": float(lo), "ci_high": float(hi)}


def filter_panel_dataframe_by_month(
    df: pd.DataFrame,
    months: list[int] | None,
) -> pd.DataFrame:
    """
    Keep only rows whose index time falls in ``months`` (1–12).
    ``months`` empty/None keeps all rows.
    """
    if not months:
        return df
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        if "time" in idx.names:
            t = pd.to_datetime(idx.get_level_values("time"))
        else:
            t = pd.to_datetime(idx.get_level_values(0))
    else:
        t = pd.DatetimeIndex(pd.to_datetime(idx))
    sel = t.month.isin(months)
    out = df.loc[sel].copy()
    return out


def parse_backbone_edges_yaml(raw: object) -> list[tuple[str, str]]:
    """Parse ``identification_backbone.edges`` from YAML (list of [src, tgt] pairs)."""
    if not raw:
        return []
    out: list[tuple[str, str]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        out.append((str(pair[0]), str(pair[1])))
    return out


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    func: Callable[[pd.DataFrame], float],
    df: pd.DataFrame,
    block_size: int = 12,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Block bootstrap confidence interval for a scalar statistic.

    Uses non-overlapping blocks of length `block_size` to respect
    monthly autocorrelation structure in climate time series.

    Returns (ci_low, ci_high) at the (alpha/2, 1-alpha/2) percentiles.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n     = len(df)
    n_blocks = n // block_size
    block_starts = np.arange(n_blocks) * block_size

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block_size) for s in chosen])
        boot_stats[i] = func(df.iloc[idx].reset_index(drop=True))

    lo = np.nanpercentile(boot_stats, 100 * alpha / 2)
    hi = np.nanpercentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def block_bootstrap_ci_clustered(
    func: Callable[[pd.DataFrame], float],
    df: pd.DataFrame,
    time_col: str,
    block_size: int = 12,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Block bootstrap where each observation has a ``time_col`` value; blocks are
    consecutive *unique* time points. Each resample draws ``n_blocks`` blocks with
    replacement and stacks all regions / rows whose time falls in chosen blocks.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ut = np.sort(pd.to_datetime(df[time_col].values).unique())
    n_time = len(ut)
    if n_time < block_size * 2:
        raise ValueError(
            f"Not enough unique times ({n_time}) for block_size={block_size} clustered bootstrap."
        )
    n_blocks = n_time // block_size
    block_starts = np.arange(n_blocks) * block_size

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        chosen_starts = rng.choice(block_starts, size=n_blocks, replace=True)
        sel = []
        for s in chosen_starts:
            chunk = ut[s : s + block_size]
            sel.extend(chunk)
        sub = df.loc[df[time_col].isin(sel)]
        boot_stats[i] = func(sub.reset_index(drop=True))

    lo = np.nanpercentile(boot_stats, 100 * alpha / 2)
    hi = np.nanpercentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Propensity score (IPW)
# ---------------------------------------------------------------------------

def _propensity_scores(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
) -> np.ndarray:
    """
    Logistic propensity scores P(T=1 | X). With an **empty** adjustment set,
    returns the sample marginal P(T=1) (valid when the backdoor is empty).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    T = df[treatment].values.astype(int)
    if not covariates:
        p = float(np.mean(T))
        p = float(np.clip(p, 0.01, 0.99))
        return np.full(len(df), p, dtype=float)

    X = StandardScaler().fit_transform(df[covariates].values)
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X, T)
    ps = lr.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    return ps


def trim_by_propensity_quantiles(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    quantiles: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Keep rows whose fitted propensity lies between marginal quantiles
    ``quantiles[0]`` and ``quantiles[1]`` (inclusive on the band).
    """
    ps = _propensity_scores(df, treatment, covariates)
    lo, hi = float(np.quantile(ps, quantiles[0])), float(np.quantile(ps, quantiles[1]))
    mask = (ps >= lo) & (ps <= hi)
    info = {
        "n_before": int(len(df)),
        "n_after": int(mask.sum()),
        "ps_band": [lo, hi],
        "quantiles": list(quantiles),
    }
    out = df.loc[mask].copy().reset_index(drop=True)
    return out, info


def estimate_ace_ipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_boot: int = 1000,
    block_size: int = 12,
    cluster_time_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inverse Probability Weighting estimate of the ATE.

    Treatment can be binary (0/1) or continuous; for continuous treatment
    uses a normalised weighting scheme (Hirano & Imbens 2004).

    If ``cluster_time_col`` is set (binary treatment only), block-bootstrap
    resamples contiguous time blocks and keeps all stacked rows per time.

    Returns dict with keys: ate, ci_low, ci_high, ps_mean, ps_std; binary IPW
    also includes an ``overlap`` diagnostics block.
    """
    T = df[treatment].values
    Y = df[outcome].values
    binary = set(np.unique(T)).issubset({0, 1})

    if binary:
        ps = _propensity_scores(df, treatment, covariates)
        # Horvitz–Thompson estimator
        def _ipw_ate(d: pd.DataFrame) -> float:
            t_ = d[treatment].values
            y_ = d[outcome].values
            ps_ = _propensity_scores(d, treatment, covariates)
            return float(np.mean(t_ * y_ / ps_) - np.mean((1 - t_) * y_ / (1 - ps_)))

        ate = _ipw_ate(df)
        if cluster_time_col:
            ci_lo, ci_hi = block_bootstrap_ci_clustered(
                _ipw_ate, df, cluster_time_col, block_size=block_size, n_boot=n_boot
            )
        else:
            ci_lo, ci_hi = block_bootstrap_ci(_ipw_ate, df, block_size=block_size, n_boot=n_boot)
        wt = np.where(T.astype(int) == 1, 1.0 / ps, 1.0 / (1.0 - ps))
        ess = float((wt.sum() ** 2) / np.sum(wt**2))
        overlap = {
            "ess_ht": ess,
            "ess_over_n": float(ess / max(len(df), 1)),
            "ps_mean": float(ps.mean()),
            "ps_std": float(ps.std()),
            "ps_min": float(ps.min()),
            "ps_max": float(ps.max()),
            "weight_p99": float(np.percentile(wt, 99)),
        }
        return {
            "ate": ate,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "ps_mean": float(ps.mean()),
            "ps_std": float(ps.std()),
            "overlap": overlap,
        }
    else:
        # Continuous treatment: use GPS (generalised propensity score)
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X  = StandardScaler().fit_transform(df[covariates].values)
        mu = Ridge().fit(X, T).predict(X)
        resid = T - mu
        gps   = np.exp(-0.5 * (resid / resid.std()) ** 2) / (resid.std() * np.sqrt(2 * np.pi))
        gps   = np.clip(gps, np.percentile(gps, 1), np.percentile(gps, 99))

        def _gps_ate(d: pd.DataFrame) -> float:
            x_  = StandardScaler().fit_transform(d[covariates].values)
            t_  = d[treatment].values
            y_  = d[outcome].values
            mu_ = Ridge().fit(x_, t_).predict(x_)
            r_  = t_ - mu_
            g_  = np.exp(-0.5 * (r_ / r_.std()) ** 2) / (r_.std() * np.sqrt(2 * np.pi))
            g_  = np.clip(g_, np.percentile(g_, 1), np.percentile(g_, 99))
            return float(np.cov(t_, y_ / g_)[0, 1] / np.var(t_))

        ate = _gps_ate(df)
        ci_lo, ci_hi = block_bootstrap_ci(_gps_ate, df, block_size=block_size, n_boot=n_boot)
        return {"ate": ate, "ci_low": ci_lo, "ci_high": ci_hi}


# ---------------------------------------------------------------------------
# Doubly-robust (regression adjustment)
# ---------------------------------------------------------------------------

def estimate_ace_dr(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_boot: int = 1000,
    block_size: int = 12,
    cluster_time_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Doubly-robust / augmented IPW estimator via econml's LinearDRLearner.
    Consistent if either the propensity or the outcome model is correctly specified.
    Returns dict with keys: ate, ci_low, ci_high.
    """
    from econml.dr import LinearDRLearner
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(df[covariates].values)
    T = df[treatment].values.astype(float)
    Y = df[outcome].values.astype(float)

    est = LinearDRLearner(
        model_propensity=LogisticRegression(max_iter=500, random_state=0),
        model_regression=RidgeCV(),
        cv=5,
        random_state=0,
    )
    est.fit(Y, T, X=X)
    ate = float(est.ate(X))

    def _dr_ate(d: pd.DataFrame) -> float:
        x_ = StandardScaler().fit_transform(d[covariates].values)
        t_ = d[treatment].values.astype(float)
        y_ = d[outcome].values.astype(float)
        e = LinearDRLearner(
            model_propensity=LogisticRegression(max_iter=400, random_state=0),
            model_regression=RidgeCV(),
            cv=3,
            random_state=0,
        )
        e.fit(y_, t_, X=x_)
        return float(e.ate(x_))

    if cluster_time_col:
        ci_lo, ci_hi = block_bootstrap_ci_clustered(
            _dr_ate, df, cluster_time_col, block_size=block_size, n_boot=n_boot
        )
    else:
        ci_lo, ci_hi = block_bootstrap_ci(_dr_ate, df, block_size=block_size, n_boot=n_boot)
    return {"ate": ate, "ci_low": ci_lo, "ci_high": ci_hi}


# ---------------------------------------------------------------------------
# do(ENSO = 0) counterfactual
# ---------------------------------------------------------------------------

def counterfactual_enso_zero(
    panel: pd.DataFrame,
    dag_edges: list[tuple[str, str]],
    treatment_col: str = "nino34",
    outcome_col: str = "tp",
    covariates: Optional[list[str]] = None,
) -> pd.Series:
    """
    SCM-based counterfactual: what would `outcome_col` look like if
    `treatment_col` (ENSO / Niño 3.4) were held at zero for all time steps?

    Implements the abduction–action–prediction procedure (Pearl 2009):
      1. Abduction  – fit a linear SCM from the DAG; compute residuals (noise terms).
      2. Action     – set treatment_col = 0 in the SCM equations.
      3. Prediction – propagate through the modified SCM to obtain Y_cf.

    Uses topological order derived from `dag_edges`.

    Returns a pd.Series of counterfactual outcome values, same index as `panel`.
    """
    from sklearn.linear_model import LinearRegression

    # Structural edges only (caller should already supply a DAG; be defensive).
    dag_edges = [(s, t) for s, t in dag_edges if s != t]

    # Build adjacency: for each node, list its parents (unique, stable order)
    nodes = list(panel.columns)
    parents: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in dag_edges:
        if src in parents and dst in parents and src not in parents[dst]:
            parents[dst].append(src)

    # Topological sort (Kahn's algorithm)
    in_degree = {n: len(parents[n]) for n in nodes}
    queue = [n for n in nodes if in_degree[n] == 0]
    topo  = []
    while queue:
        node = queue.pop(0)
        topo.append(node)
        for n in nodes:
            if node in parents[n]:
                in_degree[n] -= 1
                if in_degree[n] == 0:
                    queue.append(n)

    # Step 1: Abduction — fit linear SCM, compute noise terms
    residuals = pd.DataFrame(index=panel.index)
    models: dict[str, LinearRegression] = {}
    for node in topo:
        pa = parents[node]
        if not pa:
            residuals[node] = panel[node].values
        else:
            X_  = panel[pa].values
            y_  = panel[node].values
            lr  = LinearRegression().fit(X_, y_)
            models[node] = lr
            residuals[node] = y_ - lr.predict(X_)

    # Step 2 & 3: Action + Prediction
    # Set treatment to 0; propagate noise through structural equations
    cf_vals = pd.DataFrame(index=panel.index, columns=nodes, dtype=float)
    for node in topo:
        if node == treatment_col:
            cf_vals[node] = 0.0
        else:
            pa = parents[node]
            if not pa:
                cf_vals[node] = residuals[node].values
            else:
                X_cf = cf_vals[pa].values
                pred = models[node].predict(X_cf)
                cf_vals[node] = pred + residuals[node].values

    return cf_vals[outcome_col].rename(f"{outcome_col}_cf_enso0")
