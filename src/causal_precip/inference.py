"""
src/causal_precip/inference.py
Causal inference estimators: IPW, doubly-robust ACE, block bootstrap CIs,
and the do(ENSO=0) counterfactual.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# Propensity score (IPW)
# ---------------------------------------------------------------------------

def _propensity_scores(
    df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
) -> np.ndarray:
    """
    Logistic regression propensity scores P(T=1 | covariates).
    Falls back to a linear model if sklearn is not available.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(df[covariates].values)
    T = df[treatment].values
    lr = LogisticRegressionCV(cv=5, max_iter=500, random_state=0)
    lr.fit(X, T)
    ps = lr.predict_proba(X)[:, 1]
    # Trim extreme propensity scores (overlap assumption)
    ps = np.clip(ps, 0.01, 0.99)
    return ps


def estimate_ace_ipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_boot: int = 1000,
    block_size: int = 12,
) -> Dict[str, float]:
    """
    Inverse Probability Weighting estimate of the ATE.

    Treatment can be binary (0/1) or continuous; for continuous treatment
    uses a normalised weighting scheme (Hirano & Imbens 2004).

    Returns dict with keys: ate, ci_low, ci_high, ps_mean, ps_std.
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
        ci_lo, ci_hi = block_bootstrap_ci(_ipw_ate, df, block_size=block_size, n_boot=n_boot)
        return {"ate": ate, "ci_low": ci_lo, "ci_high": ci_hi,
                "ps_mean": float(ps.mean()), "ps_std": float(ps.std())}
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
) -> Dict[str, float]:
    """
    Doubly-robust / augmented IPW estimator via econml's LinearDRLearner.
    Consistent if either the propensity or the outcome model is correctly specified.
    Returns dict with keys: ate, ci_low, ci_high.
    """
    from econml.dr import LinearDRLearner
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(df[covariates].values)
    T = df[treatment].values.astype(float)
    Y = df[outcome].values.astype(float)

    est = LinearDRLearner(
        model_propensity=LogisticRegressionCV(cv=5, max_iter=500, random_state=0),
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
        e  = LinearDRLearner(
            model_propensity=LogisticRegressionCV(cv=3, max_iter=300, random_state=0),
            model_regression=RidgeCV(),
            cv=3,
            random_state=0,
        )
        e.fit(y_, t_, X=x_)
        return float(e.ate(x_))

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

    # Build adjacency: for each node, list its parents
    nodes = list(panel.columns)
    parents: dict[str, list[str]] = {n: [] for n in nodes}
    for src, dst in dag_edges:
        if src in parents and dst in parents:
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
