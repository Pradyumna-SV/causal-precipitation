#!/usr/bin/env python3
"""
Temporal holdout validation for the frozen primary causal claim.

This is deliberately narrow: it validates the registry's gate-B region and the
primary ACE estimand already written by script 04. Nuisance models are fit on an
early ONDJFM window, then the ATE is evaluated on a disjoint later ONDJFM window.

Output: results/holdout_validation.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from causal_precip import load_config, processed_path, results_path  # noqa: E402
from causal_precip.inference import block_bootstrap_ci, filter_panel_dataframe_by_month  # noqa: E402


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


def _time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        if "time" in idx.names:
            return pd.DatetimeIndex(pd.to_datetime(idx.get_level_values("time")))
        return pd.DatetimeIndex(pd.to_datetime(idx.get_level_values(0)))
    return pd.DatetimeIndex(pd.to_datetime(idx))


def _load_panel(region: str, cfg: dict) -> pd.DataFrame:
    path = processed_path(f"panel_{region}.nc", cfg)
    ds = xr.open_dataset(path)
    try:
        return ds.to_dataframe().dropna()
    finally:
        ds.close()


def _fit_eval_propensity(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    treatment: str,
    covariates: list[str],
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t_train = train[treatment].values.astype(int)
    if len(np.unique(t_train)) < 2:
        raise ValueError("Training window must contain both treatment classes.")

    if not covariates:
        p = float(np.mean(t_train))
        return np.full(len(eval_df), np.clip(p, 0.01, 0.99), dtype=float)

    scaler = StandardScaler().fit(train[covariates].values)
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(scaler.transform(train[covariates].values), t_train)
    ps = lr.predict_proba(scaler.transform(eval_df[covariates].values))[:, 1]
    return np.clip(ps, 0.01, 0.99)


def _estimate_holdout_ipw(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_boot: int,
    block_size: int,
) -> dict[str, Any]:
    ps = _fit_eval_propensity(train, eval_df, treatment, covariates)
    work = eval_df.copy()
    work["_ps_holdout"] = ps

    def _ate(d: pd.DataFrame) -> float:
        t = d[treatment].values.astype(int)
        y = d[outcome].values.astype(float)
        p = d["_ps_holdout"].values.astype(float)
        return float(np.mean(t * y / p) - np.mean((1 - t) * y / (1 - p)))

    ate = _ate(work)
    ci_lo, ci_hi = block_bootstrap_ci(_ate, work, block_size=block_size, n_boot=n_boot)
    t_eval = work[treatment].values.astype(int)
    wt = np.where(t_eval == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    ess = float((wt.sum() ** 2) / np.sum(wt**2))
    return {
        "ate": ate,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "ci_excludes_zero": _ci_excludes_zero(ci_lo, ci_hi),
        "ps_mean": float(ps.mean()),
        "ps_std": float(ps.std()),
        "overlap": {
            "ess_ht": ess,
            "ess_over_n": float(ess / max(len(work), 1)),
            "ps_min": float(ps.min()),
            "ps_max": float(ps.max()),
            "weight_p99": float(np.percentile(wt, 99)),
        },
    }


def _estimate_holdout_dr(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_boot: int,
    block_size: int,
) -> dict[str, Any]:
    from econml.dr import LinearDRLearner
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler

    t_train = train[treatment].values.astype(float)
    if len(np.unique(t_train)) < 2:
        raise ValueError("Training window must contain both treatment classes.")

    if covariates:
        scaler = StandardScaler().fit(train[covariates].values)
        x_train = scaler.transform(train[covariates].values)

        def _x(d: pd.DataFrame) -> np.ndarray:
            return scaler.transform(d[covariates].values)

    else:
        x_train = np.zeros((len(train), 1))

        def _x(d: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(d), 1))

    est = LinearDRLearner(
        model_propensity=LogisticRegression(max_iter=500, random_state=0),
        model_regression=RidgeCV(),
        cv=5,
        random_state=0,
    )
    est.fit(train[outcome].values.astype(float), t_train, X=x_train)

    def _ate(d: pd.DataFrame) -> float:
        return float(est.ate(_x(d)))

    ate = _ate(eval_df)
    ci_lo, ci_hi = block_bootstrap_ci(_ate, eval_df, block_size=block_size, n_boot=n_boot)
    return {
        "ate": ate,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "ci_excludes_zero": _ci_excludes_zero(ci_lo, ci_hi),
        "note": "Nuisance/DR models fit on train window; CI bootstraps evaluation-window months with nuisance models fixed.",
    }


def main() -> int:
    cfg = load_config()
    reg_path = _REPO / "config" / "analysis_registry.yaml"
    with open(reg_path) as f:
        registry = yaml.safe_load(f) or {}

    hold_cfg = (cfg.get("validation") or {}).get("temporal_holdout") or {}
    region = hold_cfg.get("region") or (registry.get("primary") or {}).get(
        "gate_b_region", "pacific_northwest"
    )
    train_start = hold_cfg.get("train_start") or cfg["date_range"]["start"]
    train_end = hold_cfg.get("train_end", "1999-12")
    eval_start = hold_cfg.get("eval_start", "2000-01")
    eval_end = hold_cfg.get("eval_end") or cfg["date_range"]["end"]

    inf = cfg.get("inference") or {}
    season_months = inf.get("season_months")
    n_boot = int(hold_cfg.get("n_boot", inf.get("n_boot", 2000)))
    block_size = int(inf.get("block_size", 12))

    rdir = results_path("", cfg).resolve()
    ace_path = rdir / "ace_all_regions.json"
    if not ace_path.exists():
        print(f"Missing {ace_path}; run inference first.", file=sys.stderr)
        return 1
    with open(ace_path) as f:
        ace_all = json.load(f)
    if region not in ace_all:
        print(f"Region {region!r} missing from {ace_path}", file=sys.stderr)
        return 1

    primary = ace_all[region]
    treatment = primary.get("treatment", "sst_warm")
    outcome = primary.get("outcome", "tp_extreme")
    covariates = list(primary.get("adjustment_set") or [])
    if treatment != "sst_warm" or outcome != "tp_extreme":
        print(
            f"Holdout script expects primary warm-SST → tp_extreme, got {treatment} → {outcome}",
            file=sys.stderr,
        )
        return 1

    df0 = _load_panel(region, cfg)
    df = filter_panel_dataframe_by_month(df0, season_months)
    if treatment not in df.columns:
        df[treatment] = (df["sst"] >= 0.5).astype(int)
    t = _time_index(df)
    train_mask = (t >= pd.Timestamp(train_start)) & (t <= pd.Timestamp(train_end))
    eval_mask = (t >= pd.Timestamp(eval_start)) & (t <= pd.Timestamp(eval_end))
    train = df.loc[train_mask].copy()
    eval_df = df.loc[eval_mask].copy()

    if len(train) < block_size * 2 or len(eval_df) < block_size * 2:
        print(
            f"Holdout windows too small after seasonal filtering: train={len(train)}, eval={len(eval_df)}",
            file=sys.stderr,
        )
        return 1
    if len(np.unique(eval_df[treatment].values.astype(int))) < 2:
        print("Evaluation window must contain both treatment classes.", file=sys.stderr)
        return 1

    ipw = _estimate_holdout_ipw(
        train,
        eval_df,
        treatment=treatment,
        outcome=outcome,
        covariates=covariates,
        n_boot=n_boot,
        block_size=block_size,
    )
    dr = _estimate_holdout_dr(
        train,
        eval_df,
        treatment=treatment,
        outcome=outcome,
        covariates=covariates,
        n_boot=n_boot,
        block_size=block_size,
    )

    full_ipw = primary.get("ipw") or {}
    full_dr = primary.get("dr") or {}
    directional = (
        np.sign(float(ipw["ate"])) == np.sign(float(full_ipw.get("ate", 0.0)))
        and np.sign(float(dr["ate"])) == np.sign(float(full_dr.get("ate", 0.0)))
        and np.sign(float(ipw["ate"])) == np.sign(float(dr["ate"]))
    )
    strict = bool(directional and ipw["ci_excludes_zero"] and dr["ci_excludes_zero"])

    out = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "estimand": {
            "region": region,
            "treatment": treatment,
            "outcome": outcome,
            "season_months": season_months,
            "adjustment_set": covariates,
            "frozen_from": str(ace_path),
        },
        "split": {
            "train_start": train_start,
            "train_end": train_end,
            "eval_start": eval_start,
            "eval_end": eval_end,
            "n_train_rows": int(len(train)),
            "n_eval_rows": int(len(eval_df)),
        },
        "estimation": {
            "n_boot": n_boot,
            "block_size": block_size,
            "nuisance_policy": "Fit propensity / DR nuisance models on train window only; evaluate ATE and bootstrap CI on held-out eval window.",
        },
        "full_sample_reference": {
            "ipw": {
                "ate": full_ipw.get("ate"),
                "ci95": [full_ipw.get("ci_low"), full_ipw.get("ci_high")],
            },
            "dr": {
                "ate": full_dr.get("ate"),
                "ci95": [full_dr.get("ci_low"), full_dr.get("ci_high")],
            },
        },
        "holdout": {
            "ipw": ipw,
            "dr": dr,
        },
        "confirmation": {
            "directional_replication": bool(directional),
            "strict_statistical_replication": strict,
            "interpretation": (
                "Strict replication requires IPW and DR held-out CIs to exclude zero in the full-sample direction; "
                "directional replication only requires held-out point estimates to agree in sign."
            ),
        },
    }
    out = _jsonify(out)

    path = results_path("holdout_validation.json", cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(
        json.dumps(
            {
                "wrote": str(path),
                "directional_replication": out["confirmation"]["directional_replication"],
                "strict_statistical_replication": out["confirmation"]["strict_statistical_replication"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
