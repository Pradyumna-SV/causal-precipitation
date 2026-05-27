#!/usr/bin/env python3
"""
Programmatic verification of repository outputs: data → results → figures.

Uses ``load_config`` + ``processed_path`` / ``raw_path`` to list on-disk inputs
so checks match the pipeline even when IDE search skips gitignored ``data/**``.

Exit codes: 0 = all checks passed, 1 = one or more failures.

Usage (from repo root, after .venv activated):
    .venv/bin/python scripts/verify_pipeline.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from causal_precip import load_config, raw_path, processed_path, results_path, figures_path  # noqa: E402
from causal_precip.inference import (  # noqa: E402
    consensus_records_to_panel_dag_edges,
    identification_dag_edges,
    parse_backbone_edges_yaml,
)


def _fail(report: dict, errors: list, msg: str) -> None:
    errors.append(msg)
    report.setdefault("failures", []).append(msg)


def _expect_months(cfg: dict) -> int:
    s = cfg["date_range"]["start"]
    e = cfg["date_range"]["end"]
    sy, sm = int(s[:4]), int(s[5:7])
    ey, em = int(e[:4]), int(e[5:7])
    return (ey - sy) * 12 + (em - sm + 1)


def _resolve_data_layout(cfg: dict) -> dict[str, Any]:
    """
    List on-disk inputs using the same ``processed_path`` / ``raw_path`` helpers as
    scripts 02–05. This is the authoritative check (IDE/workspace search may omit
    gitignored ``data/**``).
    """
    proc_dir = processed_path("", cfg).resolve()
    raw_dir = raw_path("", cfg).resolve()

    def _entries(glob_pat: str, directory: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in sorted(directory.glob(glob_pat)):
            if p.is_file():
                try:
                    st = p.stat()
                    out.append({"path": str(p), "name": p.name, "bytes": st.st_size})
                except OSError:
                    out.append({"path": str(p), "name": p.name, "bytes": None})
        return out

    return {
        "processed_dir": str(proc_dir),
        "raw_dir": str(raw_dir),
        "panel_netcdfs": _entries("panel_*.nc", proc_dir),
        "era5_raw_netcdfs": _entries("era5_*.nc", raw_dir),
    }


def _verify_raw(cfg: dict, report: dict, errors: list) -> None:
    stem = f"era5_single_{cfg['date_range']['start'].replace('-', '')}_{cfg['date_range']['end'].replace('-', '')}.nc"
    for name in (
        f"era5_single_{cfg['date_range']['start'].replace('-', '')}_{cfg['date_range']['end'].replace('-', '')}.nc",
        f"era5_plev_{cfg['date_range']['start'].replace('-', '')}_{cfg['date_range']['end'].replace('-', '')}.nc",
        f"era5_nino34_{cfg['date_range']['start'].replace('-', '')}_{cfg['date_range']['end'].replace('-', '')}.nc",
    ):
        p = raw_path(name, cfg)
        if not p.exists():
            _fail(report, errors, f"Missing raw file: {p}")
            continue
        if p.stat().st_size < 10_000:
            _fail(report, errors, f"Raw file suspiciously small: {p}")


def _verify_panels(cfg: dict, report: dict, errors: list) -> None:
    n_exp = _expect_months(cfg)
    pct = float(cfg.get("extreme_precip_percentile", 90))
    exp_frac = (100.0 - pct) / 100.0
    panel_stats: dict[str, Any] = {}

    for region in cfg["regions"]:
        path = processed_path(f"panel_{region}.nc", cfg)
        if not path.exists():
            _fail(report, errors, f"Missing panel: {path}")
            continue
        ds = xr.open_dataset(path)
        try:
            if ds.sizes.get("time") != n_exp:
                _fail(
                    report,
                    errors,
                    f"{region}: time length {ds.sizes.get('time')} != expected {n_exp}",
                )
            for v in ("tp", "sst", "nino34", "tp_extreme", "z500"):
                if v not in ds:
                    _fail(report, errors, f"{region}: missing variable {v}")
            tp = ds["tp"].values.astype(float)
            if np.isnan(tp).any():
                _fail(report, errors, f"{region}: NaN in tp")
            if abs(tp.mean()) > 1e-3:
                _fail(
                    report,
                    errors,
                    f"{region}: tp anomaly mean {tp.mean():.2e} far from 0 (check anomalies)",
                )
            ex = ds["tp_extreme"].values
            frac = float(np.mean(ex))
            if abs(frac - exp_frac) > 0.02:
                _fail(
                    report,
                    errors,
                    f"{region}: extreme fraction {frac:.3f} expected ~{exp_frac:.3f}",
                )
            if "threshold_m" not in ds["tp_extreme"].attrs and "threshold_mm_day" in ds["tp_extreme"].attrs:
                _fail(
                    report,
                    errors,
                    f"{region}: tp_extreme still uses deprecated threshold_mm_day attr",
                )
            panel_stats[region] = {
                "tp_std": float(np.nanstd(tp)),
                "extreme_frac": frac,
                "nino_tp_corr": float(np.corrcoef(ds["nino34"].values, tp)[0, 1]),
            }
        finally:
            ds.close()
    report["panels"] = panel_stats


def _verify_discovery_and_structure(cfg: dict, report: dict, errors: list) -> None:
    import networkx as nx

    for region in cfg["regions"]:
        cpath = results_path(f"consensus_dag_{region}.json", cfg)
        spath = results_path(f"structural_dag_{region}.json", cfg)
        if not cpath.exists():
            _fail(report, errors, f"Missing {cpath}")
            continue
        with open(cpath) as f:
            cons = json.load(f)
        expected = consensus_records_to_panel_dag_edges(cons["consensus_edges"])
        if not spath.exists():
            _fail(
                report,
                errors,
                f"Missing structural_dag (re-run 04): {spath}",
            )
            continue
        with open(spath) as f:
            saved = json.load(f)
        got = [(e["source"], e["target"]) for e in saved["edges"]]
        if sorted(got) != sorted(expected):
            _fail(
                report,
                errors,
                f"{region}: structural_dag mismatch recomputation — got {got} expected {expected}",
            )
        G = nx.DiGraph()
        G.add_edges_from(got)
        if not nx.is_directed_acyclic_graph(G):
            _fail(report, errors, f"{region}: structural DAG is not acyclic")
        if any(u == v for u, v in got):
            _fail(report, errors, f"{region}: structural DAG has self-loop")

        ipath = results_path(f"identification_dag_{region}.json", cfg)
        if not ipath.exists():
            _fail(
                report,
                errors,
                f"Missing identification_dag (re-run 04): {ipath}",
            )
            continue
        backbone = parse_backbone_edges_yaml(
            (cfg.get("identification_backbone") or {}).get("edges"),
        )
        expected_id = identification_dag_edges(expected, backbone)
        with open(ipath) as f:
            id_saved = json.load(f)
        got_id = [(e["source"], e["target"]) for e in id_saved["edges"]]
        if sorted(got_id) != sorted(expected_id):
            _fail(
                report,
                errors,
                f"{region}: identification_dag mismatch — got {got_id} expected {expected_id}",
            )
            continue
        Gi = nx.DiGraph()
        Gi.add_edges_from(got_id)
        if not nx.is_directed_acyclic_graph(Gi):
            _fail(report, errors, f"{region}: identification DAG is not acyclic")
        if not nx.has_path(Gi, "sst", "tp_extreme"):
            _fail(
                report,
                errors,
                f"{region}: no directed path sst → tp_extreme in identification DAG (backdoor may be vacuous)",
            )


def _verify_inference(cfg: dict, report: dict, errors: list) -> None:
    for region in cfg["regions"]:
        ap = results_path(f"ace_{region}.json", cfg)
        if not ap.exists():
            _fail(report, errors, f"Missing {ap}")
            continue
        with open(ap) as f:
            ace = json.load(f)
        ipw_b = ace.get("ipw") or {}
        if "ate" not in ipw_b or not math.isfinite(float(ipw_b["ate"])):
            _fail(report, errors, f"{region}: missing or non-finite ipw.ate")
            continue

        dr_b = ace.get("dr") or {}
        if dr_b.get("ate") is None:
            if not isinstance(dr_b.get("note"), str):
                _fail(report, errors, f"{region}: dr.ate null without explanatory note")
                continue
        else:
            if not math.isfinite(float(dr_b["ate"])):
                _fail(report, errors, f"{region}: non-finite dr ATE")
                continue

        sens = ace.get("sensitivity", {})
        smid = sens.get("method_id")
        if smid == "crude_gamma_scaling_subgroup_means":
            if sens.get("is_rosenbaum_bounds") is not False:
                _fail(report, errors, f"{region}: sensitivity must set is_rosenbaum_bounds false")
        elif smid == "not_applicable":
            pass
        else:
            _fail(
                report,
                errors,
                f"{region}: sensitivity unexpected method_id {smid!r} (re-run 04)",
            )
            continue

        ev = ace.get("e_value") or {}
        if ev.get("reference") != "not_applicable":
            for k in (
                "crude_risk_ratio",
                "crude_e_value_min",
                "adjusted_odds_ratio",
                "adjusted_or_e_value_min",
            ):
                if k not in ev:
                    _fail(report, errors, f"{region}: ace missing e_value.{k} (re-run 04)")
                    break

        if smid == "crude_gamma_scaling_subgroup_means" and ace.get("outcome") == "tp_extreme":
            crude = sens.get("crude_risk_difference")
            ipw_s = sens.get("ipw_ate")
            if crude is None or ipw_s is None:
                _fail(report, errors, f"{region}: sensitivity missing crude_risk_difference or ipw_ate")
            else:
                # Adjusted estimate may legitimately differ; flag only absurd identity bugs
                if abs(float(crude)) > 1.0 or abs(float(ipw_s)) > 1.0:
                    _fail(
                        report,
                        errors,
                        f"{region}: crude or ipw ATE magnitude > 1 on binary outcome (unlikely)",
                    )

        cf_path = results_path(f"counterfactual_tp_{region}.nc", cfg)
        if not cf_path.exists():
            _fail(report, errors, f"Missing {cf_path}")
            continue
        ds = xr.open_dataset(cf_path)
        try:
            fa = ds["tp_factual"].values.astype(float)
            cf = ds["tp_cf_enso0"].values.astype(float)
            if np.isnan(fa).any() or np.isnan(cf).any():
                _fail(report, errors, f"{region}: NaN in counterfactual netcdf")
            exp_n = ds.attrs.get("inference_n_times")
            if exp_n is not None:
                if len(fa) != int(exp_n):
                    _fail(
                        report,
                        errors,
                        f"{region}: counterfactual length {len(fa)} != inference_n_times {exp_n}",
                    )
            elif len(fa) != _expect_months(cfg):
                _fail(report, errors, f"{region}: counterfactual time dim mismatch")
            rmse = float(np.sqrt(np.mean((fa - cf) ** 2)))
            mad = float(np.mean(np.abs(fa - cf)))
            report.setdefault("counterfactual", {})[region] = {
                "rmse_factual_vs_cf": rmse,
                "mean_abs_delta": mad,
            }
            tp_std = report.get("panels", {}).get(region, {}).get("tp_std")
            if tp_std and mad < max(1e-9, 0.0025 * tp_std):
                _fail(
                    report,
                    errors,
                    f"{region}: do(ENSO=0) mean |Δtp| ~{mad:.2e} m (<0.25% of tp std {tp_std:.2e}); "
                    "SCM shift negligible.",
                )
        finally:
            ds.close()


def _verify_figures(cfg: dict, report: dict, errors: list) -> None:
    min_bytes = 3000
    for fn in (
        "fig1_domain_map.pdf",
        "fig2_enso_precip_ts.pdf",
        "fig3_pcmci_graphs.pdf",
        "fig4_varlingam_heatmaps.pdf",
        "fig5_ace_comparison.pdf",
        "fig6_counterfactual_cdfs.pdf",
        "fig7_primary_holdout_summary.pdf",
    ):
        p = figures_path(fn, cfg)
        if not p.exists():
            _fail(report, errors, f"Missing figure {p}")
            continue
        if p.stat().st_size < min_bytes:
            _fail(report, errors, f"Figure too small (corrupt?): {p}")


def _verify_robustness_report(cfg: dict, report: dict, errors: list) -> None:
    """
    If results/robustness_report.json exists (from scripts/robustness_bundle.py), validate schema.
    Absence is allowed; malformed file is not.
    """
    path = results_path("robustness_report.json", cfg)
    if not path.exists():
        report["robustness_bundle"] = {"status": "absent"}
        return
    try:
        with open(path) as f:
            rr = json.load(f)
    except json.JSONDecodeError as exc:
        _fail(report, errors, f"robustness_report.json invalid JSON: {exc}")
        return
    if rr.get("schema_version") != 1:
        _fail(
            report,
            errors,
            f"robustness_report.json schema_version must be 1, got {rr.get('schema_version')!r}",
        )
        return
    for key in ("gates", "checks", "gate_b_region"):
        if key not in rr:
            _fail(report, errors, f"robustness_report.json missing required key {key!r}")
    report["robustness_bundle"] = {"status": "ok", "path": str(path), "gate_c_pass": rr.get("gates", {}).get("C_identification_stress", {}).get("pass")}


def _verify_holdout_validation(cfg: dict, report: dict, errors: list) -> None:
    """
    Validate temporal holdout artifact shape when present.

    A weak or non-replicating holdout is a scientific result, not a pipeline
    failure; this check only fails malformed / non-finite outputs.
    """
    path = results_path("holdout_validation.json", cfg)
    if not path.exists():
        report["temporal_holdout"] = {"status": "absent"}
        return
    try:
        with open(path) as f:
            hv = json.load(f)
    except json.JSONDecodeError as exc:
        _fail(report, errors, f"holdout_validation.json invalid JSON: {exc}")
        return
    if hv.get("schema_version") != 1:
        _fail(
            report,
            errors,
            f"holdout_validation.json schema_version must be 1, got {hv.get('schema_version')!r}",
        )
        return
    for key in ("estimand", "split", "holdout", "confirmation"):
        if key not in hv:
            _fail(report, errors, f"holdout_validation.json missing required key {key!r}")
    split = hv.get("split") or {}
    if int(split.get("n_train_rows", 0) or 0) < 24:
        _fail(report, errors, "holdout_validation.json has too few train rows")
    if int(split.get("n_eval_rows", 0) or 0) < 24:
        _fail(report, errors, "holdout_validation.json has too few eval rows")
    holdout = hv.get("holdout") or {}
    for est_name in ("ipw", "dr"):
        est = holdout.get(est_name) or {}
        for key in ("ate", "ci_low", "ci_high"):
            val = est.get(key)
            if val is None or not math.isfinite(float(val)):
                _fail(report, errors, f"holdout_validation.json {est_name}.{key} is missing/non-finite")
    report["temporal_holdout"] = {
        "status": "ok",
        "path": str(path),
        "directional_replication": (hv.get("confirmation") or {}).get("directional_replication"),
        "strict_statistical_replication": (hv.get("confirmation") or {}).get("strict_statistical_replication"),
    }


def _verify_feedback_diagnostics(cfg: dict, report: dict, errors: list) -> None:
    """
    Validate reviewer-response diagnostics when present.

    These diagnostics are substantive sensitivity results, so sign changes or
    null intervals are not failures. Only malformed/non-finite artifacts fail.
    """
    required = {
        "baseline_logistic.json": ("models",),
        "backbone_sensitivity.json": ("variants",),
        "projection_lag_sensitivity.json": ("variants",),
        "holdout_interval_audit.json": ("ipw_ci_width", "dr_ci_width"),
    }
    summary: dict[str, Any] = {}
    for fn, keys in required.items():
        path = results_path(fn, cfg)
        if not path.exists():
            _fail(report, errors, f"Missing diagnostic artifact: {path}")
            continue
        try:
            with open(path) as f:
                obj = json.load(f)
        except json.JSONDecodeError as exc:
            _fail(report, errors, f"{fn} invalid JSON: {exc}")
            continue
        if obj.get("schema_version") != 1:
            _fail(report, errors, f"{fn} schema_version must be 1, got {obj.get('schema_version')!r}")
            continue
        for key in keys:
            if key not in obj:
                _fail(report, errors, f"{fn} missing required key {key!r}")
        if "models" in obj:
            for name, model in (obj.get("models") or {}).items():
                rd = model.get("risk_difference")
                ci = model.get("ci95") or []
                if rd is None or not math.isfinite(float(rd)) or len(ci) != 2:
                    _fail(report, errors, f"{fn} model {name} missing/non-finite risk difference or CI")
        if "variants" in obj:
            for name, variant in (obj.get("variants") or {}).items():
                for est_name in ("ipw", "dr"):
                    est = variant.get(est_name) or {}
                    for key in ("ate", "ci_low", "ci_high"):
                        val = est.get(key)
                        if val is None or not math.isfinite(float(val)):
                            _fail(report, errors, f"{fn} variant {name} {est_name}.{key} missing/non-finite")
        if fn == "holdout_interval_audit.json":
            for key in ("ipw_ci_width", "dr_ci_width"):
                val = obj.get(key)
                if val is None or not math.isfinite(float(val)) or float(val) <= 0:
                    _fail(report, errors, f"{fn} {key} missing/non-positive")
        summary[fn] = {"status": "ok", "path": str(path)}
    report["feedback_diagnostics"] = summary


def _contribution_assessment() -> dict[str, Any]:
    return {
        "algorithmic_novelty": (
            "low: PCMCI+, VARLiNGAM, IPW, DR, linear SCM are established; "
            "no new identification theorem or estimator class."
        ),
        "legitimate_contribution_axes": [
            "End-to-end reproducible pipeline: ERA5 western US extremes, regional aggregation, discovery→panel-DAG projection, ACE+counterfactual.",
            "Explicit lag-0 acyclic projection from time-series discovery to monthly panel (documents a standard tradeoff rather than hiding it).",
            "Engineering fixes material to correctness (duplicate CDS monthly timestamps, month-coordinate merge, cyclic/self-loop SCM pathologies).",
        ],
        "framing_for_venue": (
            "Position as rigorous applied causal-climate case study + open implementation; "
            "claims must not oversell graph recovery as ground truth or reuse mislabeled sensitivity."
        ),
        "integrity_gate": (
            "Verification script must pass before claiming 'validated pipeline' in a paper."
        ),
    }


def main() -> int:
    cfg = load_config()
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config_env": cfg.get("_env", "local"),
        "expected_months": _expect_months(cfg),
        "data_layout": _resolve_data_layout(cfg),
        "contribution_assessment": _contribution_assessment(),
    }
    errors: list[str] = []

    _verify_raw(cfg, report, errors)
    _verify_panels(cfg, report, errors)
    _verify_discovery_and_structure(cfg, report, errors)
    _verify_inference(cfg, report, errors)
    _verify_figures(cfg, report, errors)
    _verify_robustness_report(cfg, report, errors)
    _verify_holdout_validation(cfg, report, errors)
    _verify_feedback_diagnostics(cfg, report, errors)

    out_path = results_path("verification_report.json", cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report["status"] = "PASS" if not errors else "FAIL"
    report["error_count"] = len(errors)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"status": report["status"], "errors": errors, "report": str(out_path)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
