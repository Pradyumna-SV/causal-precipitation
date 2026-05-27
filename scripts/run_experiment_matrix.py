#!/usr/bin/env python3
"""
Run the experiment list in config/experiments_matrix.yaml and append JSON lines to
results/experiment_log.jsonl. Rewrites results/experiments_index.json from the log.

Each experiment sets CAUSAL_PRESET and runs the declared subset of the pipeline.
Inference without discovery copies consensus_dag_*.json from primary results/.

Usage:
    .venv/bin/python scripts/run_experiment_matrix.py
    .venv/bin/python scripts/run_experiment_matrix.py --only extreme_pct_85

Does not replace ``make pipeline`` for the primary release config.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parent.parent
_MATRIX = _REPO / "config" / "experiments_matrix.yaml"
_PRIMARY_RESULTS = _REPO / "results"


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _copy_consensus_dags(dst_results: Path) -> int:
    dst_results.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(_PRIMARY_RESULTS.glob("consensus_dag_*.json")):
        shutil.copy2(p, dst_results / p.name)
        n += 1
    return n


def _ci_excludes_zero(lo: float | None, hi: float | None) -> bool | None:
    if lo is None or hi is None:
        return None
    return not (float(lo) <= 0.0 <= float(hi))


def _summarize_ace_all_regions(root: Path) -> dict[str, Any] | None:
    p = root / "ace_all_regions.json"
    if not p.exists():
        return None
    with open(p) as f:
        raw = json.load(f)
    out: dict[str, Any] = {}
    for reg, a in raw.items():
        ipw, dr = a.get("ipw") or {}, a.get("dr") or {}
        out[reg] = {
            "ace_profile": a.get("ace_profile"),
            "treatment": a.get("treatment"),
            "outcome": a.get("outcome"),
            "ipw_ate": ipw.get("ate"),
            "ipw_ci95": [ipw.get("ci_low"), ipw.get("ci_high")],
            "ipw_ci_excludes_zero": _ci_excludes_zero(ipw.get("ci_low"), ipw.get("ci_high")),
            "dr_ate": dr.get("ate"),
            "dr_ci95": (
                [dr.get("ci_low"), dr.get("ci_high")] if dr.get("ci_low") is not None else None
            ),
            "dr_ci_excludes_zero": _ci_excludes_zero(dr.get("ci_low"), dr.get("ci_high"))
            if dr.get("ci_low") is not None
            else None,
        }
    return out


def _summarize_pooled(root: Path) -> dict[str, Any] | None:
    p = root / "pooled_ace.json"
    if not p.exists():
        return None
    with open(p) as f:
        z = json.load(f)
    ipw, dr = z.get("ipw") or {}, z.get("dr") or {}
    return {
        "artifact": "pooled_ace.json",
        "ipw_ate": ipw.get("ate"),
        "ipw_ci95": [ipw.get("ci_low"), ipw.get("ci_high")],
        "ipw_ci_excludes_zero": _ci_excludes_zero(ipw.get("ci_low"), ipw.get("ci_high")),
        "dr_ate": dr.get("ate"),
        "dr_ci95": [dr.get("ci_low"), dr.get("ci_high")],
        "dr_ci_excludes_zero": _ci_excludes_zero(dr.get("ci_low"), dr.get("ci_high")),
    }


def _preset_results_dir(preset: str) -> Path | None:
    pp = _REPO / "config" / "presets" / f"{preset}.yaml"
    if not pp.exists():
        return None
    with open(pp) as f:
        cfg = yaml.safe_load(f) or {}
    rel = (cfg.get("paths") or {}).get("results")
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else (_REPO / p).resolve()


def _run(cmd: list[str], env: dict[str, str]) -> int:
    print("+", " ".join(cmd), file=sys.stderr)
    return int(subprocess.run(cmd, cwd=_REPO, env=env).returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Run a single experiment id from the matrix.")
    parser.add_argument(
        "--fresh-log",
        action="store_true",
        help="Truncate experiment_log.jsonl before running (start a new log).",
    )
    args = parser.parse_args()

    with open(_MATRIX) as f:
        matrix = yaml.safe_load(f) or {}

    experiments = matrix.get("experiments") or []
    if args.only:
        experiments = [e for e in experiments if e.get("id") == args.only]
        if not experiments:
            print(f"No experiment with id={args.only!r}", file=sys.stderr)
            return 1

    log_path = _REPO / (matrix.get("log_path") or "results/experiment_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = _REPO / (matrix.get("summary_path") or "results/experiments_index.json")
    if args.fresh_log and log_path.exists():
        log_path.unlink()

    py = os.environ.get("PYTHON", str(_REPO / ".venv" / "bin" / "python"))
    if not Path(py).exists():
        py = sys.executable

    for exp in experiments:
        eid = exp["id"]
        desc = exp.get("description", "")
        preset = exp.get("causal_preset", "").strip() or "primary"
        steps = list(exp.get("steps") or [])
        base_env = {**os.environ, "CAUSAL_PRESET": preset}
        started = _iso()
        rc: dict[str, int] = {}
        fail = False

        print(f"\n========== EXPERIMENT {eid} (preset={preset}) ==========", file=sys.stderr)

        if "preprocess" in steps:
            rc["preprocess"] = _run([py, "scripts/02_preprocess.py"], base_env)
            fail = fail or rc["preprocess"] != 0
        if "discovery" in steps and not fail:
            rc["discovery"] = _run([py, "scripts/03_causal_discovery.py"], base_env)
            fail = fail or rc["discovery"] != 0
        if "inference" in steps and not fail:
            out_dir = _preset_results_dir(preset)
            if out_dir and "discovery" not in steps:
                n = _copy_consensus_dags(out_dir)
                if n == 0:
                    print("warning: no consensus DAGs copied to", out_dir, file=sys.stderr)
            rc["inference"] = _run([py, "scripts/04_causal_inference.py"], base_env)
            fail = fail or rc["inference"] != 0
        if "pooled_ace" in steps and not fail:
            rc["pooled_ace"] = _run([py, "scripts/pooled_ace.py"], base_env)
            fail = fail or rc["pooled_ace"] != 0

        finished = _iso()
        results_dir = _preset_results_dir(preset)
        ace_sum = _summarize_ace_all_regions(results_dir) if results_dir else None
        pooled_sum = None
        if "pooled_ace" in steps:
            pooled_sum = _summarize_pooled(_REPO / "results")

        entry = {
            "experiment_id": eid,
            "description": desc,
            "causal_preset": preset,
            "steps": steps,
            "started_at": started,
            "finished_at": finished,
            "status": "ok" if not fail else "fail",
            "returncodes": rc,
            "results_dir": str(results_dir) if results_dir else None,
            "ace_summary": ace_sum,
            "pooled_summary": pooled_sum,
        }
        with open(log_path, "a") as lf:
            lf.write(json.dumps(entry) + "\n")

        print(json.dumps({"logged": eid, "status": entry["status"]}, indent=2))

    rows: list[dict[str, Any]] = []
    if log_path.exists():
        with open(log_path) as lf:
            for line in lf:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    with open(summary_path, "w") as sf:
        json.dump(
            {
                "generated_at": _iso(),
                "n_entries": len(rows),
                "experiments": rows,
            },
            sf,
            indent=2,
        )
    print(json.dumps({"wrote_summary": str(summary_path), "n_log_entries": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
