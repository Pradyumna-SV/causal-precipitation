#!/usr/bin/env python3
"""
Run closed-list exploratory configs (CAUSAL_PRESET) via subprocess.

Each preset in config/analysis_registry.yaml with requires_pipeline lists the
steps to invoke. This script does **not** overwrite results/ unless a preset
points paths.results there — presets usually write under results/grid/<id>/.

Usage (from repo root):
    .venv/bin/python scripts/run_search_grid.py
    .venv/bin/python scripts/run_search_grid.py --preset overlap_trim

Requires: same Python as Makefile (.venv recommended).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_REGISTRY = _REPO / "config" / "analysis_registry.yaml"
PRIMARY_RESULTS = _REPO / "results"


def _copy_consensus_from_primary(target_results_dir: Path) -> None:
    """
    Inference-only presets use a separate ``paths.results`` subfolder; discovery
    outputs consensus JSONs only under the primary ``results/``. Copy DAGs so
    ``04_causal_inference.py`` can run without re-discovering.
    """
    target_results_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(PRIMARY_RESULTS.glob("consensus_dag_*.json")):
        shutil.copy2(p, target_results_dir / p.name)
        n += 1
    if n == 0:
        print("warning: no consensus_dag_*.json in primary results/", file=sys.stderr)


def _load_registry() -> dict:
    import yaml

    with open(_REGISTRY) as f:
        return yaml.safe_load(f)


def _run(cmd: list[str], env: dict[str, str]) -> int:
    print("+", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, cwd=_REPO, env=env)
    return int(r.returncode)


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        help="Run a single registry preset id (e.g. overlap_trim) instead of all exploratory.",
    )
    args = parser.parse_args()

    reg = _load_registry()
    exploratory = reg.get("exploratory_presets") or []
    if args.preset:
        exploratory = [e for e in exploratory if e.get("id") == args.preset]
        if not exploratory:
            print(f"Unknown preset id {args.preset!r} in registry.", file=sys.stderr)
            return 1

    py = os.environ.get("PYTHON", str(_REPO / ".venv" / "bin" / "python"))
    if not Path(py).exists():
        py = sys.executable

    failures = 0
    for entry in exploratory:
        pid = entry["id"]
        yaml_name = entry.get("yaml", pid)
        needs = entry.get("requires_pipeline") or []
        base_env = {**os.environ, "CAUSAL_PRESET": yaml_name}
        print(f"\n=== Grid run: {pid} (CAUSAL_PRESET={yaml_name}) ===", file=sys.stderr)

        if "preprocess" in needs:
            failures += _run([py, "scripts/02_preprocess.py"], base_env)
        if "discovery" in needs:
            failures += _run([py, "scripts/03_causal_discovery.py"], base_env)
        if "inference" in needs:
            out_dir = None
            preset_path = _REPO / "config" / "presets" / f"{yaml_name}.yaml"
            if preset_path.exists():
                with open(preset_path) as f:
                    pcfg = yaml.safe_load(f) or {}
                out_dir = (pcfg.get("paths") or {}).get("results")
            if "discovery" not in needs and out_dir:
                rel = Path(out_dir)
                target = rel if rel.is_absolute() else (_REPO / rel).resolve()
                _copy_consensus_from_primary(target)
            failures += _run([py, "scripts/04_causal_inference.py"], base_env)
        if "pooled_ace" in needs:
            failures += _run([py, "scripts/pooled_ace.py"], base_env)

        preset_path = _REPO / "config" / "presets" / f"{yaml_name}.yaml"
        out_dir = None
        if preset_path.exists():
            with open(preset_path) as f:
                pcfg = yaml.safe_load(f) or {}
            out_dir = (pcfg.get("paths") or {}).get("results")
        print(
            json.dumps(
                {"preset": pid, "causal_preset_yaml": yaml_name, "results_dir": out_dir},
                indent=2,
            )
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
