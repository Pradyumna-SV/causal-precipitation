"""
src/causal_precip/config.py
Loads and merges YAML configuration files for the causal-precipitation project.
"""

import os
from pathlib import Path

import yaml

# Repo root is two levels above this file: src/causal_precip/config.py
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on scalar conflicts."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config() -> dict:
    """
    Load base config and merge the environment-specific override.

    Environment is determined by the ENV environment variable (default: "local").

    Optional: ``CAUSAL_PRESET`` merges ``config/presets/<name>.yaml`` after ENV
    (used for exploratory grid runs without overwriting base ``results/``).

    Returns the fully merged config dict with ``_env`` and ``_preset`` keys.
    """
    env = os.environ.get("ENV", "local")

    base_path = _REPO_ROOT / "config" / "base.yaml"
    override_path = _REPO_ROOT / "config" / f"{env}.yaml"

    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)

    if override_path.exists():
        with open(override_path, "r") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)

    preset = os.environ.get("CAUSAL_PRESET", "").strip()
    if preset:
        preset_path = _REPO_ROOT / "config" / "presets" / f"{preset}.yaml"
        if preset_path.exists():
            with open(preset_path, "r") as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
        cfg["_preset"] = preset
    else:
        cfg["_preset"] = None

    cfg["_env"] = env
    return cfg
