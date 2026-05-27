"""
tests/test_config.py
Pytest tests for src/causal_precip/config.py.
"""

import os
import sys
from pathlib import Path

# Make the src package importable when running pytest from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_precip.config import _deep_merge, load_config


def test_load_config_returns_dict():
    """load_config() returns a dict containing the four required top-level keys."""
    cfg = load_config()
    assert isinstance(cfg, dict)
    for key in ("domain", "variables", "date_range", "paths"):
        assert key in cfg, f"Missing key: {key}"


def test_local_defaults_inherit_base_date_range():
    """With ENV unset or 'local', empty local.yaml should keep base ERA5 window."""
    env_backup = os.environ.pop("ENV", None)
    try:
        cfg = load_config()
        assert cfg["_env"] == "local"
        assert cfg["date_range"]["start"] == "1979-01"
        assert cfg["date_range"]["end"] == "2023-12"
    finally:
        if env_backup is not None:
            os.environ["ENV"] = env_backup


def test_deep_merge_does_not_clobber_sibling_keys():
    """Merging nested dicts preserves sibling keys not present in override."""
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"x": 99}}
    result = _deep_merge(base, override)
    assert result == {"a": {"x": 99, "y": 2}}, (
        f"Deep merge clobbered sibling key. Got: {result}"
    )
