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


def test_local_override_reduces_date_range():
    """With ENV unset or 'local', date_range.start should be >= 2010 (local override)."""
    env_backup = os.environ.pop("ENV", None)
    try:
        cfg = load_config()
        start_year = int(cfg["date_range"]["start"].split("-")[0])
        assert start_year >= 2010, (
            f"Expected local config to restrict start year to >= 2010, got {start_year}"
        )
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
