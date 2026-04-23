"""
scripts/05_figures.py
Generate all paper figures from results.

Owner: Tiffany
Run:   python scripts/05_figures.py           (local)
       ENV=nautilus python scripts/05_figures.py  (Nautilus, via k8s job)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_precip import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(cfg: dict) -> None:
    """Entry point. cfg is the fully merged config dict."""
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    cfg = load_config()
    log.info("Environment: %s", cfg.get("_env", "local"))
    log.info("Date range: %s → %s", cfg["date_range"]["start"], cfg["date_range"]["end"])
    main(cfg)
