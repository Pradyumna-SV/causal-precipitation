"""
scripts/01_download_era5.py
Download ERA5 monthly-mean variables from the Copernicus CDS API.

Three separate downloads:
  1. Single-level variables (tp, sst, swvl1, t2m) — Western US domain
  2. Pressure-level variables (z, u, v at 500/850 hPa) — Western US domain
  3. SST for the Niño 3.4 region (needed for ENSO index)

Idempotent: skips files that already exist unless --force is passed.

Run:   python scripts/01_download_era5.py
       python scripts/01_download_era5.py --force   # re-download even if files exist
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_CDsapirc = _REPO_ROOT / ".cdsapirc"

sys.path.insert(0, str(_REPO_ROOT / "src"))

from causal_precip import load_config, raw_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _ensure_cdsapi_rc() -> Path:
    """
    Point cdsapi at credentials: respect existing CDSAPI_RC / env URL+key;
    otherwise use repo-root .cdsapirc if present, else ~/.cdsapirc (default).
    """
    if os.environ.get("CDSAPI_RC"):
        return Path(os.environ["CDSAPI_RC"]).expanduser()
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        return Path(os.environ.get("CDSAPI_RC", os.path.expanduser("~/.cdsapirc")))
    if _PROJECT_CDsapirc.is_file():
        os.environ["CDSAPI_RC"] = str(_PROJECT_CDsapirc.resolve())
        log.info("Using CDS credentials from %s", _PROJECT_CDsapirc)
        return _PROJECT_CDsapirc
    return Path(os.path.expanduser("~/.cdsapirc"))


def _year_month_lists(cfg: dict) -> tuple[list[str], list[str]]:
    """Return sorted lists of year strings and zero-padded month strings."""
    start = cfg["date_range"]["start"]   # e.g. "1979-01"
    end   = cfg["date_range"]["end"]     # e.g. "2023-12"
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]),   int(end[5:7])
    years  = [str(y) for y in range(sy, ey + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    return years, months


def _cds_area(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list:
    """CDS area convention: [North, West, South, East]."""
    return [lat_max, lon_min, lat_min, lon_max]


def download_single_level(cfg: dict, client, force: bool) -> Path:
    """
    Download ERA5 monthly-mean single-level variables for the Western US.
    Variables: total_precipitation, sea_surface_temperature,
               volumetric_soil_water_layer_1, 2m_temperature.
    """
    stem    = cfg["date_range"]["start"].replace("-", "") + "_" + cfg["date_range"]["end"].replace("-", "")
    outfile = raw_path(f"era5_single_{stem}.nc", cfg)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists() and not force:
        log.info("Single-level file already exists, skipping: %s", outfile)
        return outfile

    d = cfg["domain"]
    years, months = _year_month_lists(cfg)

    log.info("Downloading single-level ERA5 (%s – %s) ...",
             cfg["date_range"]["start"], cfg["date_range"]["end"])

    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": cfg["variables"]["single_level"],
            "year":  years,
            "month": months,
            "time":  "00:00",
            "area":  _cds_area(d["lat_min"], d["lat_max"], d["lon_min"], d["lon_max"]),
            "format": "netcdf",
        },
        str(outfile),
    )
    log.info("Saved → %s (%.1f MB)", outfile, outfile.stat().st_size / 1e6)
    return outfile


def download_pressure_level(cfg: dict, client, force: bool) -> Path:
    """
    Download ERA5 monthly-mean pressure-level variables for the Western US.
    Variables: geopotential (500 hPa), u/v wind components (850 hPa).
    Both levels included in one request to minimise API round-trips.
    """
    stem    = cfg["date_range"]["start"].replace("-", "") + "_" + cfg["date_range"]["end"].replace("-", "")
    outfile = raw_path(f"era5_plev_{stem}.nc", cfg)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists() and not force:
        log.info("Pressure-level file already exists, skipping: %s", outfile)
        return outfile

    d = cfg["domain"]
    years, months = _year_month_lists(cfg)

    log.info("Downloading pressure-level ERA5 (%s – %s) ...",
             cfg["date_range"]["start"], cfg["date_range"]["end"])

    client.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type":   "monthly_averaged_reanalysis",
            "variable":       cfg["variables"]["pressure_level"],
            "pressure_level": [str(p) for p in cfg["variables"]["pressure_levels"]],
            "year":  years,
            "month": months,
            "time":  "00:00",
            "area":  _cds_area(d["lat_min"], d["lat_max"], d["lon_min"], d["lon_max"]),
            "format": "netcdf",
        },
        str(outfile),
    )
    log.info("Saved → %s (%.1f MB)", outfile, outfile.stat().st_size / 1e6)
    return outfile


def download_nino34_sst(cfg: dict, client, force: bool) -> Path:
    """
    Download SST for the Niño 3.4 box (5°S–5°N, 170°W–120°W).
    This region lies outside the Western US domain, so it needs a separate call.
    """
    stem    = cfg["date_range"]["start"].replace("-", "") + "_" + cfg["date_range"]["end"].replace("-", "")
    outfile = raw_path(f"era5_nino34_{stem}.nc", cfg)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if outfile.exists() and not force:
        log.info("Niño 3.4 file already exists, skipping: %s", outfile)
        return outfile

    nr = cfg["nino34_region"]
    years, months = _year_month_lists(cfg)

    log.info("Downloading Niño 3.4 SST (%s – %s) ...",
             cfg["date_range"]["start"], cfg["date_range"]["end"])

    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable":     ["sea_surface_temperature"],
            "year":  years,
            "month": months,
            "time":  "00:00",
            "area":  _cds_area(nr["lat_min"], nr["lat_max"], nr["lon_min"], nr["lon_max"]),
            "format": "netcdf",
        },
        str(outfile),
    )
    log.info("Saved → %s (%.1f MB)", outfile, outfile.stat().st_size / 1e6)
    return outfile


def main(cfg: dict, force: bool = False) -> None:
    try:
        import cdsapi
    except ImportError:
        log.error("cdsapi not installed. Run: pip install cdsapi")
        sys.exit(1)

    creds_path = _ensure_cdsapi_rc()
    try:
        client = cdsapi.Client()
    except Exception as exc:
        log.error(
            "CDS API client initialisation failed: %s\n"
            "Put a CDS config at %s (repo root) with:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <uid>:<api-key>\n"
            "Or set CDSAPI_RC to that file, or CDSAPI_URL + CDSAPI_KEY.\n"
            "Tried config path: %s",
            exc,
            _PROJECT_CDsapirc,
            creds_path,
        )
        sys.exit(1)

    single_file = download_single_level(cfg, client, force)
    plev_file   = download_pressure_level(cfg, client, force)
    nino34_file = download_nino34_sst(cfg, client, force)

    log.info("All downloads complete.")
    log.info("  Single-level : %s", single_file)
    log.info("  Pressure-level: %s", plev_file)
    log.info("  Niño 3.4 SST : %s", nino34_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 data from CDS API.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output files already exist.")
    args = parser.parse_args()

    cfg = load_config()
    log.info("Environment : %s", cfg.get("_env", "local"))
    log.info("Date range  : %s → %s", cfg["date_range"]["start"], cfg["date_range"]["end"])
    main(cfg, force=args.force)
