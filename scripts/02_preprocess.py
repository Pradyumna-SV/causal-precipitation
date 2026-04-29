"""
scripts/02_preprocess.py
Compute anomalies, build region-averaged panels, run stationarity tests,
construct the extreme-precipitation binary indicator.

Outputs (all in cfg['paths']['processed']):
  panel_{region}.nc       — time × variable Dataset (anomalies + nino34 + tp_extreme)
  stationarity_tests.json — ADF p-values for every time series

Run:   python scripts/02_preprocess.py           (local)
       ENV=nautilus python scripts/02_preprocess.py  (Nautilus k8s)
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from causal_precip import (
    WMO_BASE,
    area_weighted_mean,
    build_region_panel,
    compute_anomalies,
    extreme_precip_flag,
    load_config,
    nino34_index,
    open_raw_plev,
    open_raw_single,
    processed_path,
    select_bbox,
)
from causal_precip.data import CDS_SHORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Variables extracted from single-level file
SINGLE_VARS: dict[str, str] = {
    "tp":    "total_precipitation",
    "sst":   "sea_surface_temperature",
    "swvl1": "volumetric_soil_water_layer_1",
    "t2m":   "2m_temperature",
}

# Pressure-level variable → (level_hPa,)
PLEV_VARS: dict[str, tuple[int, str]] = {
    "z500": (500, "geopotential"),
    "u850": (850, "u_component_of_wind"),
    "v850": (850, "v_component_of_wind"),
}


def _adf_pvalue(ts: np.ndarray) -> float:
    """Augmented Dickey-Fuller test p-value (H0: unit root / non-stationary)."""
    from statsmodels.tsa.stattools import adfuller

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adfuller(ts[~np.isnan(ts)], autolag="AIC")
    return float(result[1])


def load_anomaly_fields(cfg: dict) -> dict[str, xr.DataArray]:
    """
    Load all raw ERA5 files, clip to Western US domain, compute anomalies.
    Returns a dict mapping short var name → anomaly DataArray.
    """
    d = cfg["domain"]
    anomalies: dict[str, xr.DataArray] = {}

    # --- Single-level variables ---
    log.info("Loading single-level file …")
    ds_sl = open_raw_single(cfg)
    for short, long_name in SINGLE_VARS.items():
        cds_name = CDS_SHORT[long_name]
        if cds_name not in ds_sl:
            log.warning("Variable %s (%s) not found in single-level file, skipping.", cds_name, long_name)
            continue
        da = ds_sl[cds_name]
        da = select_bbox(da, d["lat_min"], d["lat_max"], d["lon_min"], d["lon_max"])
        da.name = short
        anom = compute_anomalies(da, base_period=WMO_BASE)
        anom.name = short
        anomalies[short] = anom
        log.info("  %s: anomaly computed (base period %s – %s)", short, *WMO_BASE)

    # --- Pressure-level variables ---
    log.info("Loading pressure-level file …")
    ds_pl = open_raw_plev(cfg)

    plev_coord = None
    for cname in ("pressure_level", "level", "plev"):
        if cname in ds_pl.dims or cname in ds_pl.coords:
            plev_coord = cname
            break

    for short, (level, long_name) in PLEV_VARS.items():
        cds_name = CDS_SHORT[long_name]
        if cds_name not in ds_pl:
            log.warning("Variable %s not found in pressure-level file, skipping.", cds_name)
            continue
        da = ds_pl[cds_name]
        if plev_coord is not None:
            da = da.sel({plev_coord: level}, method="nearest")
        da = select_bbox(da, d["lat_min"], d["lat_max"], d["lon_min"], d["lon_max"])
        da.name = short
        anom = compute_anomalies(da, base_period=WMO_BASE)
        anom.name = short
        anomalies[short] = anom
        log.info("  %s @ %d hPa: anomaly computed", short, level)

    return anomalies


def build_panels(
    anomalies: dict[str, xr.DataArray],
    nino34: xr.DataArray,
    cfg: dict,
) -> dict[str, xr.Dataset]:
    """
    For each sub-region, spatially average all anomaly fields, append Niño 3.4,
    and append the binary extreme-precipitation indicator.
    Returns {region_name: xr.Dataset}.
    """
    panels: dict[str, xr.Dataset] = {}

    for region in cfg["regions"]:
        log.info("Building panel for region: %s", region)
        ds = build_region_panel(anomalies, region, cfg)

        # Append Niño 3.4 index (already a time series)
        ds["nino34"] = nino34.sel(time=ds.time)

        # Binary extreme-precipitation flag
        if "tp" in ds:
            ds["tp_extreme"] = extreme_precip_flag(
                ds["tp"], cfg.get("extreme_precip_percentile", 90)
            )

        ds.attrs["region"]     = region
        ds.attrs["base_period"] = f"{WMO_BASE[0]} – {WMO_BASE[1]}"
        panels[region] = ds

    return panels


def run_stationarity_tests(panels: dict[str, xr.Dataset]) -> dict:
    """Run ADF tests on every time series in every panel. Returns a nested dict."""
    results: dict = {}
    for region, ds in panels.items():
        results[region] = {}
        for vname in ds.data_vars:
            if vname == "tp_extreme":
                continue
            ts = ds[vname].values.astype(float)
            pval = _adf_pvalue(ts)
            stationary = pval < 0.05
            results[region][vname] = {"adf_pvalue": pval, "stationary": stationary}
            level = logging.INFO if stationary else logging.WARNING
            log.log(level, "  ADF [%s / %s]: p=%.4f — %s",
                    region, vname, pval, "stationary" if stationary else "NON-STATIONARY")
    return results


def save_panels(panels: dict[str, xr.Dataset], cfg: dict) -> None:
    out_dir = processed_path("", cfg).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for region, ds in panels.items():
        outfile = processed_path(f"panel_{region}.nc", cfg)
        # Drop integer flag before saving (xarray encoding quirk)
        ds_save = ds.copy()
        if "tp_extreme" in ds_save:
            ds_save["tp_extreme"] = ds_save["tp_extreme"].astype(np.int16)
        ds_save.to_netcdf(outfile)
        log.info("Saved panel → %s", outfile)


def main(cfg: dict) -> None:
    # 1. Load and compute anomaly fields
    anomalies = load_anomaly_fields(cfg)

    # 2. Niño 3.4 index
    log.info("Computing Niño 3.4 index …")
    nino34 = nino34_index(cfg, base_period=WMO_BASE)

    # 3. Build region panels
    panels = build_panels(anomalies, nino34, cfg)

    # 4. Stationarity tests
    log.info("Running ADF stationarity tests …")
    stat_results = run_stationarity_tests(panels)

    stat_file = processed_path("stationarity_tests.json", cfg)
    stat_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stat_file, "w") as f:
        json.dump(stat_results, f, indent=2)
    log.info("Stationarity results → %s", stat_file)

    # 5. Save panels
    save_panels(panels, cfg)

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    cfg = load_config()
    log.info("Environment : %s", cfg.get("_env", "local"))
    log.info("Date range  : %s → %s", cfg["date_range"]["start"], cfg["date_range"]["end"])
    main(cfg)
