"""
src/causal_precip/data.py
ERA5 loading, climatology subtraction, and anomaly computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr

# WMO standard 30-year base period (avoids circularity from using the full record)
WMO_BASE: Tuple[str, str] = ("1979-01", "2010-12")

# CDS long-name → NetCDF short variable name written by the CDS API
CDS_SHORT: dict[str, str] = {
    "total_precipitation": "tp",
    "sea_surface_temperature": "sst",
    "volumetric_soil_water_layer_1": "swvl1",
    "2m_temperature": "t2m",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def raw_path(filename: str, cfg: dict) -> Path:
    return (_repo_root() / cfg["paths"]["raw_data"] / filename).resolve()


def processed_path(filename: str, cfg: dict) -> Path:
    return (_repo_root() / cfg["paths"]["processed"] / filename).resolve()


def results_path(filename: str, cfg: dict) -> Path:
    return (_repo_root() / cfg["paths"]["results"] / filename).resolve()


def figures_path(filename: str, cfg: dict) -> Path:
    return (_repo_root() / cfg["paths"]["figures"] / filename).resolve()


def _raw_stem(cfg: dict) -> str:
    """Build the date-range suffix used in raw file names."""
    start = cfg["date_range"]["start"].replace("-", "")
    end   = cfg["date_range"]["end"].replace("-", "")
    return f"{start}_{end}"


def _normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalise ERA5 coordinate order:
      - latitude  ascending  (some CDS downloads are descending)
      - longitude ascending  (should already be, but enforce)
    """
    if "latitude" in ds.dims and ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))
    if "longitude" in ds.dims and len(ds.longitude) > 1:
        ds = ds.sortby("longitude")
    return ds


# ---------------------------------------------------------------------------
# Raw file openers
# ---------------------------------------------------------------------------

def open_raw_single(cfg: dict) -> xr.Dataset:
    """Open the single-level ERA5 file (tp, sst, swvl1, t2m)."""
    path = raw_path(f"era5_single_{_raw_stem(cfg)}.nc", cfg)
    ds = xr.open_dataset(path, chunks={"time": 60})
    return _normalise_coords(ds)


def open_raw_plev(cfg: dict) -> xr.Dataset:
    """Open the pressure-level ERA5 file (z, u, v at 500 / 850 hPa)."""
    path = raw_path(f"era5_plev_{_raw_stem(cfg)}.nc", cfg)
    ds = xr.open_dataset(path, chunks={"time": 60})
    return _normalise_coords(ds)


def open_raw_nino34(cfg: dict) -> xr.Dataset:
    """Open the Niño 3.4 region SST file (5°S–5°N, 170°W–120°W)."""
    path = raw_path(f"era5_nino34_{_raw_stem(cfg)}.nc", cfg)
    ds = xr.open_dataset(path, chunks={"time": 60})
    return _normalise_coords(ds)


# ---------------------------------------------------------------------------
# Climatology and anomaly computation
# ---------------------------------------------------------------------------

def compute_climatology(
    da: xr.DataArray,
    base_period: Tuple[str, str] = WMO_BASE,
) -> xr.DataArray:
    """
    Monthly climatology over the WMO base period.
    Returns a DataArray indexed by month (1–12).
    """
    da_base = da.sel(time=slice(*base_period))
    if da_base.time.size == 0:
        # Fall back to the full record if the base period is outside the loaded range
        da_base = da
    clim = da_base.groupby("time.month").mean("time")
    clim.attrs = {**da.attrs, "long_name": da.attrs.get("long_name", "") + " climatology"}
    return clim


def compute_anomalies(
    da: xr.DataArray,
    clim: Optional[xr.DataArray] = None,
    base_period: Tuple[str, str] = WMO_BASE,
) -> xr.DataArray:
    """
    Subtract monthly climatology from da.
    Computes climatology internally if not supplied.
    """
    if clim is None:
        clim = compute_climatology(da, base_period)
    anom = da.groupby("time.month") - clim
    anom.attrs = {**da.attrs, "long_name": da.attrs.get("long_name", "") + " anomaly"}
    return anom


# ---------------------------------------------------------------------------
# Niño 3.4 index
# ---------------------------------------------------------------------------

def nino34_index(
    cfg: dict,
    base_period: Tuple[str, str] = WMO_BASE,
) -> xr.DataArray:
    """
    Niño 3.4 SST anomaly index: area-weighted mean SST anomaly
    over 5°S–5°N, 170°W–120°W.  Returns a 1-D time series.
    """
    from .regions import select_bbox, area_weighted_mean

    nr  = cfg["nino34_region"]
    ds  = open_raw_nino34(cfg)
    sst = ds[CDS_SHORT["sea_surface_temperature"]]

    sst_box  = select_bbox(sst, nr["lat_min"], nr["lat_max"], nr["lon_min"], nr["lon_max"])
    sst_mean = area_weighted_mean(sst_box)
    nino34   = compute_anomalies(sst_mean, base_period=base_period)

    nino34.name = "nino34"
    nino34.attrs["long_name"] = "Niño 3.4 SST anomaly index"
    nino34.attrs["units"]     = "K"
    return nino34


# ---------------------------------------------------------------------------
# Extreme precipitation indicator
# ---------------------------------------------------------------------------

def extreme_precip_flag(
    tp_ts: xr.DataArray,
    percentile: float = 90.0,
) -> xr.DataArray:
    """
    Binary indicator: 1 if monthly tp_ts >= the `percentile`-th quantile, else 0.
    Uses the full time axis (not just the base period) for threshold computation
    so that ENSO modulation of extremes is preserved.
    """
    threshold = float(np.nanpercentile(tp_ts.values, percentile))
    flag = xr.where(tp_ts >= threshold, 1, 0).astype(np.int8)
    flag.name = "tp_extreme"
    flag.attrs = {
        "long_name": f"Extreme precipitation indicator (>= {percentile}th pct)",
        "percentile": percentile,
        "threshold_mm_day": threshold,
        "units": "1",
    }
    return flag
