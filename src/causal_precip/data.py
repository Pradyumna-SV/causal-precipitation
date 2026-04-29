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

# CDS long-name → possible NetCDF variable names (new CDS API may use different names)
# Listed in order of preference; first match wins.
CDS_CANDIDATES: dict[str, list[str]] = {
    "total_precipitation":          ["tp", "total_precipitation"],
    "sea_surface_temperature":      ["sst", "sea_surface_temperature"],
    "volumetric_soil_water_layer_1":["swvl1", "volumetric_soil_water_layer_1", "swvl"],
    "2m_temperature":               ["t2m", "2m_temperature"],
    "geopotential":                 ["z", "geopotential"],
    "u_component_of_wind":          ["u", "u_component_of_wind"],
    "v_component_of_wind":          ["v", "v_component_of_wind"],
}
# Keep CDS_SHORT as the primary (first) short name for backwards compatibility
CDS_SHORT: dict[str, str] = {k: v[0] for k, v in CDS_CANDIDATES.items()}


def _find_var(ds: xr.Dataset, long_name: str) -> Optional[str]:
    """Return the first matching variable name in ds for the given long_name."""
    for candidate in CDS_CANDIDATES.get(long_name, [long_name]):
        if candidate in ds:
            return candidate
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_path(cfg_path: str, filename: str) -> Path:
    """Join cfg_path and filename; use cfg_path as-is if it is absolute."""
    p = Path(cfg_path)
    base = p if p.is_absolute() else _repo_root() / p
    return (base / filename).resolve()


def raw_path(filename: str, cfg: dict) -> Path:
    return _resolve_path(cfg["paths"]["raw_data"], filename)


def processed_path(filename: str, cfg: dict) -> Path:
    return _resolve_path(cfg["paths"]["processed"], filename)


def results_path(filename: str, cfg: dict) -> Path:
    return _resolve_path(cfg["paths"]["results"], filename)


def figures_path(filename: str, cfg: dict) -> Path:
    return _resolve_path(cfg["paths"]["figures"], filename)


def _raw_stem(cfg: dict) -> str:
    """Build the date-range suffix used in raw file names."""
    start = cfg["date_range"]["start"].replace("-", "")
    end   = cfg["date_range"]["end"].replace("-", "")
    return f"{start}_{end}"


def _unzip_if_needed(path: Path) -> Path:
    """
    If `path` is actually a zip archive (CDS API sometimes wraps NetCDF in zip),
    extract the first .nc file inside it next to the original and return that path.
    The zip file is left in place so reruns are idempotent.
    """
    import zipfile
    if not zipfile.is_zipfile(path):
        return path
    with zipfile.ZipFile(path) as zf:
        nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_names:
            raise ValueError(f"Zip archive {path} contains no .nc files: {zf.namelist()}")
        target = path.parent / nc_names[0]
        if not target.exists():
            zf.extract(nc_names[0], path.parent)
    return target


def _normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalise ERA5 coordinate conventions from the new CDS API (2024+):
      - rename valid_time → time  (CDS API v2 breaking change)
      - latitude  ascending
      - longitude ascending
    """
    # New CDS API uses 'valid_time' instead of 'time'
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

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
    if not path.exists():
        raise FileNotFoundError(f"Single-level ERA5 file not found: {path}")
    path = _unzip_if_needed(path)
    ds = xr.open_dataset(path, engine="netcdf4")
    return _normalise_coords(ds)


def open_raw_plev(cfg: dict) -> xr.Dataset:
    """Open the pressure-level ERA5 file (z, u, v at 500 / 850 hPa)."""
    path = raw_path(f"era5_plev_{_raw_stem(cfg)}.nc", cfg)
    if not path.exists():
        raise FileNotFoundError(f"Pressure-level ERA5 file not found: {path}")
    path = _unzip_if_needed(path)
    ds = xr.open_dataset(path, engine="netcdf4")
    return _normalise_coords(ds)


def open_raw_nino34(cfg: dict) -> xr.Dataset:
    """Open the Niño 3.4 region SST file (5°S–5°N, 170°W–120°W)."""
    path = raw_path(f"era5_nino34_{_raw_stem(cfg)}.nc", cfg)
    if not path.exists():
        raise FileNotFoundError(f"Niño 3.4 ERA5 file not found: {path}")
    path = _unzip_if_needed(path)
    ds = xr.open_dataset(path, engine="netcdf4")
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
