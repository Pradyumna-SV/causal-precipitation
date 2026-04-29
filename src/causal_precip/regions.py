"""
src/causal_precip/regions.py
Sub-region bounding-box selection and area-weighted spatial averaging.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _lat_dim(da: xr.DataArray) -> str:
    for name in ("latitude", "lat"):
        if name in da.dims:
            return name
    raise ValueError(f"No latitude dimension found in {list(da.dims)}")


def _lon_dim(da: xr.DataArray) -> str:
    for name in ("longitude", "lon"):
        if name in da.dims:
            return name
    raise ValueError(f"No longitude dimension found in {list(da.dims)}")


def select_bbox(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    """
    Clip da to [lat_min, lat_max] × [lon_min, lon_max].
    Works with both ascending and descending latitude coordinates.
    """
    ld = _lat_dim(da)
    od = _lon_dim(da)

    lat_vals = da[ld].values
    lat_slice = (
        slice(lat_max, lat_min) if lat_vals[0] > lat_vals[-1]
        else slice(lat_min, lat_max)
    )
    return da.sel({ld: lat_slice, od: slice(lon_min, lon_max)})


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Cosine-latitude area-weighted spatial mean.
    Collapses all spatial dimensions (latitude/longitude or lat/lon).
    """
    ld = _lat_dim(da)
    weights = np.cos(np.deg2rad(da[ld])).clip(min=0)
    spatial_dims = [d for d in da.dims if d in ("latitude", "longitude", "lat", "lon")]
    return da.weighted(weights).mean(dim=spatial_dims)


def region_mean(
    da: xr.DataArray,
    region_name: str,
    cfg: dict,
) -> xr.DataArray:
    """
    Convenience wrapper: select a named region from config and return the
    area-weighted mean time series.
    """
    r  = cfg["regions"][region_name]
    ts = area_weighted_mean(
        select_bbox(da, r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"])
    )
    ts.name = f"{da.name or 'var'}_{region_name}"
    return ts


def build_region_panel(
    var_das: dict[str, xr.DataArray],
    region_name: str,
    cfg: dict,
) -> xr.Dataset:
    """
    Build a Dataset of region-averaged time series for all variables.

    Parameters
    ----------
    var_das     : mapping from short variable name to anomaly DataArray
    region_name : key in cfg['regions']
    cfg         : merged config dict

    Returns
    -------
    xr.Dataset with one variable per key in var_das, all indexed by time.
    """
    series = {}
    for vname, da in var_das.items():
        ts = region_mean(da, region_name, cfg)
        ts.name = vname
        series[vname] = ts
    ds = xr.Dataset(series)
    ds.attrs["region"] = region_name
    return ds
