from __future__ import annotations

import collections
import logging
from typing import TYPE_CHECKING

import odc.stac
import pystac
import xarray as xr
from pystac.extensions.eo import EOExtension

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odc.geo.geobox import GeoBox


logger = logging.getLogger(__name__)


def partition_items_based_on_bands(
    items: Sequence[pystac.Item],
) -> dict[tuple[str, ...] | None, list[pystac.Item]]:
    """Separate items into groups identified by the bands that items contain.

    Also raise a warning when no band information can be read from STAC metadata

    :param items: list of raster stac metadata represented as pystac.Item
    :type items: Sequence[pystac.Item]
    :return: dictionary of items with key being the common bands
    :rtype: Mapping[str|tuple[str], Sequence[pystac.Item]]
    """
    result = collections.defaultdict(list)
    for item in items:
        ext = EOExtension.ext(item)
        if not EOExtension.has_extension(item) or ext.bands is None:
            logger.warning(
                f"Raster item: {item.id} does not have STAC band information. This could lead to unpredictable error when loading into xarray. For consistency, it's best to annotate the band information for the STAC metadata using EOExtension."
            )
        bands = tuple(sorted([b.name for b in ext.bands])) if ext.bands else None
        result[bands].append(item)
    return result


def _odc_load_wrapper(
    items: Sequence[pystac.Item],
    geobox: GeoBox | None,
    bands: str | Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    ds = odc.stac.load(items, bands, geobox=geobox)
    # NOTE: odc stac load uses odc.geo.xr.xr_coords to set dimension name
    # it either uses latitude/longitude or y/x depending on the underlying crs
    # so there is no proper way to know which one it uses aside from trying
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.rename({"longitude": x_col, "latitude": y_col})
    elif "x" in ds.dims and "y" in ds.dims:
        ds = ds.rename({"x": x_col, "y": y_col})
    if "time" in ds.dims:
        ds = ds.rename({"time": t_col})
    return ds


def stac_load_raster(
    items: Sequence[pystac.Item],
    geobox: GeoBox | None,
    bands: str | Sequence[str] | None = None,
    x_col: str = "x",
    y_col: str = "y",
    t_col: str = "time",
) -> xr.Dataset:
    """Loader raster data

    Provides a thin wrapper for `odc.stac.load`. Also rename the `x`, `y` and `time` dimensions
    to input values `x_col`, `y_col`, and `t_col` for subsequent merging

    :param items: list of stac Item that has raster as the source asset
    :type items: Sequence[pystac.Item]
    :param geobox: object that defines the spatial extent, shape, and crs of the output dataset
    :type geobox: GeoBox | None
    :param bands: selected bands to read from source tiff. If None, read all bands, defaults to None
    :type bands: str | Sequence[str] | None, optional
    :param x_col: renamed x coordinate, defaults to "x"
    :type x_col: str, optional
    :param y_col: renamed y coordinate, defaults to "y"
    :type y_col: str, optional
    :param t_col: renamed time coordinate, defaults to "time"
    :type t_col: str, optional
    :return: raster data as xarray.Dataset
    :rtype: xr.Dataset
    """
    # NOTE: odc stac load does not work well when throwing in items with different band information
    # For instance, if item1 is RGB and item2 is DSM, the DSM layer will not be loaded
    # A way to fix this is to separate items based on common bands. For the previous example,
    # we can load DSM items and RGB items separately, then merge the resulting xarray ds together.
    # This fix is applied in the code below, which partition items based on bands, load the rasters using
    # odc stac load, then merge the resulting xarray datasets.
    band_map = partition_items_based_on_bands(items)
    if bands is not None:
        if isinstance(bands, str):
            processed_bands = tuple([bands])
        else:
            processed_bands = tuple(sorted(bands))
        if processed_bands not in band_map:
            raise ValueError(f"No bands found in raster items: {bands}")
        bands_to_load: dict[tuple[str, ...] | None, list[pystac.Item]] = {
            processed_bands: band_map[processed_bands]
        }
    else:
        bands_to_load = band_map
    ds = []
    for band_info, band_items in bands_to_load.items():
        ds.append(_odc_load_wrapper(band_items, geobox, band_info, x_col, y_col, t_col))
    return xr.merge(ds)
