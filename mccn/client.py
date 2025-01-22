from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import pystac
import pystac_client
import xarray as xr

from mccn.extent import GeoBoxBuilder
from mccn.filter import CollectionFilter
from mccn.loader.point import stac_load_point
from mccn.loader.raster import stac_load_raster
from mccn.loader.utils import ASSET_KEY
from mccn.loader.vector import stac_load_vector

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox

    from mccn._types import GroupbyOption, InterpMethods, MergeMethods


class MCCN:
    def __init__(
        self,
        endpoint: str,
        collection_id: str,
        geobox: GeoBox | None = None,
        shape: int | tuple[int, int] | None = None,
        asset_key: str | Mapping[str, str] = ASSET_KEY,
        # Shared load config
        x_col: str = "x",
        y_col: str = "y",
        t_col: str = "time",
        # Raster load config
        bands: Sequence[str] | None = None,
        # Vector load config
        groupby: GroupbyOption = "id",
        vector_fields: Sequence[str] | dict[str, Sequence[str]] | None = None,
        alias_renaming: dict[str, tuple[str, str]] | None = None,
        # Point load config
        point_fields: Sequence[str] | None = None,
        merge_method: MergeMethods = "mean",
        interp_method: InterpMethods | None = "nearest",
    ) -> None:
        self.endpoint = endpoint
        self.collection_id = collection_id
        self.collection = self.get_collection(endpoint, collection_id)
        self.shape = shape
        self.geobox = self.get_geobox(self.collection, geobox, self.shape)
        self.asset_key = asset_key
        self.collection_filter = CollectionFilter(
            self.collection, self.geobox, self.asset_key
        )
        # Shared load config
        self.x_col = x_col
        self.y_col = y_col
        self.t_col = t_col
        # Raster load config
        self.bands = bands
        # Vector load config
        self.groupby = groupby
        self.vector_fields = vector_fields
        self.alias_renaming = alias_renaming
        # Point load config
        self.point_fields = point_fields
        self.merge_method = merge_method
        self.interp_method = interp_method

    def get_collection(
        self,
        endpoint: str,
        collection_id: str,
    ) -> pystac.Collection:
        if endpoint.startswith("http"):
            res = pystac_client.Client.open(endpoint)
        else:
            res = pystac_client.Client.from_file(endpoint)
        return res.get_collection(collection_id)

    def get_geobox(
        self,
        collection: pystac.Collection,
        geobox: GeoBox | None = None,
        shape: int | tuple[int, int] | None = None,
    ) -> GeoBox:
        if geobox is not None:
            return geobox
        if shape is None:
            raise ValueError(
                "If geobox is not defined, shape must be provided to calculate geobox from collection"
            )
        return GeoBoxBuilder.from_collection(collection, shape)

    def load_raster(self) -> xr.Dataset:
        return stac_load_raster(
            self.collection_filter.raster,
            self.geobox,
            self.bands,
            self.x_col,
            self.y_col,
            self.t_col,
        )

    def load_vector(self) -> xr.Dataset:
        return stac_load_vector(
            self.collection_filter.vector,
            self.geobox,
            self.groupby,
            self.vector_fields,
            self.x_col,
            self.y_col,
            self.asset_key,
            self.alias_renaming,
        )

    def load_point(self) -> xr.Dataset:
        return stac_load_point(
            self.collection_filter.point,
            self.geobox,
            self.asset_key,
            self.point_fields,
            self.x_col,
            self.y_col,
            self.t_col,
            self.merge_method,
            self.interp_method,
        )

    def load(self) -> xr.Dataset:
        items = []
        if self.collection_filter.raster:
            items.append(
                stac_load_raster(
                    self.collection_filter.raster,
                    self.geobox,
                    self.bands,
                    self.x_col,
                    self.y_col,
                    self.t_col,
                )
            )
        if self.collection_filter.vector:
            items.append(
                stac_load_vector(
                    self.collection_filter.vector,
                    self.geobox,
                    self.groupby,
                    self.vector_fields,
                    self.x_col,
                    self.y_col,
                    self.asset_key,
                    self.alias_renaming,
                )
            )
        if self.collection_filter.point:
            items.append(
                stac_load_point(
                    self.collection_filter.point,
                    self.geobox,
                    self.asset_key,
                    self.point_fields,
                    self.x_col,
                    self.y_col,
                    self.t_col,
                    self.merge_method,
                    self.interp_method,
                )
            )
        return xr.merge(items)
