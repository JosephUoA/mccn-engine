from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from io import BytesIO
from typing import Iterator, List, Optional, Dict

from odc.geo.geobox import GeoBox
from odc.stac import stac_load
import pystac_client
from tqdm import tqdm
import rioxarray
import xarray

from mccn.loader import stac_load_vector  # , ShapefileDriver
from mccn.wcs_importer import WcsImporterFactory


class Mccn:
    def __init__(self, stac_url: str) -> None:
        # This class needs to be responsible for gathering both public WCS and node generated STAC
        # described data into an x-array data structure.
        print(f"Connection to the STAC endpoint at {stac_url}.")
        self.stac_client = pystac_client.Client.open(stac_url)
        self.lat_count = None
        self.lon_count = None
        self.bbox = None

    def _query(self, col_id: str) -> Iterator:
        # TODO: Add options for querying.
        query = self.stac_client.search(collections=[col_id])
        return query.items()

    def _load_stac(self, col_id: str, bands: Optional[List[str]] = None, groupby: str = "id",
                  crs: Optional[str] = None, geobox: Optional[GeoBox] = None,
                  lazy: bool = False) -> xarray.Dataset:
        # TODO: Expose other parameters to the stac_load function
        print(f"Loading data for {col_id}.")
        pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        # To lazy load we need to pass something to the chunks parameter of stac_load() function.
        if lazy:
            chunks = {'x': 2048, 'y': 2048}
        else:
            chunks = None
        # TODO: We likely need a unique load function here depending on data type. For example
        # TODO: we need to write at least one for raster data and time series data.

        xx = stac_load(
            self._query(col_id), bands=bands, groupby=groupby, chunks=chunks,  # type: ignore
            progress=tqdm, pool=pool, crs=crs, geobox=geobox
        )
        if 'x' in xx.xindexes and 'y' in xx.xindexes:
            min_lon = xx.x.min().item()
            max_lon = xx.x.max().item()
            min_lat = xx.y.min().item()
            max_lat = xx.y.max().item()
        elif 'longitude' in xx.xindexes and 'latitude' in xx.xindexes:
            min_lon = xx.longitude.min().item()
            max_lon = xx.longitude.max().item()
            min_lat = xx.latitude.min().item()
            max_lat = xx.latitude.max().item()
        else:
            raise AttributeError("Spatial axes must contain either x and y or "
                                 "longitude and latitude.")
        self.bbox = [min_lon, min_lat, max_lon, max_lat]
        return xx

    @staticmethod
    def load_public(source: str, bbox: List[float], layername=None):
        # Demo basic load function for WCS endpoint. Only DEM is currently supported.
        response = WcsImporterFactory().get_wcs_importer(source).get_data(bbox, layername)
        return rioxarray.open_rasterio(BytesIO(response.read()))

    @staticmethod
    def plot(xx: xarray.Dataset):
        # TODO: Only plots the time=0 index of the array. Options are to handle multiple time
        # TODO: indices in this function or stipulate one index as parameter.
        reduce_dim = "band"
        xx_0 = xx.isel(time=0).to_array(reduce_dim)
        return xx_0.plot.imshow(
            col=reduce_dim,
            size=5,
            vmin=int(xx_0.min()),
            vmax=int(xx_0.max()),
        )

    def load(self, col_id: str, bands: Optional[List[str]] = None, groupby: str = "id",
             crs: Optional[str] = None, geobox: Optional[GeoBox] = None, lazy: bool = False,
             source: Optional[Dict[str, str]] = None, mask=None):
        """
        Load the STAC items for a given collection ID into an xarray dataset. Several options are
        available for sub-selection and transformation of the data upon loading.
        :param col_id: STAC collection id for study.
        :param bands: List of band names to load, defaults to All. Also accepts single band name as
        input
        :param groupby:
        :param crs:
        :param geobox:
        :param lazy:
        :param source:
        :param mask:
        :return:
        """
        xx = self._load_stac(col_id, bands=bands, groupby=groupby, crs=crs, geobox=geobox,
                            lazy=lazy)
        if mask:
            vector_array = stac_load_vector(
                list(self._query(col_id)), gbox=geobox
            )
            # TODO: Combine rasterised vector layers with other raster layers.
        # TODO: The following works only for a single layer from the DEM endpoint. This code for
        # TODO: combining data needs to be generalised for all use cases.
        if source is not None:
            for source_name, layer_name in source.items():
                if source_name != "dem":
                    raise NotImplementedError(f"Datacube stacking for {source_name} has not been"
                                              f"implemented.")
                yy = self.load_public(source=source_name, bbox=self.bbox, layername=layer_name)
                yy = yy.rename({"x": "longitude", "y": "latitude"})
                yy = yy.interp(
                    longitude=list(xx.longitude.values),
                    latitude=list(xx.latitude.values),
                    method="linear",
                    kwargs={"fill_value": "extrapolate"}
                )
                # While the datasets have the same EPSG, they are defined differently in the
                # spatial_ref layer. This aligns them but have to investigate further.
                yy["spatial_ref"] = xx.spatial_ref
                # DEM is a static over time dataset.
                yy = yy.expand_dims(dim={"time": xx.time})
                yy = yy.squeeze(dim="band", drop=True)
                # This is where the layer in the datacube is named
                yy = yy.to_dataset(name="elevation")
                xx = xarray.combine_by_coords([xx, yy])  # type: ignore

        return xx
