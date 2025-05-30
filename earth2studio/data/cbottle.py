import os
import pathlib
from datetime import datetime, timedelta
from urllib.parse import urljoin

import numpy as np
import torch
import xarray as xr
from loguru import logger

try:
    import earth2grid
    from cbottle.checkpointing import Checkpoint
    from cbottle.datasets.base import TimeUnit
    from cbottle.datasets.dataset_2d import encode_sst
    from cbottle.datasets.dataset_3d import get_batch_info
    from cbottle.denoiser_factories import get_denoiser
    from cbottle.diffusion_samplers import (
        StackedRandomGenerator,
        edm_sampler_from_sigma,
    )
except ImportError:
    earth2grid = None
    Checkpoint = None
    StackedRandomGenerator = None
    edm_sampler_from_sigma = None
    get_batch_info = None
    TimeUnit = None
    get_denoiser = None

from earth2studio.data.base import DataSource
from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import CBottleLexicon
from earth2studio.models.auto import Package
from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.utils.imports import check_extra_imports
from earth2studio.utils.type import TimeArray, VariableArray

HPX_LEVEL = 6


@check_extra_imports("cbottle", ["cbottle", "earth2grid"])
class CBottle3D(torch.nn.Module, AutoModelMixin):
    """_summary_

    Parameters
    ----------
    core_model : torch.nn.Module
        _description_
    sst_ds : xr.Dataset
        _description_
    lat_lon : bool, optional
        _description_, by default True
    sigma_max : float, optional
        _description_, by default 80
    seed : int, optional
        _description_, by default 0
    cache : bool, optional
        _description_, by default False
    verbose : bool, optional
        _description_, by default True
    """

    VARIABLES = np.array(list(CBottleLexicon.VOCAB.keys()))

    def __init__(
        self,
        core_model: torch.nn.Module,
        sst_ds: xr.Dataset,
        lat_lon: bool = True,
        sigma_max: float = 80,
        seed: int = 0,
        cache: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self.core_model = core_model
        self.sst = sst_ds
        self.lat_lon = lat_lon
        self.sigma_max = sigma_max
        self.seed = seed
        self.variables = np.array(list(CBottleLexicon.VOCAB.keys()))

        self._cache = cache
        self.verbose = verbose

        # Set up SST Lat Lon to HPX regridder
        target_grid = earth2grid.healpix.Grid(
            HPX_LEVEL, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        lon_center = self.sst.lon.values
        # need to workaround bug where earth2grid fails to interpolate in circular manner
        # if lon[0] > 0
        # hack: rotate both src and target grids by the same amount so that src_lon[0] == 0
        # See https://github.com/NVlabs/earth2grid/issues/21
        src_lon = lon_center - lon_center[0]
        target_lon = (target_grid.lon - lon_center[0]) % 360
        grid = earth2grid.latlon.LatLonGrid(self.sst.lat.values, src_lon)
        self.sst_regridder = grid.get_bilinear_regridder_to(
            target_grid.lat, lon=target_lon
        )

        # Empty tensor just to make tracking current device easier
        self.register_buffer("device_buffer", torch.empty(0))
        # Set seed of random generator
        self.set_seed(seed=seed)

    @torch.inference_mode()
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in CBottle3D lexicon.

        Returns
        -------
        xr.DataArray
            Generated data from CBottle
        """
        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        input = self.get_cbottle_input(time[0])
        batch_info = get_batch_info(time_step=0, time_unit=TimeUnit.HOUR)

        varidx = np.where(variable == self.variables)[0]
        print(varidx)

        device = self.device_buffer.device
        condition = input["condition"].to(device)
        labels = input["labels"].unsqueeze(0).to(device)
        images = input["target"].unsqueeze(0).to(device)
        second_of_day = input["second_of_day"].unsqueeze(0).to(device).float()
        day_of_year = input["day_of_year"].unsqueeze(0).to(device).float()
        sigma_max = torch.Tensor([self.sigma_max]).to(device)

        # Hack right now
        self.rnd = StackedRandomGenerator(device, seeds=[self.seed] * images.shape[0])
        latents = self.rnd.randn(
            (
                images.shape[0],
                self.core_model.img_channels,
                self.core_model.time_length,
                self.core_model.domain.numel(),
            ),
            device=device,
        )

        xT = latents * sigma_max

        # labels_when_nan = None
        # if config.denoiser_type == DenoiserType.mask_filling:
        #     labels_when_nan = build_labels(labels, config.denoiser_when_nan)
        # elif config.denoiser_type == DenoiserType.infill:
        #     labels = build_labels(labels, config.denoiser_when_nan)

        # Gets appropriate denoiser based on config
        D = get_denoiser(
            net=self.core_model,
            images=images,
            labels=labels,
            condition=condition,
            second_of_day=second_of_day,
            day_of_year=day_of_year,
            denoiser_type="standard",  # 'mask_filling', 'infill', 'standard'
            sigma_max=sigma_max,
            labels_when_nan=None,
        )

        out = edm_sampler_from_sigma(
            D,
            xT,
            randn_like=torch.randn_like,
            sigma_max=int(sigma_max),  # Convert to int for type compatibility
        )

        x = batch_info.denormalize(out)
        x = x[:, varidx]
        # ring_order = self.core_model.domain._grid.reorder(earth2grid.healpix.PixelOrder.RING, x)

        if self.lat_lon:
            # Convert back into lat lon
            nlat, nlon = 721, 1440
            latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
                nlat, nlon, includes_south_pole=True
            )
            regridder = earth2grid.get_regridder(
                self.core_model.domain._grid, latlon_grid
            ).to(device)
            field_regridded = regridder(x).squeeze(2)

            return xr.DataArray(
                data=field_regridded[None, :].cpu().numpy(),
                dims=["time", "lead_time", "variable", "lat", "lon"],
                coords={
                    "time": np.array(time),
                    "lead_time": np.array([timedelta(0)]),
                    "variable": np.array(variable),
                    "lat": np.linspace(90, -90, nlat, endpoint=False),
                    "lon": np.linspace(0, 360, nlon, endpoint=False),
                },
            )

    def get_cbottle_input(
        self,
        time: datetime,
        label: int = 1,  # 0 for ICON, 1 for ERA5
    ) -> dict[str, torch.Tensor]:
        """

        Args:
            arr: (c, x) in NEST order. in standard units
            sst: (c, x) in NEST order. in deg K

        Returns:
            output dict, condition and target in HEALPIX_PAD_XY order

        """
        labels = torch.nn.functional.one_hot(torch.tensor(label), num_classes=1024)

        time_arr = np.array([time], dtype="datetime64[ns]")
        sst_data = torch.from_numpy(
            self.sst["tosbcs"].interp(time=time_arr, method="linear").values + 273.15
        ).to(self.device_buffer.device)
        sst_data = self.sst_regridder(sst_data)

        cond = encode_sst(sst_data.cpu())

        def reorder(x: torch.Tensor) -> torch.Tensor:
            x = torch.as_tensor(x)
            return earth2grid.healpix.reorder(
                x, earth2grid.healpix.PixelOrder.NEST, earth2grid.healpix.HEALPIX_PAD_XY
            )

        day_start = time.replace(hour=0, minute=0, second=0)
        year_start = day_start.replace(month=1, day=1)
        second_of_day = (time - day_start) / timedelta(seconds=1)
        day_of_year = (time - year_start) / timedelta(seconds=86400)

        # ["rlut", "rsut", "rsds"]
        nan_channels = [38, 39, 42]
        target = np.zeros(
            (self.variables.shape[0], 1, 4**HPX_LEVEL * 12), dtype=np.float32
        )
        target[nan_channels, ...] = np.nan

        out = {
            "target": torch.tensor(target),
            "labels": labels,
            "condition": reorder(cond),
            "second_of_day": torch.tensor([second_of_day]),
            "day_of_year": torch.tensor([day_of_year]),
        }
        # out["timestamp"] = datetime.timestamp(time)

        return out

    def set_seed(self, seed: int) -> None:
        """Set seed of CBottle latent variable generator

        Parameters
        ----------
        seed : int
            Seed value
        """
        # batch = images.shape[0]
        batch = 1
        self.rnd = StackedRandomGenerator(
            self.device_buffer.device, seeds=[seed] * batch
        )

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "cbottle")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_cbottle")
        return cache_location

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for CBottle3D, governed but the CMIP SST data
        used to train it

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:

            if time < datetime(year=1940, month=1, day=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1940 for CBottle3D"
                )

            if time >= datetime(year=2022, month=12, day=16, hour=12):
                raise ValueError(
                    f"Requested date time {time} needs to be before December 16th, 2022 for CBottle3D"
                )

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained CBottle3D model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidia/earth-2/cbottle@1.0",
            cache_options={
                "cache_storage": Package.default_cache("cbottle"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports("cbottle", ["cbottle", "earth2grid"])
    def load_model(cls, package: Package) -> DataSource:
        """Load AI datasource from package"""

        with Checkpoint(package.resolve("cBottle-3d.zip")) as checkpoint:
            core_model = checkpoint.read_model()

        core_model.eval()
        core_model.requires_grad_(False)
        core_model.float()

        # Manually handle the sst data, its a OpenDAP data source which isnt fsspec
        # supported so we are doing things manually. Hopefully we can clean this up
        # with the NGC file system
        sst_url = "https://esgf.ceda.ac.uk/thredds/dodsC/esg_cmip6/input4MIPs/CMIP6Plus/CMIP/PCMDI/PCMDI-AMIP-1-1-9/ocean/mon/tosbcs/gn/v20230512/"
        sst_file = (
            "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc"
        )
        sst_cached = os.path.join(Package.default_cache("cbottle"), sst_file)

        if not os.path.isfile(sst_cached):
            logger.warning("Downloading SST dataset for CBottle3D")
            sst_ds = xr.open_dataset(
                urljoin(sst_url, sst_file),
                engine="netcdf4",
                cache=False,
            ).load()
            sst_ds.to_netcdf(sst_cached)

        sst_ds = xr.open_dataset(
            sst_cached,
            engine="h5netcdf",
            storage_options=None,
            cache=False,
        ).load()

        return cls(core_model, sst_ds)
