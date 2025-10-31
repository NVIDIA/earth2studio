from collections.abc import Callable
from datetime import datetime, timedelta

import os
import numpy as np
import xarray as xr
from huggingface_hub import hf_hub_download
from scipy.special import roots_legendre

from earth2studio.data.base import DataSource
from earth2studio.data.arco import ARCO
from earth2studio.data.rx import SurfaceGeoPotential
from earth2studio.data.utils import prep_data_inputs
from earth2studio.models.auto import Package
from earth2studio.utils.type import TimeArray, VariableArray

def _build_variable_mappings() -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional mappings between Earth2Studio (E2S) and FME variable names.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Mapping from Earth2Studio variable names to FME variable names and vice versa
    """
    mapping: dict[str, str] = {
        # Near-surface and forcing
        "u10m": "UGRD10m",
        "v10m": "VGRD10m",
        "t2m": "TMP2m",
        "q2m": "Q2m",
        "sp": "PRESsfc",
        "skt": "surface_temperature",
        "z": "HGTsfc",
        "mtdwswrf": "DSWRFtoa",
        "land_abs": "land_fraction",
        "ocean_abs": "ocean_fraction",
        "sic_abs": "sea_ice_fraction",
        "global_mean_co2": "global_mean_co2",
        "t850": "TMP850",
        "z500": "h500",
        # Precip: E2S "tp" is accumulation; map to rate placeholder
        "tp": "PRATEsfc",
        # Diagnostics
        "mtuwlwrf": "ULWRFtoa",
        "msuwlwrf": "ULWRFsfc",
        "msdwswrf": "DSWRFsfc",
        "msdwlwrf": "DLWRFsfc",
        "msuwswrf": "USWRFsfc",
        "mtuwswrf": "USWRFtoa",
        "msshf": "SHTFLsfc",
        "mslhf": "LHTFLsfc",
        "mttwp": "tendency_of_total_water_path_due_to_advection",
    }
    # Model levels (k = 0..7)
    for k in range(8):
        mapping[f"u{k}k"] = f"eastward_wind_{k}"
        mapping[f"v{k}k"] = f"northward_wind_{k}"
        mapping[f"t{k}k"] = f"air_temperature_{k}"
        mapping[f"qtot{k}k"] = f"specific_total_water_{k}"

    fme_to_e2s = {v: k for k, v in mapping.items()}
    return mapping, fme_to_e2s


# Public constants for reuse across model and data modules
E2S_TO_FME, FME_TO_E2S = _build_variable_mappings()

# ACE2 uses F90 regular gaussian grid internally
# https://confluence.ecmwf.int/display/OIFS/4.3+OpenIFS%3A+Horizontal+Resolution+and+Configurations
ACE_GRID_LAT = np.degrees(np.arcsin(roots_legendre(2 * 90)[0]))
ACE_GRID_LON = np.linspace(0.5, 359.5, 4*90, endpoint=True)

class ACE2ERA5Data:
    """ACE2-ERA5 data source providing forcing or initial-conditions data.
    Files are downloaded on-demand and cached automatically, or loaded from a user-specified
    local directory when `local=True`. Data are served as-is; no transformations are applied.

    Provides all input variables described in the ACE2-ERA5 paper.

    Parameters
    ----------
    base_path : str
        HuggingFace repo ID or local directory path. Defaults to `allenai/ACE2-ERA5`.
    mode : str
        Either "forcing" or "initial_conditions". Controls which data tree and filenames are used.
        Defaults to "forcing".
    forcing_subdir : str, optional
        Subdirectory for forcing data. Defaults to `forcing_data` containing files `forcing_YYYY.nc`.
    ic_subdir : str, optional
        Subdirectory for initial conditions. Defaults to `initial_conditions` containing files `ic_YYYY.nc`.
    local : bool, optional
        If True, data are loaded from a local directory instead of downloaded from HuggingFace. Defaults to False.

    References
    ----------
    - ACE2-ERA5 paper: https://arxiv.org/html/2411.11268v1
    """

    _IC_ALLOWED_YEARS = [1940, 1950, 1979, 2001, 2020]

    def __init__(
        self,
        base_path: str = "allenai/ACE2-ERA5",
        mode: str = "forcing",
        forcing_subdir: str = "forcing_data",
        ic_subdir: str = "initial_conditions",
        local: bool = False,
    ) -> None:
        if mode not in ["forcing", "initial_conditions"]:
            raise ValueError("mode must be either 'forcing' or 'initial_conditions'")
        self._base_path = base_path
        self._mode = mode
        self._forcing_subdir = forcing_subdir
        self._ic_subdir = ic_subdir
        self._local = local
        self.lat = ACE_GRID_LAT
        self.lon = ACE_GRID_LON

    def _validate_ic_times(self, time_list: list[datetime]) -> None:
        for t in time_list:
            if t.year not in self._IC_ALLOWED_YEARS:
                raise ValueError(
                    f"Initial condition time year {t.year} is not supported. Allowed years: {self._IC_ALLOWED_YEARS}"
                )
            if not (t.day == 1 and t.hour == 0 and t.minute == 0 and t.second == 0 and t.microsecond == 0):
                raise ValueError(
                    "Initial condition times must be the first of each month at 00:00 UTC"
                )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        time_list, var_list_e2s = prep_data_inputs(time, variable)

        if self._mode == "initial_conditions":
            self._validate_ic_times(time_list)

        # Map requested Earth2Studio variable names to FME names present in files
        try:
            var_list_fme = [E2S_TO_FME[v] for v in var_list_e2s]
        except KeyError as e:
            raise KeyError(f"Unknown ACE2ERA5 variable id: {v}")

        # Determine years to fetch
        years = sorted({t.year for t in time_list})
        paths: list[str] = []
        for y in years:
            if self._mode == "forcing":
                filename = f"{self._forcing_subdir}/forcing_{y}.nc"
                cache_name = "ace2era5"
            else:
                filename = f"{self._ic_subdir}/ic_{y}.nc"
                cache_name = "ace2era5"

            if self._local:
                path = os.path.join(self._base_path, filename)
            else:
                pkg = Package(
                    f"hf://{self._base_path}",
                    cache_options={
                        "cache_storage": Package.default_cache(cache_name),
                        "same_names": True,
                    },
                )
                path = pkg.resolve(filename)

            paths.append(path)

        # Open and concat across years
        dsets = [xr.open_dataset(p, engine="netcdf4") for p in paths]
        ds = xr.concat(dsets, dim="time") if len(dsets) > 1 else dsets[0]

        # Standardize lat/lon coord names
        if "latitude" in ds.coords or "longitude" in ds.coords:
            ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.coords})

        # Subset time and variables; select exact requested timestamps and order
        ds = ds.sel(time=time_list)
        ds = ds[var_list_fme]

        lat_coords = ds["lat"].values
        lon_coords = ds["lon"].values

        # Build output DataArray [time, variable, lat, lon] with E2S variable names order
        arrays = []
        for fme_name in var_list_fme:
            da = ds[fme_name]
            # Ensure dims ordered [time, lat, lon]
            if "time" in da.dims and "lat" in da.dims and "lon" in da.dims:
                da = da.transpose("time", "lat", "lon")
            elif "lat" in da.dims and "lon" in da.dims:
                da = da.expand_dims("time").assign_coords(time=time_list).transpose("time", "lat", "lon")
            elif "time" in da.dims:
                # CO2 is time-only
                da = da.expand_dims({"lat": lat_coords, "lon": lon_coords})
                da = da.transpose("time", "lat", "lon")
            else:
                raise ValueError(f"Unknown ACE2 variable dims: {da.dims}")
            arrays.append(da)
        stacked = xr.concat(arrays, dim="variable")
        stacked = stacked.assign_coords(variable=np.array(var_list_e2s, dtype=object))
        # Ensure canonical dim order
        stacked = stacked.transpose("time", "variable", "lat", "lon")

        # Use predefined lat/lon coords which are equivalent up to machine precision
        # Mitigates errors that would otherwise arise from checks in `handshake_coords`
        stacked = stacked.assign_coords(lat=self.lat, lon=self.lon)
        return stacked


class ACE2ARCOForcingData:
    """ACE2 forcing data source backed by ARCO (ERA5). Due to slight differences in the originating
    ERA5 source and the regridding operation, the results from using the ACE2 model forced by this
    data source may differ slightly from the model forced by the officially released forcing data.

    Users may pass their own global mean CO2 forcing by defining a callable that returns the CO2 concentration
    (ppm) for a given UTC datetime, and passing it as the `co2_fn` parameter.
    
    Provides the following variables:
    - mtdwswrf: Mean top-downward shortwave radiative flux at TOA (W/m^2), mean
    - skt: Skin temperature
    - z: Surface height of topography (m), invariant (from geopotential/g)
    - land_abs: Land grid cell area fraction invariant (from ERA5 land_sea_mask)
    - ocean_abs: Ocean grid cell area fraction. Fills NaNs with 0.0.
    - sic_abs: Absolute sea-ice grid cell area fraction. Fills NaNs with 0.0. (from ERA5 sea_ice_cover and land_sea_mask)
    - global_mean_co2: Global mean atmospheric CO2 (ppm), computed viauser-supplied time function

    Parameters
    ----------
    co2_fn : Callable[[datetime], float]
        Function returning CO2 concentration (ppm) for a given UTC datetime.
    arco : ARCO | None
        Optional ARCO instance to reuse; if None, a new instance is created.
    g : float
        Gravitational acceleration for converting geopotential to meters.
    """

    _FORCING_VARIABLES = [
        "mtdwswrf",
        "skt",
        "z",
        "land_abs",
        "ocean_abs",
        "sic_abs",
        "global_mean_co2",
    ]

    def __init__(
        self,
        co2_fn: Callable[[datetime], float],
        g: float = 9.80665,
    ) -> None:
        self._source = ARCO()
        self._co2_fn = co2_fn
        self.lat = ARCO.ARCO_LAT
        self.lon = ARCO.ARCO_LON
        self._g = g
    
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray = _FORCING_VARIABLES,
    ) -> xr.DataArray:

        time_list, var_list = prep_data_inputs(time, variable)

        # Determine required ARCO variables
        need_arco: set[str] = set()
        for v in var_list:
            if v in ["skt", "z"]:
                need_arco.add(v)
            elif v == "land_abs":
                need_arco.add("lsm")
            elif v in ["ocean_abs", "sic_abs"]:
                need_arco.add("lsm")
                need_arco.add("sic")
            elif v in ["global_mean_co2", "mtdwswrf"]:
                # CO2 scalar handled by Callable, no ARCO dependency; mtdwswrf handled by special case below
                pass
            else:
                raise KeyError(f"Unknown ACE2 forcing variable id: {v}")

        arco_da: xr.DataArray | None = None
        if len(need_arco) > 0:
            arco_da = self._source(time=time_list, variable=sorted(list(need_arco)))

        # Prepare output array
        out = xr.DataArray(
            data=np.empty((len(time_list), len(var_list), len(self.lat), len(self.lon)), dtype=np.float32),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time_list,
                "variable": var_list,
                "lat": self.lat,
                "lon": self.lon,
            },
        )

        # Fill each requested variable
        for j, v in enumerate(var_list):
            if v in ["skt"]:
                out[:, j] = arco_da.sel(variable=v)
            elif v == "z":
                out[:, j] = arco_da.sel(variable="z")/self._g
            elif v == "land_abs":
                out[:, j] = arco_da.sel(variable="lsm")
            elif v == "ocean_abs":
                lsm = arco_da.sel(variable="lsm")
                sic = arco_da.sel(variable="sic")
                ocean_abs = (1.0 - lsm) * (1.0 - sic)
                ocean_abs = ocean_abs.fillna(0.0)
                out[:, j] = ocean_abs.clip(min=0.0, max=1.0)
            elif v == "sic_abs":
                lsm = arco_da.sel(variable="lsm")
                sic = arco_da.sel(variable="sic")
                sic_abs = sic * (1.0 - lsm)
                sic_abs = sic_abs.fillna(0.0)
                out[:, j] = sic_abs.clip(min=0.0, max=1.0)
            elif v == "global_mean_co2":
                # Broadcast scalar across spatial grid per time
                values = np.array([self._co2_fn(t) for t in time_list], dtype=np.float32)
                # reshape to [time, 1, lat, lon] then broadcast
                tile = values.reshape(len(time_list), 1, 1)
                tile = np.broadcast_to(tile, (len(time_list), len(self.lat), len(self.lon)))
                out[:, j] = tile
            elif v != "mtdwswrf":
                raise KeyError(f"Unhandled ACE2 forcing variable id: {v}")

        # Special treatment for mtdwswrf: need to pull multiple times and average between them
        # This is to match the insolation distribution expected by ACE2
        if "mtdwswrf" in var_list:
            base_times = time_list.copy()
            unrolled_times = []
            for t in time_list:
                # Pull the 5 1hr steps preceding each 6hr timestep, then average to get 6hr mean
                if t.hour % 6 != 0:
                    raise ValueError(f"Times requested from ACE2ERA5ForcingData must be in whole hour multiples of 6hrs if requesting mtdwswrf variable")
                unrolled_times.extend([t+timedelta(hours=i) for i in range(-5, 1, 1)])
            flux_da = self._source(time=unrolled_times, variable=["mtdwswrf"])
            flux_da = flux_da.coarsen(time=6, boundary='exact').mean(['time']).sel(variable="mtdwswrf")
            flux_da = flux_da.assign_coords(time=base_times)
            out[:, var_list.index("mtdwswrf")] = flux_da

        return out
    