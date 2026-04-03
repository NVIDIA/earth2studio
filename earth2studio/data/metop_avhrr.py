# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import uuid
import zipfile
from datetime import datetime, timedelta

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import MetOpAVHRRLexicon
from earth2studio.lexicon.base import E2STUDIO_SCHEMA
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

try:
    import eumdac
    from satpy import Scene
except ImportError:
    OptionalDependencyFailure("data")
    eumdac = None  # type: ignore[assignment]
    Scene = None  # type: ignore[assignment,misc]

# Spacecraft ID mapping
_SPACECRAFT_MAP = {
    "1": "Metop-B",
    "2": "Metop-A",
    "3": "Metop-C",
    "4": "Metop-SGA1",
    "M01": "Metop-B",
    "M02": "Metop-A",
    "M03": "Metop-C",
}

# Subsampling factor for AVHRR (2048 pixels/line is very dense)
_DEFAULT_SUBSAMPLE = 16

# AVHRR EPS native navigation constants
_NAV_SAMPLE_RATE = 20  # pixels between navigation tie points
_NAV_NUM_POINTS = 103  # number of tie-point positions per scan line
_NAV_FIRST_PIXEL = 4  # 0-based pixel index of first tie point


def _parse_avhrr_with_satpy(
    nat_file: str,
    variables: list[str],
    subsample: int = _DEFAULT_SUBSAMPLE,
) -> pd.DataFrame:
    """Parse an AVHRR Level 1B EPS native file using satpy.

    Uses an optimized approach that loads only calibrated channel data through
    satpy (avoiding expensive geolocation interpolation) and reads raw
    navigation tie points directly from the EPS MDR records. This is ~12x
    faster than loading full-resolution lat/lon through satpy.

    Parameters
    ----------
    nat_file : str
        Path to the .nat format AVHRR file
    variables : list[str]
        E2S variable names to extract (e.g. ["avhrr01", "avhrr04"])
    subsample : int, optional
        Scan-line subsampling factor to reduce data volume, by default 16.
        Across-track sampling uses the 103 EPS navigation tie points
        (every 20th pixel) regardless of this setting.

    Returns
    -------
    pd.DataFrame
        One row per (pixel, channel) observation with columns:
        time, lat, lon, observation, variable, satellite,
        scan_angle, channel_index, solza, solaza, satellite_za, satellite_aza
    """
    import dask

    dask.config.set(num_workers=4)

    scn = Scene(reader="avhrr_l1b_eps", filenames=[nat_file])

    # Map e2s variable names to satpy dataset names
    var_to_satpy: dict[str, str] = {}
    for v in variables:
        satpy_name, _ = MetOpAVHRRLexicon[v]
        var_to_satpy[v] = satpy_name

    # Load only channel data (calibrated BT/radiance) — no geo datasets.
    # This triggers _read_all() internally but skips the expensive
    # geotie-point interpolation that satpy performs for lat/lon.
    satpy_datasets = list(set(var_to_satpy.values()))
    try:
        scn.load(satpy_datasets, generate=False)
    except Exception as exc:
        logger.warning("Failed to load datasets from {}: {}", nat_file, exc)
        return pd.DataFrame()

    # Access raw MDR records from satpy's file handler.
    # After scn.load(), the file handler has already parsed the binary file.
    reader = list(scn._readers.values())[0]
    fh = list(reader.file_handlers.values())[0][0]
    mdr = fh.sections[("mdr", np.int8(2))]

    # Read raw navigation tie points (103 per scan line, subsampled by line).
    # EARTH_LOCATIONS: (n_scans, 103, 2) big-endian int32, scale 1e4 degrees.
    # ANGULAR_RELATIONS: (n_scans, 103, 4) big-endian int16, scale 1e2 degrees.
    # Order within dim-2: [solar_zenith, sat_zenith, solar_azimuth, sat_azimuth].
    el = mdr["EARTH_LOCATIONS"][::subsample].compute()  # (N, 103, 2)
    ar = mdr["ANGULAR_RELATIONS"][::subsample].compute()  # (N, 103, 4)

    lats = (el[:, :, 0] / 1e4).astype(np.float32)
    lons = (el[:, :, 1] / 1e4).astype(np.float32)
    # Convert [-180, 180] -> [0, 360]
    lons = np.where(lons < 0, lons + 360.0, lons)

    solza = (ar[:, :, 0] / 100.0).astype(np.float32)
    satza = (ar[:, :, 1] / 100.0).astype(np.float32)
    solazi = (ar[:, :, 2] / 100.0).astype(np.float32)
    satazi = (ar[:, :, 3] / 100.0).astype(np.float32)

    n_lines, n_cols = lats.shape  # (subsampled scan lines, 103 tie points)
    n_pixels = n_lines * n_cols

    # Extract satellite name and sensing times from scene attributes
    satellite = "Metop"
    start_time: datetime | None = None
    end_time: datetime | None = None
    for ds_name in satpy_datasets:
        if ds_name in scn:
            attrs = scn[ds_name].attrs
            platform_name = attrs.get("platform_name", "")
            if platform_name:
                satellite = platform_name
            if start_time is None:
                start_time = attrs.get("start_time")
                end_time = attrs.get("end_time")

    if start_time is None:
        start_time = datetime(2000, 1, 1)
    if end_time is None:
        end_time = start_time

    # Compute per-scan-line times (linear interpolation between start/end)
    total_seconds = (end_time - start_time).total_seconds()
    # n_lines is the subsampled count; map back to original line indices
    n_total_lines = el.shape[0]  # same as n_lines (already subsampled above)
    if n_total_lines > 1:
        dt_per_line = total_seconds / (n_total_lines - 1)
    else:
        dt_per_line = 0.0

    times = np.empty(n_pixels, dtype="datetime64[ns]")
    for i in range(n_lines):
        line_time = start_time + timedelta(seconds=i * dt_per_line)
        line_ns = np.datetime64(line_time, "ns")
        times[i * n_cols : (i + 1) * n_cols] = line_ns

    lat_flat = lats.ravel()
    lon_flat = lons.ravel()
    solza_flat = solza.ravel()
    satza_flat = satza.ravel()
    solazi_flat = solazi.ravel()
    satazi_flat = satazi.ravel()

    # Pixel indices of the 103 navigation tie points across track
    nav_cols = np.arange(
        _NAV_FIRST_PIXEL,
        _NAV_FIRST_PIXEL + _NAV_NUM_POINTS * _NAV_SAMPLE_RATE,
        _NAV_SAMPLE_RATE,
    )[:_NAV_NUM_POINTS]

    # Build per-channel DataFrames, sampling channel data at tie-point columns
    frames: list[pd.DataFrame] = []
    channel_num_map = {"1": 1, "2": 2, "3a": 3, "3b": 4, "4": 5, "5": 6}

    for e2s_var, satpy_name in var_to_satpy.items():
        try:
            # Subsample scan lines, pick tie-point pixel columns
            data = scn[satpy_name].values[::subsample][:, nav_cols]
        except (KeyError, IndexError):
            logger.warning("Dataset '{}' not available, skipping", satpy_name)
            continue

        obs = data.ravel().astype(np.float32)
        ch_idx = MetOpAVHRRLexicon.VOCAB[e2s_var]
        ch_num = channel_num_map.get(ch_idx, 0)

        df = pd.DataFrame(
            {
                "time": pd.to_datetime(times),
                "lat": lat_flat,
                "lon": lon_flat,
                "observation": obs,
                "variable": e2s_var,
                "satellite": satellite,
                "scan_angle": satza_flat,
                "channel_index": np.full(n_pixels, ch_num, dtype=np.uint16),
                "solza": solza_flat,
                "solaza": solazi_flat,
                "satellite_za": satza_flat,
                "satellite_aza": satazi_flat,
            }
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    # Drop invalid observations (NaN obs, NaN coords, out-of-range lat)
    result = result.dropna(subset=["observation", "lat", "lon"])
    result = result[(result["lat"] >= -90) & (result["lat"] <= 90)]

    return result


@check_optional_dependencies()
class MetOpAVHRR:
    """EUMETSAT MetOp AVHRR Level 1B radiance and brightness temperature observations.

    The Advanced Very High Resolution Radiometer (AVHRR) is a 6-channel
    cross-track scanning radiometer aboard the MetOp series of polar-orbiting
    satellites. It measures calibrated radiances in visible (0.58-1.6 µm)
    and infrared (3.7-12.5 µm) bands at 1 km spatial resolution.

    Channels 1-2 and 3A provide reflectances (visible/NIR), while channels
    3B, 4, and 5 provide brightness temperatures (thermal IR). Channels 3A
    and 3B cannot operate simultaneously.

    This data source downloads Level 1B products from the EUMETSAT Data Store
    and uses the satpy ``avhrr_l1b_eps`` reader to parse the EPS native
    binary format.

    Parameters
    ----------
    satellite : str, optional
        Satellite platform filter. One of "Metop-B", "Metop-C", or None
        (all available). By default None.
    subsample : int, optional
        Spatial subsampling factor. AVHRR has 2048 pixels per scan line;
        subsampling reduces data volume. By default 16.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric ± window) or a tuple (lower, upper) for asymmetric
        windows, by default np.timedelta64(1, 'h')
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress and info, by default True
    async_timeout : int, optional
        Time in seconds after which download will be cancelled if not finished,
        by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount
    of data to your local machine for large requests. Each AVHRR orbit file
    is approximately 1 GB.

    Note
    ----
    Requires EUMETSAT Data Store credentials. Set the following environment
    variables:

    - ``EUMETSAT_CONSUMER_KEY``: Your EUMETSAT API consumer key
    - ``EUMETSAT_CONSUMER_SECRET``: Your EUMETSAT API consumer secret

    Register at https://eoportal.eumetsat.int/ to obtain credentials.

    Note
    ----
    Additional information on the data repository:

    - https://data.eumetsat.int/data/map/EO:EUM:DAT:METOP:AVHRRL1
    - https://user.eumetsat.int/s3/eup-strapi-media/pdf_ten_97231_eps_avhrr_l1_pgs_d2b7482b08.pdf

    Badges
    ------
    region:global dataclass:observation product:temp product:sat
    """

    SOURCE_ID = "earth2studio.data.metop_avhrr"
    COLLECTION_ID = "EO:EUM:DAT:METOP:AVHRRL1"

    SCHEMA = pa.schema(
        [
            E2STUDIO_SCHEMA.field("time"),
            E2STUDIO_SCHEMA.field("lat"),
            E2STUDIO_SCHEMA.field("lon"),
            E2STUDIO_SCHEMA.field("observation"),
            E2STUDIO_SCHEMA.field("variable"),
            E2STUDIO_SCHEMA.field("satellite"),
            E2STUDIO_SCHEMA.field("scan_angle"),
            E2STUDIO_SCHEMA.field("channel_index"),
            E2STUDIO_SCHEMA.field("solza"),
            E2STUDIO_SCHEMA.field("solaza"),
            E2STUDIO_SCHEMA.field("satellite_za"),
            E2STUDIO_SCHEMA.field("satellite_aza"),
        ]
    )

    def __init__(
        self,
        satellite: str | None = None,
        subsample: int = _DEFAULT_SUBSAMPLE,
        time_tolerance: TimeTolerance = np.timedelta64(1, "h"),
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ) -> None:
        self._satellite = satellite
        self._subsample = subsample
        self._cache = cache
        self._verbose = verbose
        self._tmp_cache_hash: str | None = None
        self.async_timeout = async_timeout

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

        self._consumer_key = os.environ.get("EUMETSAT_CONSUMER_KEY", "")
        self._consumer_secret = os.environ.get("EUMETSAT_CONSUMER_SECRET", "")
        if not self._consumer_key or not self._consumer_secret:
            logger.warning(
                "EUMETSAT_CONSUMER_KEY and/or EUMETSAT_CONSUMER_SECRET not set. "
                "Data fetching will fail."
            )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Function to get AVHRR radiance/brightness temperature observations.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpAVHRRLexicon
            (e.g. "avhrr01", "avhrr04", "avhrr3b").
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            AVHRR observation data in long format.
        """
        try:
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        df = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, variable, fields), timeout=self.async_timeout
            )
        )

        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        return df

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get AVHRR data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variables to return. Must be in MetOpAVHRRLexicon.
        fields : str | list[str] | pa.Schema | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            AVHRR observation data in long format.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        schema = self.resolve_fields(fields)

        # Validate variables against lexicon
        for v in variable_list:
            if v not in MetOpAVHRRLexicon.VOCAB:
                raise KeyError(
                    f"Variable '{v}' not found in MetOpAVHRRLexicon. "
                    f"Available: {list(MetOpAVHRRLexicon.VOCAB.keys())}"
                )

        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Compute overall time window
        all_times = [
            t.replace(tzinfo=None) if hasattr(t, "tzinfo") and t.tzinfo else t
            for t in time_list
        ]
        dt_min = min(all_times) + self._tolerance_lower
        dt_max = max(all_times) + self._tolerance_upper

        # Download products
        nat_files = await asyncio.to_thread(self._download_products, dt_min, dt_max)

        if not nat_files:
            logger.warning(
                "No AVHRR products found for time range {} to {}", dt_min, dt_max
            )
            return self._empty_dataframe(schema)

        # Parse each product with satpy
        frames: list[pd.DataFrame] = []
        for fpath in nat_files:
            df = await asyncio.to_thread(
                _parse_avhrr_with_satpy,
                fpath,
                variable_list,
                self._subsample,
            )
            if not df.empty:
                frames.append(df)

        if not frames:
            return self._empty_dataframe(schema)

        result = pd.concat(frames, ignore_index=True)

        # Filter by time windows
        time_masks = []
        for t in all_times:
            t_min = t + self._tolerance_lower
            t_max = t + self._tolerance_upper
            mask = (result["time"] >= pd.Timestamp(t_min)) & (
                result["time"] <= pd.Timestamp(t_max)
            )
            time_masks.append(mask)

        if time_masks:
            combined_mask = time_masks[0]
            for mask in time_masks[1:]:
                combined_mask = combined_mask | mask
            result = result[combined_mask]

        # Select requested fields
        available_cols = [c for c in schema.names if c in result.columns]
        result = result[available_cols].reset_index(drop=True)

        return result

    def _download_products(self, dt_start: datetime, dt_end: datetime) -> list[str]:
        """Download AVHRR products from EUMETSAT Data Store.

        Parameters
        ----------
        dt_start : datetime
            Search window start (UTC)
        dt_end : datetime
            Search window end (UTC)

        Returns
        -------
        list[str]
            Paths to downloaded native format files
        """
        token = eumdac.AccessToken(
            credentials=(self._consumer_key, self._consumer_secret)
        )
        datastore = eumdac.DataStore(token)
        collection = datastore.get_collection(self.COLLECTION_ID)

        search_kwargs: dict = {
            "dtstart": dt_start,
            "dtend": dt_end,
        }
        if self._satellite:
            # eumdac search API expects friendly satellite names
            search_kwargs["sat"] = self._satellite

        products = collection.search(**search_kwargs)

        downloaded: list[str] = []
        for product in products:
            if self._verbose:
                logger.info(
                    "Downloading AVHRR product: {} ({}–{})",
                    product,
                    getattr(product, "sensing_start", "?"),
                    getattr(product, "sensing_end", "?"),
                )

            # Find the .nat file entry
            nat_entry = None
            try:
                entries = product.entries
                for entry in entries:
                    if str(entry).endswith(".nat"):
                        nat_entry = entry
                        break
            except Exception:  # noqa: S110
                pass

            cache_name = f"{product}.nat"
            cache_path = os.path.join(self.cache, cache_name)
            if os.path.isfile(cache_path):
                downloaded.append(cache_path)
                continue

            try:
                with product.open(entry=nat_entry) as stream:
                    raw = stream.read()

                # Handle ZIP-wrapped products
                if raw[:2] == b"PK":
                    import io

                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        nat_names = [n for n in zf.namelist() if n.endswith(".nat")]
                        if nat_names:
                            raw = zf.read(nat_names[0])

                with open(cache_path, "wb") as f:
                    f.write(raw)
                downloaded.append(cache_path)
            except Exception as exc:
                logger.warning("Failed to download product {}: {}", product, exc)

        return downloaded

    def _empty_dataframe(self, schema: pa.Schema) -> pd.DataFrame:
        """Create an empty DataFrame matching the schema.

        Parameters
        ----------
        schema : pa.Schema
            Target schema

        Returns
        -------
        pd.DataFrame
            Empty DataFrame with correct columns
        """
        return pd.DataFrame({name: pd.Series(dtype="object") for name in schema.names})

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Convert fields parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | None
            Field specification. None returns the full SCHEMA.

        Returns
        -------
        pa.Schema
            A PyArrow schema containing only the requested fields

        Raises
        ------
        KeyError
            If a requested field name is not in the SCHEMA
        TypeError
            If a field type doesn't match the SCHEMA
        """
        if fields is None:
            return cls.SCHEMA

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            for field in fields:
                if field.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{field.name}' not found in class SCHEMA. "
                        f"Available fields: {cls.SCHEMA.names}"
                    )
                expected_type = cls.SCHEMA.field(field.name).type
                if field.type != expected_type:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, "
                        f"expected {expected_type} from class SCHEMA"
                    )
            return fields

        selected_fields = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not found in class SCHEMA. "
                    f"Available fields: {cls.SCHEMA.names}"
                )
            selected_fields.append(cls.SCHEMA.field(name))

        return pa.schema(selected_fields)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "metop_avhrr")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_metop_avhrr_{self._tmp_cache_hash}"
            )
        return cache_location
