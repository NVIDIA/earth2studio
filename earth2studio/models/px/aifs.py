# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
import importlib.metadata
import json
import zipfile
from collections import OrderedDict
from collections.abc import Generator, Iterator

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyError,
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    import anemoi.inference  # noqa: F401
    import anemoi.models  # noqa: F401
    import earthkit.regrid  # noqa: F401
    import ecmwf.opendata  # noqa: F401
    import flash_attn  # noqa: F401
except ImportError:
    OptionalDependencyFailure("aifs")

_SUPPORTED_AIFS_VERSIONS = ("1.0", "1.1")
_AIFS11_MIN_VERSIONS = {
    "anemoi-inference": "0.6.3",
    "anemoi-models": "0.5.0",
}

VARIABLES = [
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "w50",
    "w100",
    "w150",
    "w200",
    "w250",
    "w300",
    "w400",
    "w500",
    "w600",
    "w700",
    "w850",
    "w925",
    "w1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "u10m",
    "v10m",
    "d2m",
    "t2m",
    "lsm",
    "msl",
    "sdor",
    "skt",
    "slor",
    "sp",
    "tcw",
    "z",
    "cp06",
    "tp06",
    "cos_latitude",
    "cos_longitude",
    "sin_latitude",
    "sin_longitude",
    "cos_julian_day",
    "cos_local_time",
    "sin_julian_day",
    "sin_local_time",
    "insolation",
    "u100m",
    "v100m",
    "hcc",
    "lcc",
    "mcc",
    "ro",
    "sf",
    "ssrd06",
    "stl1",
    "stl2",
    "strd06",
    "swvl1",
    "swvl2",
    "tcc",
]  # from config.json >> dataset.variables

# NOTE:
# AIFS uses these as "generated forcings" that are computed at runtime and inserted
# into the model input tensor. The indices for these are derived from the checkpoint
# metadata (`ai-models.json`), since the ordering differs between checkpoint versions.
_AIFS_GENERATED_FORCING_CKPT_VARIABLES = [
    "cos_latitude",
    "cos_longitude",
    "sin_latitude",
    "sin_longitude",
    "cos_julian_day",
    "cos_local_time",
    "sin_julian_day",
    "sin_local_time",
    "insolation",
]
_AIFS_TIME_VARYING_FORCING_CKPT_VARIABLES = [
    "cos_julian_day",
    "cos_local_time",
    "sin_julian_day",
    "sin_local_time",
    "insolation",
]


@check_optional_dependencies()
class AIFS(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Artificial Intelligence Forecasting System (AIFS), a data driven forecast model
    developed by the European Centre for Medium-Range Weather Forecasts (ECMWF). AIFS is
    based on a graph neural network (GNN) encoder and decoder, and a sliding window
    transformer processor, and is trained on ECMWF's ERA5 re-analysis and ECMWF's
    operational numerical weather prediction (NWP) analyses.
    Consists of a single model with a time-step size of 6 hours.

    Note
    ----
    This model uses the checkpoints provided by ECMWF.
    Multiple checkpoint versions are supported. Use:

    - `AIFS.load_default_package()` for the default (AIFS-Single v1.0)
    - `AIFS.load_default_package(version="1.1")` for AIFS-Single v1.1

    The checkpoint metadata (`ai-models.json`) is used to derive the correct variable
    ordering and indices for each checkpoint version.
    For additional information see the following resources:

    - https://arxiv.org/abs/2406.01465
    - https://huggingface.co/ecmwf/aifs-single-1.0
    - https://huggingface.co/ecmwf/aifs-single-1.1
    - https://github.com/ecmwf/anemoi-core

    Parameters
    ----------
    model : torch.nn.Module
        Core PyTorch module with the pretrained AIFS weights loaded.
    latitudes : torch.Tensor
        Latitude values for the native octahedral grid, registered as a buffer for
        interpolation.
    longitudes : torch.Tensor
        Longitude values for the native octahedral grid, registered as a buffer for
        interpolation.
    interpolation_matrix : torch.Tensor
        CSR sparse matrix mapping ERA5 lat/lon inputs onto the octahedral grid.
    inverse_interpolation_matrix : torch.Tensor
        CSR sparse matrix mapping outputs from the octahedral grid back to ERA5
        lat/lon.

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        interpolation_matrix: torch.Tensor,
        inverse_interpolation_matrix: torch.Tensor,
        ckpt_variables: list[str] | None = None,
        variables: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("latitudes", latitudes)
        self.register_buffer("longitudes", longitudes)
        self.register_buffer("interpolation_matrix", interpolation_matrix)
        self.register_buffer(
            "inverse_interpolation_matrix", inverse_interpolation_matrix
        )

        # Variable ordering is checkpoint dependent (v1.0 vs v1.1 differ).
        # We keep both the raw checkpoint variable names and the Earth2Studio-facing
        # variable names aligned by index.
        if ckpt_variables is None or variables is None:
            self._ckpt_variables = VARIABLES
            self._variables = VARIABLES
        else:
            self._ckpt_variables = ckpt_variables
            self._variables = variables

        self._ckpt_name_to_index = {v: i for i, v in enumerate(self._ckpt_variables)}
        self._generated_forcing_data_indices = [
            self._ckpt_name_to_index[v]
            for v in _AIFS_GENERATED_FORCING_CKPT_VARIABLES
            if v in self._ckpt_name_to_index
        ]
        gen_set = set(self._generated_forcing_data_indices)

        # These are indices into the checkpoint variable ordering.
        self._input_full_data_indices = self.model.data_indices.data.input.full.to(
            dtype=torch.int64
        ).tolist()
        self._input_forcing_data_indices = (
            self.model.data_indices.data.input.forcing.to(dtype=torch.int64).tolist()
        )

        # Variables we actually fetch from a datasource (generated forcings excluded),
        # in the same order as `data_indices.data.input.full`.
        self._input_fetch_data_indices = [
            i for i in self._input_full_data_indices if i not in gen_set
        ]

        # Positions (0..102) within the model input tensor (which is ordered by
        # `data_indices.data.input.full`).
        self._time_varying_positions_in_input_full: dict[str, int] = {}
        for name in _AIFS_TIME_VARYING_FORCING_CKPT_VARIABLES:
            idx = self._ckpt_name_to_index.get(name)
            if idx is None:
                continue
            try:
                self._time_varying_positions_in_input_full[name] = (
                    self._input_full_data_indices.index(idx)
                )
            except ValueError:
                continue

        self._forcing_positions_in_input_full = [
            self._input_full_data_indices.index(i)
            for i in self._input_forcing_data_indices
        ]

    @property
    def input_variables(self) -> list[str]:
        return [self._variables[i] for i in self._input_fetch_data_indices]

    @property
    def output_variables(self) -> list[str]:
        gen_set = set(self._generated_forcing_data_indices)
        return [v for i, v in enumerate(self._variables) if i not in gen_set]

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(self.input_variables),
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model
        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(self.output_variables),
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        if input_coords is None:
            return output_coords

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key not in ["batch", "time"]:
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]

        output_coords["lead_time"] = (
            input_coords["lead_time"][-1] + output_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls, version: str = "1.0") -> Package:
        """Load prognostic package"""
        if version not in _SUPPORTED_AIFS_VERSIONS:
            raise ValueError(
                f"Unsupported AIFS-Single version '{version}'. "
                + f"Supported versions: {_SUPPORTED_AIFS_VERSIONS}"
            )
        if version == "1.1":
            cls._require_aifs11_optional_dependencies()
        package = Package(
            f"hf://ecmwf/aifs-single-{version}",
            cache_options={
                "cache_storage": Package.default_cache(f"aifs-single-{version}"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls, package: Package, version: str | None = None
    ) -> PrognosticModel:
        """Load prognostic from package"""
        if version is not None and version not in _SUPPORTED_AIFS_VERSIONS:
            raise ValueError(
                f"Unsupported AIFS-Single version '{version}'. "
                + f"Supported versions: {_SUPPORTED_AIFS_VERSIONS}"
            )
        if version == "1.1":
            cls._require_aifs11_optional_dependencies()

        # Load model
        ckpt_candidates: list[tuple[str, str]] = []
        if version == "1.0":
            ckpt_candidates = [("aifs-single-mse-1.0.ckpt", "1.0")]
        elif version == "1.1":
            ckpt_candidates = [("aifs-single-mse-1.1.ckpt", "1.1")]
        else:
            # Best-effort autodetect based on package root and available ckpt name
            if "aifs-single-1.0" in package.root:
                ckpt_candidates = [
                    ("aifs-single-mse-1.0.ckpt", "1.0"),
                    ("aifs-single-mse-1.1.ckpt", "1.1"),
                ]
            elif "aifs-single-1.1" in package.root:
                ckpt_candidates = [
                    ("aifs-single-mse-1.1.ckpt", "1.1"),
                    ("aifs-single-mse-1.0.ckpt", "1.0"),
                ]
            else:
                ckpt_candidates = [
                    ("aifs-single-mse-1.1.ckpt", "1.1"),
                    ("aifs-single-mse-1.0.ckpt", "1.0"),
                ]

        model_path = None
        resolved_version = None
        last_err: Exception | None = None
        for ckpt_name, v in ckpt_candidates:
            try:
                model_path = package.resolve(ckpt_name)
                resolved_version = v
                break
            except Exception as e:  # pragma: no cover - depends on remote FS
                last_err = e
                continue

        if model_path is None or resolved_version is None:
            msg = "Could not resolve any known AIFS-Single checkpoint from package."
            if version is not None:
                msg += f" Requested version={version}."
            raise FileNotFoundError(msg) from last_err

        # Ensure v1.1 dependency validation also runs for autodetected checkpoints.
        if resolved_version == "1.1":
            cls._require_aifs11_optional_dependencies()

        model = torch.load(
            model_path, weights_only=False, map_location=torch.ones(1).device
        )
        model.eval()

        # Define the path to the metadata file
        metadata_path = "inference-last/anemoi-metadata/ai-models.json"

        # Extract metadata and supporting arrays from the zip file
        with zipfile.ZipFile(model_path, "r") as zipf:  # NOTE: this is totally baffling
            # Load metadata
            metadata = json.load(zipf.open(metadata_path))

            # Load supporting arrays
            supporting_arrays = {}
            for key, entry in metadata.get("supporting_arrays_paths", {}).items():
                supporting_arrays[key] = np.frombuffer(
                    zipf.read(entry["path"]),
                    dtype=entry["dtype"],
                ).reshape(entry["shape"])

        # Checkpoint-specific variable ordering (v1.0 vs v1.1 differ)
        ckpt_variables = (
            metadata.get("dataset", {}).get("variables")
            if isinstance(metadata, dict)
            else None
        )
        if not isinstance(ckpt_variables, list) or not all(
            isinstance(v, str) for v in ckpt_variables
        ):
            ckpt_variables = VARIABLES

        variables = [cls._ckpt_var_to_e2s(v) for v in ckpt_variables]

        # Load interpolation matrix
        # TODO: Maybe change this to allow for multiple packages?
        interpolation_package = Package(
            "https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear",
            cache_options={
                "cache_storage": Package.default_cache(
                    f"aifs-single-{resolved_version}_interpolation_matrix"
                ),
                "same_names": True,
            },
        )
        interpolation_matrix_path = interpolation_package.resolve(
            "9533e90f8433424400ab53c7fafc87ba1a04453093311c0b5bd0b35fedc1fb83.npz"
        )
        interpolation_matrix = np.load(interpolation_matrix_path)
        torch_interpolation_matrix = torch.sparse_csr_tensor(
            crow_indices=torch.from_numpy(interpolation_matrix["indptr"]),
            col_indices=torch.from_numpy(interpolation_matrix["indices"]),
            values=torch.from_numpy(interpolation_matrix["data"]),
            size=(interpolation_matrix["shape"][0], interpolation_matrix["shape"][1]),
            dtype=torch.float64,
        )
        inverse_interpolation_package = Package(
            "https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear/",
            cache_options={
                "cache_storage": Package.default_cache(
                    f"aifs-single-{resolved_version}_inverse_interpolation_matrix"
                ),
                "same_names": True,
            },
        )
        inverse_interpolation_matrix_path = inverse_interpolation_package.resolve(
            "7f0be51c7c1f522592c7639e0d3f95bcbff8a044292aa281c1e73b842736d9bf.npz"
        )
        inverse_interpolation_matrix = np.load(inverse_interpolation_matrix_path)
        torch_inverse_interpolation_matrix = torch.sparse_csr_tensor(
            crow_indices=torch.from_numpy(inverse_interpolation_matrix["indptr"]),
            col_indices=torch.from_numpy(inverse_interpolation_matrix["indices"]),
            values=torch.from_numpy(inverse_interpolation_matrix["data"]),
            size=(
                inverse_interpolation_matrix["shape"][0],
                inverse_interpolation_matrix["shape"][1],
            ),
            dtype=torch.float64,
        )

        return cls(
            model,
            latitudes=torch.Tensor(supporting_arrays["latitudes"]).reshape(1, 1, -1, 1),
            longitudes=torch.Tensor(supporting_arrays["longitudes"]).reshape(
                1, 1, -1, 1
            ),
            interpolation_matrix=torch_interpolation_matrix,
            inverse_interpolation_matrix=torch_inverse_interpolation_matrix,
            ckpt_variables=ckpt_variables,
            variables=variables,
        )

    @staticmethod
    def _ckpt_var_to_e2s(name: str) -> str:
        """Translate checkpoint variable names into Earth2Studio variable IDs."""
        # Surface shorthand used by ECMWF / anemoi configs
        if name == "10u":
            return "u10m"
        if name == "10v":
            return "v10m"
        if name == "2d":
            return "d2m"
        if name == "2t":
            return "t2m"
        if name == "100u":
            return "u100m"
        if name == "100v":
            return "v100m"

        # 6-hour accumulations in Earth2Studio naming
        if name == "cp":
            return "cp06"
        if name == "tp":
            return "tp06"
        if name == "ssrd":
            return "ssrd06"
        if name == "strd":
            return "strd06"

        # Pressure level variables are encoded as e.g. q_50 in the checkpoint
        # but q50 in Earth2Studio.
        if "_" in name:
            parts = name.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return f"{parts[0]}{parts[1]}"

        return name

    @classmethod
    def _require_aifs11_optional_dependencies(cls) -> None:
        """Validate that the environment is compatible with loading AIFS-Single v1.1.

        This is primarily to provide a clearer, actionable error message when a user
        attempts to load the v1.1 checkpoint with a v1.0 anemoi stack.
        """

        def _ver(pkg: str) -> str:
            try:
                return importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError as e:
                raise OptionalDependencyError("aifs11", "AIFS (v1.1)", e) from e

        def _parse(v: str) -> tuple[int, ...]:
            # Minimal semantic-ish parsing to avoid an extra dependency on 'packaging'.
            core = v.split("+", 1)[0].split(".", 3)
            out: list[int] = []
            for part in core:
                num = ""
                for ch in part:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                out.append(int(num) if num else 0)
            return tuple(out)

        for pkg, min_v in _AIFS11_MIN_VERSIONS.items():
            installed = _ver(pkg)
            if _parse(installed) < _parse(min_v):
                err = ImportError(
                    f"{pkg}>={min_v} is required for AIFS-Single v1.1, "
                    + f"but found {installed}. Install with `uv add earth2studio --extra aifs11`."
                )
                raise OptionalDependencyError("aifs11", "AIFS (v1.1)", err) from err

    def get_cos_sin_julian_day(
        self,
        time_array: np.datetime64,
        longitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cosine and sine of Julian day"""
        days = (
            time_array.astype("datetime64[D]") - time_array.astype("datetime64[Y]")
        ).astype(np.float32)
        hours = (
            time_array.astype("datetime64[h]") - time_array.astype("datetime64[D]")
        ).astype(np.float32)
        julian_days = days + (hours / 24.0)
        normalized = 2 * np.pi * (julian_days / 365.25)
        cos_julian_day = torch.full_like(
            longitudes, np.cos(normalized), dtype=torch.float32
        )
        sin_julian_day = torch.full_like(
            longitudes, np.sin(normalized), dtype=torch.float32
        )
        return cos_julian_day, sin_julian_day

    def get_cos_sin_local_time(
        self,
        time_array: np.datetime64,
        longitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cosine and sine of local time"""
        hours = (
            time_array.astype("datetime64[h]") - time_array.astype("datetime64[D]")
        ).astype(np.float32)
        normalized_time = 2 * np.pi * (hours / 24.0)
        normalized_longitudes = 2 * np.pi * (longitudes / 360.0)
        tau = normalized_time + normalized_longitudes
        cos_local_time = torch.cos(tau)
        sin_local_time = torch.sin(tau)
        return cos_local_time, sin_local_time

    def get_cosine_zenith_fields(
        self,
        date: np.datetime64,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Get cosine zenith fields for input time array"""

        # Get Julian day
        days = (date.astype("datetime64[D]") - date.astype("datetime64[Y]")).astype(
            np.float32
        )
        hours = (date.astype("datetime64[h]") - date.astype("datetime64[D]")).astype(
            np.float32
        )
        seconds = (date.astype("datetime64[s]") - date.astype("datetime64[h]")).astype(
            np.float32
        )
        julian_day = days + seconds / 86400.0

        # Convert angle to tensor
        angle = torch.tensor(
            julian_day / 365.25 * torch.pi * 2, device=latitudes.device
        )

        # declination in [degrees]
        declination = (
            0.396372
            - 22.91327 * torch.cos(angle)
            + 4.025430 * torch.sin(angle)
            - 0.387205 * torch.cos(2 * angle)
            + 0.051967 * torch.sin(2 * angle)
            - 0.154527 * torch.cos(3 * angle)
            + 0.084798 * torch.sin(3 * angle)
        )

        # time correction in [h.degrees]
        time_correction = (
            0.004297
            + 0.107029 * torch.cos(angle)
            - 1.837877 * torch.sin(angle)
            - 0.837378 * torch.cos(2 * angle)
            - 2.340475 * torch.sin(2 * angle)
        )

        # Convert to radians
        declination = torch.deg2rad(declination)
        latitudes = torch.deg2rad(latitudes)

        # Calculate sine and cosine of declination and latitude
        sindec_sinlat = torch.sin(declination) * torch.sin(latitudes)
        cosdec_coslat = torch.cos(declination) * torch.cos(latitudes)

        # Solar hour angle
        solar_angle = torch.deg2rad((hours - 12) * 15 + longitudes + time_correction)
        zenith_angle = sindec_sinlat + cosdec_coslat * torch.cos(solar_angle)

        # Clip negative values
        return torch.clamp(zenith_angle, min=0.0)

    def _prepare_input(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepare input tensor and coordinates for the AIFS model."""
        # Interpolate the input tensor to the model grid
        shape = x.shape
        x = x.flatten(start_dim=4)
        x = x.flatten(end_dim=3)
        x = torch.swapaxes(x, 0, -1)
        x = x.to(dtype=torch.float64)
        x = self.interpolation_matrix @ x
        x = x.to(dtype=torch.float32)
        x = torch.swapaxes(x, 0, -1)
        x = x.reshape([shape[0] * shape[1], shape[2], shape[3], -1])
        x = torch.swapaxes(x, 2, 3)
        n_bt = shape[0] * shape[1]
        n_lead = shape[2]
        n_nodes = x.shape[2]

        # Reconstruct full feature tensor in checkpoint variable space (ordering and
        # indices are checkpoint dependent).
        x_full = torch.zeros(
            (n_bt, n_lead, n_nodes, len(self._variables)),
            device=x.device,
            dtype=torch.float32,
        )
        x_full[..., self._input_fetch_data_indices] = x

        # Compute generated forcings
        cos_latitude = torch.cos(torch.deg2rad(self.latitudes)).to(dtype=torch.float32)
        sin_latitude = torch.sin(torch.deg2rad(self.latitudes)).to(dtype=torch.float32)
        cos_longitude = torch.cos(torch.deg2rad(self.longitudes)).to(
            dtype=torch.float32
        )
        sin_longitude = torch.sin(torch.deg2rad(self.longitudes)).to(
            dtype=torch.float32
        )
        cos_latitude = cos_latitude.repeat(n_bt, n_lead, 1, 1)
        sin_latitude = sin_latitude.repeat(n_bt, n_lead, 1, 1)
        cos_longitude = cos_longitude.repeat(n_bt, n_lead, 1, 1)
        sin_longitude = sin_longitude.repeat(n_bt, n_lead, 1, 1)

        cos_julian_day_0, sin_julian_day_0 = self.get_cos_sin_julian_day(
            coords["time"][0] - np.timedelta64(6, "h"), self.longitudes
        )
        cos_julian_day_1, sin_julian_day_1 = self.get_cos_sin_julian_day(
            coords["time"][0], self.longitudes
        )
        cos_julian_day = torch.cat([cos_julian_day_0, cos_julian_day_1], dim=1).repeat(
            n_bt, 1, 1, 1
        )
        sin_julian_day = torch.cat([sin_julian_day_0, sin_julian_day_1], dim=1).repeat(
            n_bt, 1, 1, 1
        )

        cos_local_time_0, sin_local_time_0 = self.get_cos_sin_local_time(
            coords["time"][0] - np.timedelta64(6, "h"), self.longitudes
        )
        cos_local_time_1, sin_local_time_1 = self.get_cos_sin_local_time(
            coords["time"][0], self.longitudes
        )
        cos_local_time = torch.cat([cos_local_time_0, cos_local_time_1], dim=1).repeat(
            n_bt, 1, 1, 1
        )
        sin_local_time = torch.cat([sin_local_time_0, sin_local_time_1], dim=1).repeat(
            n_bt, 1, 1, 1
        )

        cos_zenith_angle_0 = self.get_cosine_zenith_fields(
            coords["time"][0] - np.timedelta64(6, "h"), self.latitudes, self.longitudes
        )
        cos_zenith_angle_1 = self.get_cosine_zenith_fields(
            coords["time"][0], self.latitudes, self.longitudes
        )
        cos_zenith_angle = torch.cat(
            [cos_zenith_angle_0, cos_zenith_angle_1], dim=1
        ).repeat(n_bt, 1, 1, 1)

        def _set(var: str, value: torch.Tensor) -> None:
            idx = self._ckpt_name_to_index.get(var)
            if idx is None:
                return
            x_full[:, :, :, idx : idx + 1] = value

        _set("cos_latitude", cos_latitude)
        _set("sin_latitude", sin_latitude)
        _set("cos_longitude", cos_longitude)
        _set("sin_longitude", sin_longitude)
        _set("cos_julian_day", cos_julian_day)
        _set("sin_julian_day", sin_julian_day)
        _set("cos_local_time", cos_local_time)
        _set("sin_local_time", sin_local_time)
        _set("insolation", cos_zenith_angle)

        # Select only model input features (ordered by data_indices.data.input.full)
        x = x_full[..., self._input_full_data_indices]
        return x

    def _update_input(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:
        """Update time based inputs."""

        # Select only inputs
        # From AnemoiModelInterface.DataIndices
        # https://anemoi.readthedocs.io/projects/models/en/latest/modules/data_indices.html#usage-information
        x = x[..., self.model.data_indices.data.input.full]

        # Get cos, sin of Julian day
        cos_julian_day, sin_julian_day = self.get_cos_sin_julian_day(
            coords["time"][0] + coords["lead_time"][0], self.longitudes
        )

        # Get cos, sin of local time
        cos_local_time, sin_local_time = self.get_cos_sin_local_time(
            coords["time"][0] + coords["lead_time"][0], self.longitudes
        )

        # Get cosine zenith angle
        cos_zenith_angle = self.get_cosine_zenith_fields(
            coords["time"][0] + coords["lead_time"][0], self.latitudes, self.longitudes
        )

        # Add terms to x
        pos = self._time_varying_positions_in_input_full.get("cos_julian_day")
        if pos is not None:
            x[:, 1:2, :, pos : pos + 1] = cos_julian_day
        pos = self._time_varying_positions_in_input_full.get("cos_local_time")
        if pos is not None:
            x[:, 1:2, :, pos : pos + 1] = cos_local_time
        pos = self._time_varying_positions_in_input_full.get("sin_julian_day")
        if pos is not None:
            x[:, 1:2, :, pos : pos + 1] = sin_julian_day
        pos = self._time_varying_positions_in_input_full.get("sin_local_time")
        if pos is not None:
            x[:, 1:2, :, pos : pos + 1] = sin_local_time
        pos = self._time_varying_positions_in_input_full.get("insolation")
        if pos is not None:
            x[:, 1:2, :, pos : pos + 1] = cos_zenith_angle

        return x

    def _prepare_output(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepare input tensor and coordinates for the AIFS model."""
        # Remove generated forcings
        all_indices = torch.arange(x.size(-1))
        keep = torch.isin(
            all_indices,
            torch.tensor(
                self._generated_forcing_data_indices, device=all_indices.device
            ),
            invert=True,
        )
        x = x[..., keep]
        shape = x.shape

        # Interpolate the model grid to the lat lon grid
        x = x[:, 1:2]
        x = x.flatten(end_dim=1)
        x = torch.swapaxes(x, 0, 1)
        x = x.flatten(start_dim=1)
        x = x.to(dtype=torch.float64)
        x = self.inverse_interpolation_matrix @ x
        x = x.to(dtype=torch.float32)
        x = torch.reshape(x, [x.shape[0], shape[0], shape[-1]])
        x = torch.swapaxes(x, 0, 1)
        x = torch.swapaxes(x, 1, 2)
        x = torch.reshape(
            x,
            [
                coords["batch"].shape[0],
                coords["time"].shape[0],
                coords["lead_time"].shape[0],
                coords["variable"].shape[0],
                coords["lat"].shape[0],
                coords["lon"].shape[0],
            ],
        )
        return x

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)
        with torch.autocast(device_type=str(x.device), dtype=torch.float16):
            y = self.model.predict_step(x)
            out = torch.empty(
                (x.shape[0], x.shape[1], x.shape[2], len(self._variables)),
                device=x.device,
            )
            out[..., 0, :, self.model.data_indices.data.input.full] = x[
                :,
                1,
            ]
            out[..., 1, :, self.model.data_indices.data.output.full] = y
            out[..., 1, :, self._input_forcing_data_indices] = x[
                :, 1, :, self._forcing_positions_in_input_full
            ]

        return out, output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 6 hours in the future
        """
        _ = self.output_coords(coords)  # NOTE: Quick fix for exception handling
        x = self._prepare_input(x, coords)
        x, coords = self._forward(x, coords)
        x = self._prepare_output(x, coords)
        return x, coords

    def _fill_input(self, x: torch.Tensor, coords: CoordSystem) -> torch.Tensor:
        out = torch.empty(
            (
                x.shape[0],
                x.shape[1],
                x.shape[2],
                len(self._variables),
                x.shape[4],
                x.shape[5],
            ),
            device=x.device,
        )
        indices = torch.cat(
            [
                self.model.data_indices.data.input.prognostic,
                self.model.data_indices.data.input.forcing,
            ]
        )

        # Sort the concatenated tensor
        indices = indices.sort().values

        # Remove generated forcings (computed at runtime)
        mask = ~torch.isin(
            indices,
            torch.tensor(
                self._generated_forcing_data_indices,
                device=indices.device,
                dtype=torch.int64,
            ),
        )

        out[:, :, 0, indices[mask]] = x[0, 0, 0, ...]
        out[:, :, 1, indices[mask]] = x[0, 0, 1, ...]

        keep = torch.ones(len(self._variables), dtype=torch.bool, device=out.device)
        for idx in self._generated_forcing_data_indices:
            keep[idx] = False
        out = out[:, :, :, keep, ...]
        selected = [self._variables[i] for i, k in enumerate(keep.tolist()) if k]

        out_coords = coords.copy()
        out_coords["variable"] = np.array(selected)

        return out, out_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        self.output_coords(coords)
        first_out, coords_out = self._fill_input(x, coords)
        coords_out["lead_time"] = coords["lead_time"][1:]
        yield first_out[:, :, 1:], coords_out

        # Prepare input tensor
        x = self._prepare_input(x, coords)

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward is identity operator
            y, coords_out = self._forward(x, coords)

            # Prepare output tensor
            output_tensor = self._prepare_output(y, coords_out)

            # Rear hook
            output_tensor, coords_out = self.rear_hook(output_tensor, coords_out)

            # Yield output tensor
            yield output_tensor, coords_out.copy()

            # Update coordinates
            coords["lead_time"] = (
                coords["lead_time"]
                + self.output_coords(self.input_coords())["lead_time"]
            )
            # Prepare input tensor
            x = self._update_input(y, coords)

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system
        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
