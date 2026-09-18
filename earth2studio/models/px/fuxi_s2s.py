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

import hashlib
import os
import shutil
import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.models.utils import create_ort_session
from earth2studio.utils import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    from onnxruntime import InferenceSession  # type: ignore[import-untyped]
except ImportError:
    OptionalDependencyFailure("fuxi")
    InferenceSession = TypeVar("InferenceSession")  # type: ignore

PRESSURE_LEVELS = (
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    50,
)

VARIABLES = [
    *[f"z{level}" for level in PRESSURE_LEVELS],
    *[f"t{level}" for level in PRESSURE_LEVELS],
    *[f"u{level}" for level in PRESSURE_LEVELS],
    *[f"v{level}" for level in PRESSURE_LEVELS],
    *[f"q{level}" for level in PRESSURE_LEVELS],
    "t2m",
    "d2m",
    "sst",
    "ttr",
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "msl",
    "tcwv",
    "tp",
]

_TTR_INDEX = VARIABLES.index("ttr")
_TP_INDEX = VARIABLES.index("tp")

# Keep ONNX channel names/indices separate from public temporal quantity labels.
DAILY_VARIABLES = [
    (
        f"{variable}:mean:1h:25h"
        if variable in ("tp", "ttr")
        else f"{variable}:mean:0h:24h"
    )
    for variable in VARIABLES
]


def _atomic_copy(source: Any, destination: Path) -> Path:
    """Copy a file stream to a destination atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(source, temporary)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _resolve_model_assets(package: Package) -> Path:
    """Resolve an ONNX graph beside its external data file."""
    external_path = Path(package.resolve("fuxi_s2s"))
    onnx_path = Path(package.resolve("fuxi_s2s.onnx"))
    if onnx_path.name == "fuxi_s2s.onnx" and external_path == onnx_path.with_name(
        "fuxi_s2s"
    ):
        return onnx_path

    identity_parts = [package.root]
    for path in (external_path, onnx_path):
        file_stat = path.stat()
        identity_parts.extend(
            (
                str(path),
                str(file_stat.st_dev),
                str(file_stat.st_ino),
                str(file_stat.st_size),
                str(file_stat.st_mtime_ns),
            )
        )
    identity = hashlib.sha256("\0".join(identity_parts).encode()).hexdigest()
    cache_storage = getattr(package.fs, "storage", None)
    if isinstance(cache_storage, (list, tuple)) and cache_storage:
        cache_directory = Path(cache_storage[-1]).resolve()
    else:
        cache_directory = onnx_path.parent.resolve()
    asset_directory = cache_directory / "fuxi_s2s_assets" / identity

    with external_path.open("rb") as source:
        _atomic_copy(source, asset_directory / "fuxi_s2s")
    with onnx_path.open("rb") as source:
        return _atomic_copy(source, asset_directory / "fuxi_s2s.onnx")


@check_optional_dependencies()
class FuXiS2S(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
    """FuXi-S2S global daily-mean prognostic model.

    FuXi-S2S consumes daily means from two consecutive UTC calendar days and
    predicts the following daily mean. A timestamp at 00:00 UTC labels the
    corresponding calendar-day aggregate; it is not an instantaneous midnight
    state.

    Note
    ----
    This model uses the ONNX checkpoint from the original publication repository. For
    additional information see the following resources:

    - https://www.nature.com/articles/s41467-024-50714-1
    - https://github.com/tpys/FuXi-S2S
    - https://zenodo.org/records/15718402
    - https://huggingface.co/datasets/FudanFuXi/FuXi-S2S

    Parameters
    ----------
    onnx_path : str
        Path to the FuXi-S2S ONNX graph. Its external weight file named
        ``fuxi_s2s`` must be in the same directory.

    Note
    ----
    Initial conditions must contain two consecutive UTC daily means on the
    model's 1.5-degree grid. Instantaneous fields use calendar-day averages
    from 00--23 UTC. Accumulated ``tp`` and ``ttr`` fields use the 24
    interval-ending values from 01 UTC through 00 UTC of the following day.
    Their daily means retain the units of each one-hour accumulation; for
    example, multiply predicted ``tp`` by 24 to obtain a daily total.
    Public variable labels declare these windows explicitly:
    ``t2m:mean:0h:24h``, ``tp:mean:1h:25h``, and ``ttr:mean:1h:25h``.
    The compact ``mean:24h`` modifier instead describes the preceding day
    and must not be substituted at the same start-of-day timestamp.
    Sea-surface temperature must retain ``NaN`` values over land. The wrapper
    does not aggregate hourly fields or regrid initial conditions; callers must
    provide these prepared daily inputs through an Earth2Studio data source.

    The official ONNX graph samples flow-dependent perturbations internally, so
    each forecast trajectory is one stochastic ensemble member. Member ``00`` in
    the official inference script is the first stochastic member, not a
    deterministic control.

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Example
    -------
    Prepared daily data must carry the qualified labels returned by
    ``input_coords()["variable"]``. For formal input data consult
    https://zenodo.org/records/15718402. A statistics-aware fetch layer can
    split each label at its first colon, fetch the base variable over the
    declared window, and reduce it with ``apply_time_statistic``. Use hourly
    source data to reproduce the daily sampling convention.

    ```python
    package = FuXiS2S.load_default_package()
    model = FuXiS2S.load_model(package).to("cuda:0")
    requested_variables = model.input_coords()["variable"]
    # Includes "t2m:mean:0h:24h" and "tp:mean:1h:25h".
    ```

    Badges
    ------
    region:global class:subseasonal-seasonal product:wind product:precip product:temp
    product:atmos product:ocean year:2024 gpu:40gb backend:onnx
    """

    def __init__(self, onnx_path: str) -> None:
        super().__init__()

        self.register_buffer("device_buffer", torch.empty(0))
        self.onnx_path = onnx_path
        self.ort: InferenceSession | None = None
        self._time_step = np.timedelta64(1, "D")

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Coordinate system for two consecutive UTC daily means.
        """
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array(
                    [np.timedelta64(-1, "D"), np.timedelta64(0, "D")]
                ),
                "variable": np.array(DAILY_VARIABLES),
                "lat": np.linspace(90, -90, 121, endpoint=True),
                "lon": np.linspace(0, 360, 240, endpoint=False),
            },
            dynamic=("batch", "time"),
            statistics={label: label.split(":", 1)[1] for label in DAILY_VARIABLES},
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate input coordinates and return the next daily coordinates.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinates with two consecutive daily lead times.

        Returns
        -------
        CoordinateSystem
            Output coordinates for the daily mean one day after the latest
            input.
        """
        if "lead_time" not in input_coords.coords:
            raise ValueError("Missing lead_time coordinate")
        lead = input_coords.lead_time.values
        if (
            input_coords.lead_time.dims != ("lead_time",)
            or lead.size != 2
            or not np.issubdtype(lead.dtype, np.timedelta64)
            or np.isnat(lead).any()
        ):
            raise ValueError("lead_time must contain two finite timedeltas")
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        self._initial_step(input_coords)
        return coord_array_like(
            input_coords, {"lead_time": lead[-1:] + self._time_step}
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the FuXi-S2S package from an immutable Hugging Face mirror.

        Returns
        -------
        Package
            Package pointing to the mirrored FuXi-S2S checkpoint.

        Note
        ----
        The mirror contains unchanged assets from the official Zenodo record.
        The checkpoint is licensed CC BY-NC-ND 4.0 and restricted to
        non-commercial research use by its authors.
        """
        return Package(
            "hf://Artamta/FuXi-S2S-ONNX@5d7a6b132aaaaa070d2856d002f95911140db0ff",
            cache_options={
                "cache_storage": Package.default_cache("fuxi_s2s"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> PrognosticModel:
        """Load FuXi-S2S from an Earth2Studio package.

        Parameters
        ----------
        package : Package
            Package containing ``fuxi_s2s.onnx`` and its external data file.

        Returns
        -------
        PrognosticModel
            Loaded FuXi-S2S prognostic wrapper.
        """
        onnx_path = _resolve_model_assets(package)
        return cls(str(onnx_path))

    def to(self, device: str | torch.device | int) -> PrognosticModel:
        """Move the wrapper and ONNX Runtime session to a device.

        Parameters
        ----------
        device : str | torch.device | int
            Target PyTorch device.

        Returns
        -------
        PrognosticModel
            This model on the requested device.
        """
        target_device = torch.device(device)
        if target_device.index is None and target_device.type == "cuda":
            target_device = torch.device("cuda", torch.cuda.current_device())

        current_device = self.device_buffer.device
        super().to(target_device)
        if self.ort is not None and target_device != current_device:
            self.ort = create_ort_session(self.onnx_path, target_device)
        return self

    def _get_ort_session(self) -> InferenceSession:
        """Create the ONNX Runtime session on first use."""
        if self.ort is None:
            self.ort = create_ort_session(self.onnx_path, self.device_buffer.device)
        return self.ort

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert Earth2Studio fields to FuXi-S2S model units."""
        model_input = x.clone()

        ttr = model_input.select(-3, _TTR_INDEX)
        ttr.div_(3600.0)

        tp = model_input.select(-3, _TP_INDEX)
        tp.copy_(
            torch.clamp(
                torch.nan_to_num(tp, nan=0.0) * 1000.0,
                min=0.0,
                max=1000.0,
            )
        )

        return model_input

    def _prepare_output(self, x: torch.Tensor) -> torch.Tensor:
        """Convert FuXi-S2S output to Earth2Studio units."""
        output = x.clone()

        ttr = output.select(-3, _TTR_INDEX)
        ttr.mul_(3600.0)

        tp = output.select(-3, _TP_INDEX)
        tp.div_(1000.0)

        return output

    def _initial_step(self, coords: CoordinateSystem) -> int:
        lead_days = float(coords["lead_time"].values[-1] / self._time_step)
        if not np.isfinite(lead_days) or lead_days < 0 or not lead_days.is_integer():
            raise ValueError(
                "Latest lead time must be a non-negative whole number of days"
            )
        return int(lead_days)

    @staticmethod
    def _day_of_year(time: np.ndarray) -> np.ndarray:
        """Return FuXi-S2S day-of-year encoding."""
        day: np.ndarray = time.astype("datetime64[D]")
        year_start: np.ndarray = time.astype("datetime64[Y]").astype("datetime64[D]")
        day_of_year = (day - year_start).astype(np.int64) + 1
        return np.minimum(day_of_year, 365).astype(np.float32) / 365.0

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordinateSystem,
        step: int,
    ) -> torch.Tensor:
        """Run one FuXi-S2S ONNX step.

        Uses ORT IO bindings so GPU tensors stay on-device throughout
        inference, avoiding redundant GPU-to-CPU-to-GPU copies.

        Note
        ----
        See the `ONNX Runtime Python API
        <https://onnxruntime.ai/docs/api/python/api_summary.html>`_
        for details on the IO-binding interface.
        """
        ort_session = self._get_ort_session()
        device = self.device_buffer.device
        input_names = {model_input.name for model_input in ort_session.get_inputs()}
        output_name = ort_session.get_outputs()[0].name

        model_input = self._prepare_input(x.float()).reshape(
            -1,
            len(self.input_coords()["lead_time"]),
            len(VARIABLES),
            len(self.input_coords()["lat"]),
            len(self.input_coords()["lon"]),
        )
        output = torch.empty_like(model_input)
        valid_times = np.tile(
            coords["time"].values + coords["lead_time"].values[-1],
            x.shape[0],
        )
        day_of_year = self._day_of_year(valid_times)

        for index in range(model_input.shape[0]):
            binding = ort_session.io_binding()

            # Bind the main tensor input directly from device memory
            sample_input = model_input[index : index + 1].contiguous()
            binding.bind_input(
                name="input",
                device_type=device.type,
                device_id=device.index if device.index is not None else 0,
                element_type=np.float32,
                shape=tuple(sample_input.shape),
                buffer_ptr=sample_input.data_ptr(),
            )

            # Scalar auxiliaries are always CPU-resident
            if "step" in input_names:
                step_tensor = torch.tensor(
                    [step], dtype=torch.float32, device=torch.device("cpu")
                )
                binding.bind_input(
                    name="step",
                    device_type="cpu",
                    device_id=0,
                    element_type=np.float32,
                    shape=(1,),
                    buffer_ptr=step_tensor.data_ptr(),
                )
            if "doy" in input_names:
                doy_tensor = torch.tensor(
                    [day_of_year[index]],
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
                binding.bind_input(
                    name="doy",
                    device_type="cpu",
                    device_id=0,
                    element_type=np.float32,
                    shape=(1,),
                    buffer_ptr=doy_tensor.data_ptr(),
                )

            # Bind output on the same device as the input tensor
            sample_output = torch.empty_like(sample_input).contiguous()
            binding.bind_output(
                name=output_name,
                device_type=device.type,
                device_id=device.index if device.index is not None else 0,
                element_type=np.float32,
                shape=tuple(sample_output.shape),
                buffer_ptr=sample_output.data_ptr(),
            )

            ort_session.run_with_iobinding(binding)
            output[index : index + 1] = sample_output

        output = output.reshape(
            x.shape[0],
            x.shape[1],
            len(self.input_coords()["lead_time"]),
            len(VARIABLES),
            len(self.input_coords()["lat"]),
            len(self.input_coords()["lon"]),
        )
        prediction = self._prepare_output(output[:, :, -1:]).to(dtype=x.dtype)
        return torch.cat((x[:, :, -1:], prediction), dim=2)

    @batch_func()
    def _step(self, x: xr.DataArray) -> xr.DataArray:
        signature = self.output_coords(x)
        if "time" not in x.coords or x.time.dims != ("time",):
            raise ValueError("A one-dimensional time coordinate is required")
        times = x.time.values
        if not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
            raise ValueError("time must contain finite datetimes")
        tensor, _ = x.e2s.to_torch()
        tensor = tensor.to(self.device_buffer.device)
        rolling = self._forward(tensor, x, self._initial_step(x))
        output = from_torch(rolling[:, :, -1:], signature, name=x.name)
        output.encoding = x.encoding.copy()
        return output

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Run FuXi-S2S one daily step.

        Parameters
        ----------
        x : xr.DataArray
            Two consecutive UTC daily means, backed by NumPy or CuPy, with
            qualified variable labels and temporal-statistics metadata.

        Returns
        -------
        xr.DataArray
            Predicted next daily mean on the model device.
        """
        return self._step(x)

    def _default_generator(
        self,
        x: xr.DataArray,
    ) -> Generator[xr.DataArray, None, None]:
        """Advance FuXi-S2S while retaining its two-day rolling state."""
        self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        encoding = x.encoding.copy()
        x = from_torch(tensor.to(self.device_buffer.device), x)
        x.encoding = encoding
        yield x.isel(lead_time=slice(-1, None)).copy(deep=False)

        while True:
            x = self.front_hook(x)
            prediction = self.rear_hook(self._step(x))
            previous, _ = x.isel(lead_time=slice(-1, None)).e2s.to_torch()
            future, _ = prediction.e2s.to_torch()
            signature = coord_array_like(
                prediction,
                {
                    "lead_time": np.concatenate(
                        (x.lead_time.values[-1:], prediction.lead_time.values)
                    )
                },
            )
            x = from_torch(
                torch.cat(
                    (previous.to(future.device), future),
                    dim=x.get_axis_num("lead_time"),
                ),
                signature,
            )
            x.encoding = prediction.encoding.copy()
            yield prediction.copy(deep=False)

    def create_iterator(
        self,
        x: xr.DataArray,
    ) -> Iterator[xr.DataArray]:
        """Create a daily FuXi-S2S forecast iterator.

        Parameters
        ----------
        x : xr.DataArray
            Two consecutive prepared UTC daily means with coordinates and statistics.

        Yields
        ------
        xr.DataArray
            Initial current day followed by successive daily predictions.
        """
        yield from self._default_generator(x)
