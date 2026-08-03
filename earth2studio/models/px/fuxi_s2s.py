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

import os
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.models.utils import create_ort_session
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from onnxruntime import InferenceSession  # type: ignore[import-untyped]
except ImportError:
    OptionalDependencyFailure("fuxi-s2s")
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
_ZENODO_ROOT = "https://zenodo.org/records/15718402/files"
_MODEL_ARCHIVE = "model-1.0.tar?download=1"


def _atomic_copy(source: Any, destination: Path) -> Path:
    """Copy an archive stream to a destination atomically."""
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


def _extract_tar_member(
    archive_path: str,
    member_name: str,
    destination: Path,
) -> Path:
    """Extract one regular-file member from a tar archive."""
    if destination.exists():
        return destination

    with tarfile.open(archive_path, mode="r:*") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as error:
            raise FileNotFoundError(
                f"Could not find {member_name!r} in {archive_path}"
            ) from error
        if not member.isfile():
            raise ValueError(f"Archive member {member_name!r} is not a regular file")
        source = archive.extractfile(member)
        if source is None:
            raise OSError(f"Could not read archive member {member_name!r}")
        with source:
            return _atomic_copy(source, destination)


def _resolve_default_assets(package: Package) -> Package:
    """Download and safely unpack the official Zenodo model assets."""
    asset_directory = Path(package.cache) / "assets"
    model_archive = package.resolve(_MODEL_ARCHIVE)

    _extract_tar_member(
        model_archive,
        "model-1.0/fuxi_s2s.onnx",
        asset_directory / "fuxi_s2s.onnx",
    )
    _extract_tar_member(
        model_archive,
        "model-1.0/fuxi_s2s",
        asset_directory / "fuxi_s2s",
    )
    return Package(str(asset_directory))


@check_optional_dependencies()
class FuXiS2S(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """FuXi-S2S global daily-mean prognostic model.

    FuXi-S2S consumes daily means from two consecutive UTC calendar days and
    predicts the following daily mean. A timestamp at 00:00 UTC labels the
    corresponding calendar-day aggregate; it is not an instantaneous midnight
    state.

    For more information see:

    - https://www.nature.com/articles/s41467-024-50714-1
    - https://github.com/tpys/FuXi-S2S
    - https://zenodo.org/records/15718402

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
    Sea-surface temperature must retain ``NaN`` values over land. The wrapper
    does not aggregate hourly fields or regrid initial conditions; callers must
    provide these prepared daily inputs through an Earth2Studio data source.

    The official ONNX graph samples flow-dependent perturbations internally, so
    each forecast trajectory is one stochastic ensemble member. Member ``00`` in
    the official inference script is the first stochastic member, not a
    deterministic control. To generate multiple members with Earth2Studio, use
    :func:`earth2studio.run.ensemble` with
    :class:`earth2studio.perturbation.Zero`. ``Zero`` prevents an additional
    initial-condition perturbation while preserving FuXi-S2S's internal
    stochastic sampling.

    Warning
    -------
    The official checkpoint is licensed under CC BY-NC-ND 4.0. Its Zenodo
    record restricts it to research use and prohibits commercial or competition
    use without prior author permission. These restrictions apply to the
    checkpoint, independently of Earth2Studio's Apache-2.0 source-code license.

    Badges
    ------
    region:global class:s2s product:wind product:precip product:temp product:atmos
    product:ocean year:2024
    """

    def __init__(self, onnx_path: str) -> None:
        super().__init__()

        self.register_buffer("device_buffer", torch.empty(0))
        self.onnx_path = onnx_path
        self.ort: InferenceSession | None = None
        self._time_step = np.timedelta64(1, "D")

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordSystem
            Coordinate system for two consecutive UTC daily means.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [np.timedelta64(-1, "D"), np.timedelta64(0, "D")]
                ),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 121, endpoint=True),
                "lon": np.linspace(0, 360, 240, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Validate input coordinates and return the next daily coordinates.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates with two consecutive daily lead times.

        Returns
        -------
        CoordSystem
            Output coordinates for the daily mean one day after the latest
            input.
        """
        target_input_coords = self.input_coords()
        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )

        for index, key in enumerate(target_input_coords):
            handshake_dim(test_coords, key, index)
            if key not in ("batch", "time"):
                handshake_coords(test_coords, target_input_coords, key)
        self._initial_step(input_coords)

        output_coords = input_coords.copy()
        output_coords["lead_time"] = input_coords["lead_time"][-1:] + self._time_step
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the official FuXi-S2S package from Zenodo.

        Returns
        -------
        Package
            Package pointing to the official Zenodo record.

        Note
        ----
        The official checkpoint is licensed CC BY-NC-ND 4.0 and restricted to
        non-commercial research use.
        """
        return Package(
            _ZENODO_ROOT,
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
            The default Zenodo package is unpacked into this layout
            automatically.

        Returns
        -------
        PrognosticModel
            Loaded FuXi-S2S prognostic wrapper.
        """
        if package.root.rstrip("/") == _ZENODO_ROOT:
            package = _resolve_default_assets(package)

        package.resolve("fuxi_s2s")
        onnx_path = package.resolve("fuxi_s2s.onnx")
        return cls(onnx_path)

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

    def _initial_step(self, coords: CoordSystem) -> int:
        lead_days = float(coords["lead_time"][-1] / self._time_step)
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
        coords: CoordSystem,
        step: int,
    ) -> torch.Tensor:
        """Run one FuXi-S2S ONNX step."""
        ort_session = self._get_ort_session()
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
            coords["time"] + coords["lead_time"][-1],
            x.shape[0],
        )
        day_of_year = self._day_of_year(valid_times)

        for index in range(model_input.shape[0]):
            ort_inputs = {
                "input": model_input[index : index + 1].contiguous().cpu().numpy()
            }
            if "step" in input_names:
                ort_inputs["step"] = np.array([step], dtype=np.float32)
            if "doy" in input_names:
                ort_inputs["doy"] = np.array(
                    [day_of_year[index]],
                    dtype=np.float32,
                )

            sample_output = ort_session.run(
                [output_name],
                ort_inputs,
            )[0]
            output[index : index + 1] = torch.from_numpy(sample_output).to(
                device=model_input.device,
            )

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
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run FuXi-S2S one daily step.

        Parameters
        ----------
        x : torch.Tensor
            Two consecutive UTC daily means.
        coords : CoordSystem
            Coordinates describing the input tensor.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Predicted next daily mean and its coordinates.
        """
        step = self._initial_step(coords)
        output_coords = self.output_coords(coords)
        x = x.to(self.device_buffer.device)
        rolling_output = self._forward(x, coords, step)
        return rolling_output[:, :, -1:], output_coords

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Advance FuXi-S2S while retaining its two-day rolling state."""
        self.output_coords(coords)
        coords = coords.copy()
        x = x.to(self.device_buffer.device)

        initial_coords = coords.copy()
        initial_coords["lead_time"] = coords["lead_time"][-1:]
        yield x[:, :, -1:], initial_coords

        while True:
            x, coords = self.front_hook(x, coords)
            output_coords = self.output_coords(coords)
            step = self._initial_step(coords)
            rolling_output = self._forward(x, coords, step)
            prediction = rolling_output[:, :, -1:]
            prediction, output_coords = self.rear_hook(prediction, output_coords)
            rolling_output = torch.cat(
                (rolling_output[:, :, :-1], prediction),
                dim=2,
            )
            yield prediction, output_coords.copy()

            x = rolling_output
            coords["lead_time"] = np.concatenate(
                (coords["lead_time"][-1:], output_coords["lead_time"])
            )

    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create a daily FuXi-S2S forecast iterator.

        Parameters
        ----------
        x : torch.Tensor
            Two consecutive UTC daily means.
        coords : CoordSystem
            Coordinates describing the input tensor.

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Initial current day followed by successive daily predictions.
        """
        yield from self._default_generator(x, coords)
