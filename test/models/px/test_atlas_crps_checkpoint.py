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

from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from earth2studio import run
from earth2studio.data import Random
from earth2studio.io import ZarrBackend
from earth2studio.models.px import AtlasCRPS
from earth2studio.perturbation import Zero
from earth2studio.utils.checkpoint import Checkpoint
from earth2studio.utils.type import CoordSystem


class _Processor(torch.nn.Module):
    downsample_grid_shape = (2, 3)

    @property
    def normalizer_in(self) -> "_Processor":
        return self

    normalizer_out = normalizer_in

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    unnormalize = normalize

    def intep(self, x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        return x

    def preprocess_input(
        self, x: torch.Tensor, date: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x

    def preprocess_conditioning(
        self, high_res: torch.Tensor, low_res: torch.Tensor
    ) -> torch.Tensor:
        return low_res

    def postprocess(self, x: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
        return x_cur + 2 * x  # Physical and latent trajectories must differ.


class _Transformer(torch.nn.Module):
    def forward(self, prev: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return 0.1 * prev + 0.2 * current + torch.randn_like(current) + torch.rand(())


class _Decoder(torch.nn.Module):
    def forward(self, high_res: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual


class _SmallAtlasCRPS(AtlasCRPS):
    def __init__(self) -> None:
        # Dummy components do not require the optional pretrained-model dependencies.
        AtlasCRPS.__init__.__wrapped__(
            self, _Transformer(), _Processor(), _Decoder(), _Processor()
        )

    def input_coords(self) -> CoordSystem:
        return CoordSystem(
            batch=np.empty(0),
            time=np.empty(0),
            lead_time=np.array([-self.DT, np.timedelta64(0, "h")]),
            variable=np.array(["t2m"]),
            lat=np.arange(2),
            lon=np.arange(3),
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        coords = input_coords.copy()
        coords["lead_time"] = coords["lead_time"][-1:] + self.DT
        return coords


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("level,stop", [(2, 0), (2, 2), (1, 2), (0, 2)])
def test_atlas_crps_checkpoint_rollout(
    tmp_path: Path, device: str, level: int, stop: int
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = _SmallAtlasCRPS().to(device)
    coords = model.input_coords()
    coords["batch"] = np.arange(2)
    coords["time"] = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    x = torch.arange(48, device=device, dtype=torch.float32).reshape(2, 2, 2, 1, 2, 3)
    torch.manual_seed(123)
    expected = list(islice(model.create_iterator(x, coords), 5))

    torch.manual_seed(123)
    with Checkpoint("atlas", path=tmp_path, level=level) as ckpt:
        model = _SmallAtlasCRPS().to(device)
        iterator = model.create_iterator(x, coords)
        for _ in range(stop + 1):
            actual, actual_coords = next(iterator)
            ckpt.write(lead_time=actual_coords["lead_time"][-1])
        torch.testing.assert_close(actual, expected[stop][0], rtol=0, atol=0)
    iterator.close()

    torch.manual_seed(987)
    with Checkpoint("atlas", path=tmp_path, level=level):
        model = _SmallAtlasCRPS().to(device)
        iterator = model.create_iterator(x, coords)
        if level < 2:
            actual, actual_coords = next(iterator)
            torch.testing.assert_close(actual, x[:, :, -1:])
            assert actual_coords["lead_time"][-1] == np.timedelta64(0, "h")
        else:
            for wanted, wanted_coords in expected[stop + 1 :]:
                actual, actual_coords = next(iterator)
                assert list(actual_coords) == list(wanted_coords)
                for key in wanted_coords:
                    np.testing.assert_array_equal(
                        actual_coords[key], wanted_coords[key]
                    )
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    iterator.close()


@pytest.mark.parametrize("stop", [1, 3])
def test_atlas_crps_checkpoint_ensemble(tmp_path: Path, stop: int) -> None:
    checkpoint = Checkpoint(
        "atlas", path=tmp_path, mode="append", history_size=20, flush_interval=1
    )
    torch.manual_seed(123)
    np.random.seed(123)
    with checkpoint:
        model = _SmallAtlasCRPS()
        domain = CoordSystem(lat=np.arange(2), lon=np.arange(3))
        expected = run.ensemble(
            ["2024-01-01"],
            3,
            5,
            model,
            Random(domain),
            ZarrBackend(),
            Zero(),
            batch_size=2,
            device=torch.device("cpu"),
            verbose=False,
            checkpoint=checkpoint,
        )

    # Select an on-disk boundary in the first batch, then resume across later batches.
    torch.manual_seed(987)
    np.random.seed(123)
    with checkpoint.select(stop) as selected:
        model = _SmallAtlasCRPS()
        actual = run.ensemble(
            ["2024-01-01"],
            3,
            5,
            model,
            Random(domain),
            ZarrBackend(),
            Zero(),
            batch_size=2,
            device=torch.device("cpu"),
            verbose=False,
            checkpoint=selected,
        )
    np.testing.assert_array_equal(actual["t2m"][2:], expected["t2m"][2:])
    if stop < 3:
        np.testing.assert_array_equal(
            actual["t2m"][:2, :, stop + 1 :], expected["t2m"][:2, :, stop + 1 :]
        )
