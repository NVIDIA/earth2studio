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

"""Schema-compat check: NNJAAIObsSat/NNJAAIObsConv -> HealDA.prep_conv /
prep_sat_sensor.

These tests exercise the *real* prep_conv/prep_sat_sensor transformation
code against NNJA-AI-sourced DataFrames, without requiring a GPU, earth2grid,
nvidia-physicsnemo, or the (multi-GB) model checkpoint -- only the small
per-sensor normalization-stats CSVs from the HealDA HF package, which
``prep_sat_sensor`` needs to build its channel LUT. ``prep_conv`` does not
touch ``self`` at all, so it is called unbound.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nnja_ai", reason="nnja-ai not installed")

import earth2studio.models.da.healda as healda_mod  # noqa: E402
from earth2studio.data import NNJAAIObsConv, NNJAAIObsSat  # noqa: E402


def _load_sensor_stats(package, sensors):
    sensor_stats: dict[str, dict[str, np.ndarray]] = {}
    for sensor in sensors:
        df = pd.read_csv(package.resolve(f"stats/{sensor}_normalizations.csv"))
        df = df[df["Platform_ID"] == -1].sort_values("Raw_Channel_ID")
        means = df["obs_mean"].to_numpy(dtype=np.float32)
        stds = df["obs_std"].to_numpy(dtype=np.float32)
        raw_ids = df["Raw_Channel_ID"].to_numpy()
        max_raw = int(raw_ids.max())
        lut = np.full(max_raw + 1, 0, dtype=int)
        for local_idx, raw in enumerate(raw_ids, start=1):
            lut[int(raw)] = local_idx
        sensor_stats[sensor] = {"means": means, "stds": stds, "raw_to_local": lut}
    return sensor_stats


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
def test_nnja_ai_conv_obs_compatible_with_healda_prep_conv():
    ds = NNJAAIObsConv(time_tolerance=timedelta(minutes=30))
    df = ds(datetime(2024, 1, 1, 0), ["t", "u", "v", "q", "pres"])
    assert not df.empty

    prepped = healda_mod.HealDA.prep_conv(None, df)
    assert set(prepped["local_channel"].unique()).issubset(
        set(healda_mod.CONV_VAR_CHANNEL.values())
    )
    assert prepped["sensor"].eq("conv").all()
    assert not prepped["observation"].isna().any()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
def test_nnja_ai_sat_obs_compatible_with_healda_prep_sat_sensor():
    ds = NNJAAIObsSat(time_tolerance=timedelta(minutes=30))
    df = ds(datetime(2024, 1, 1, 0), ["amsua"])
    assert not df.empty

    package = healda_mod.HealDA.load_default_package()
    sensor_stats = _load_sensor_stats(package, ["amsua"])
    fake_self = SimpleNamespace(_sensor_stats=sensor_stats)

    prepped = healda_mod.HealDA.prep_sat_sensor(fake_self, df, "amsua")
    assert prepped["local_channel"].min() >= 0
    assert prepped["sensor"].eq("amsua").all()
    assert not prepped["observation"].isna().any()
