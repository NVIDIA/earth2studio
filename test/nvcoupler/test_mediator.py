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

import numpy as np
import pytest
import torch

from earth2studio.nvcoupler.dictionary import (
    DEFAULT_DICTIONARY,
    CellMethod,
    FieldDictionary,
    FieldEntry,
)
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.field import Field
from earth2studio.nvcoupler.mediator import (
    AccumulationMediator,
    TrailingAverageMediator,
)
from earth2studio.nvcoupler.testing import grid_coords

T0 = np.datetime64("2024-01-01")
H = np.timedelta64(1, "h")


def _z1000(value, hours):
    return Field(
        torch.full((4, 8), float(value)),
        grid_coords(4, 8),
        "geopotential_at_1000hpa",
        "m2 s-2",
        valid_time=T0 + hours * H,
    )


def test_trailing_average_over_window():
    med = TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])
    assert med.timestep == np.timedelta64(48, "h")
    assert med.import_names == ["geopotential_at_1000hpa"]
    # 8 x 6h samples with values 1..8 -> mean 4.5
    for i in range(1, 9):
        med.import_state.add(_z1000(i, 6 * i))
    med.run(T0 + np.timedelta64(48, "h"))
    out = med.export_state["geopotential_at_1000hpa_48h_mean"]
    assert torch.allclose(out.data, torch.full((4, 8), 4.5))
    assert out.valid_time == T0 + np.timedelta64(48, "h")
    assert med.samples_last_window["geopotential_at_1000hpa_48h_mean"] == 8
    # accumulator reset: next window with values 10, 20 -> mean 15
    med.import_state.add(_z1000(10, 54))
    med.import_state.add(_z1000(20, 60))
    med.run(T0 + np.timedelta64(96, "h"))
    assert torch.allclose(
        med.export_state["geopotential_at_1000hpa_48h_mean"].data,
        torch.full((4, 8), 15.0),
    )


def test_duplicate_valid_time_ignored():
    med = TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])
    f = _z1000(4, 6)
    med.import_state.add(f)
    med.import_state.add(f)  # same valid_time -> not double counted
    med.run(T0 + np.timedelta64(48, "h"))
    assert torch.allclose(
        med.export_state["geopotential_at_1000hpa_48h_mean"].data,
        torch.full((4, 8), 4.0),
    )


def test_sum_and_max_reductions():
    d = FieldDictionary(DEFAULT_DICTIONARY)
    med_sum = AccumulationMediator(
        "psum", ["total_precipitation_48h_sum"], dictionary=d
    )
    for i, v in enumerate([1.0, 2.0, 3.0]):
        med_sum.import_state.add(
            Field(
                torch.full((4, 8), v),
                grid_coords(4, 8),
                "total_precipitation_6h",
                "kg m-2",
                valid_time=T0 + i * H,
            )
        )
    med_sum.run(T0 + np.timedelta64(48, "h"))
    assert torch.allclose(
        med_sum.export_state["total_precipitation_48h_sum"].data,
        torch.full((4, 8), 6.0),
    )

    med_max = AccumulationMediator("tmax", ["air_temperature_2m_24h_max"], dictionary=d)
    for i, v in enumerate([280.0, 295.0, 290.0]):
        med_max.import_state.add(
            Field(
                torch.full((4, 8), v),
                grid_coords(4, 8),
                "air_temperature_2m",
                "K",
                valid_time=T0 + i * H,
            )
        )
    med_max.run(T0 + np.timedelta64(24, "h"))
    assert torch.allclose(
        med_max.export_state["air_temperature_2m_24h_max"].data,
        torch.full((4, 8), 295.0),
    )


def test_two_reductions_of_same_base_field():
    """Two derived fields of one base (24h max AND 24h mean of t2m) must
    each accumulate every delivery — regression for the base->derived map
    silently keeping only the last derived field."""
    d = FieldDictionary(DEFAULT_DICTIONARY)
    d.register(
        FieldEntry(
            "air_temperature_2m_24h_mean",
            "K",
            "24 h mean 2 m temperature",
            frozenset(),
            CellMethod("air_temperature_2m", "mean", np.timedelta64(24, "h")),
        )
    )
    med = AccumulationMediator(
        "med",
        ["air_temperature_2m_24h_max", "air_temperature_2m_24h_mean"],
        dictionary=d,
    )
    # the shared base import is deduped, not advertised twice
    assert med.import_names == ["air_temperature_2m"]
    values = [280.0, 295.0, 290.0]
    for i, v in enumerate(values):
        med.import_state.add(
            Field(
                torch.full((4, 8), v),
                grid_coords(4, 8),
                "air_temperature_2m",
                "K",
                valid_time=T0 + i * H,
            )
        )
    med.run(T0 + np.timedelta64(24, "h"))
    assert torch.allclose(
        med.export_state["air_temperature_2m_24h_max"].data,
        torch.full((4, 8), 295.0),
    )
    assert torch.allclose(
        med.export_state["air_temperature_2m_24h_mean"].data,
        torch.full((4, 8), sum(values) / len(values)),
    )
    assert med.samples_last_window["air_temperature_2m_24h_max"] == 3
    assert med.samples_last_window["air_temperature_2m_24h_mean"] == 3


def test_mediator_requires_no_ic():
    med = TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])
    assert med.requires_ic is False
    med.initialize()  # no-arg initialize is safe


def test_compute_without_samples_raises():
    med = TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])
    with pytest.raises(CouplingError, match="no samples"):
        med.run(T0 + np.timedelta64(48, "h"))


def test_field_without_cell_method_rejected():
    with pytest.raises(CouplingError, match="cell_method"):
        AccumulationMediator("med", ["sea_surface_temperature"])


def test_trailing_average_rejects_non_mean():
    with pytest.raises(CouplingError, match="not mean"):
        TrailingAverageMediator("med", ["total_precipitation_48h_sum"])


def test_mixed_windows_need_explicit_window():
    d = FieldDictionary(DEFAULT_DICTIONARY)
    with pytest.raises(CouplingError, match="differing"):
        AccumulationMediator(
            "med",
            ["geopotential_at_1000hpa_48h_mean", "air_temperature_2m_24h_max"],
            dictionary=d,
        )


def test_gradient_flows_through_mean():
    med = TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])
    xs = [torch.full((4, 8), float(i), requires_grad=True) for i in (1, 3)]
    for i, x in enumerate(xs):
        med.import_state.add(
            Field(
                x,
                grid_coords(4, 8),
                "geopotential_at_1000hpa",
                "m2 s-2",
                valid_time=T0 + i * H,
            )
        )
    med.run(T0 + np.timedelta64(48, "h"))
    med.export_state["geopotential_at_1000hpa_48h_mean"].data.sum().backward()
    for x in xs:
        assert x.grad is not None and torch.allclose(x.grad, torch.full((4, 8), 0.5))
