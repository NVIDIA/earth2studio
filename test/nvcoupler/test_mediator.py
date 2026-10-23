# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.field import Field
from earth2studio.nvcoupler.mediator import AccumulationMediator

DERIVED = "geopotential_at_1000hpa_48h_mean"
BASE = "geopotential_at_1000hpa"


def _field(value: float, hour: int) -> Field:
    return Field(
        xr.DataArray(
            np.full((2, 3), value),
            dims=("lat", "lon"),
            coords={"lat": [10.0, -10.0], "lon": [0.0, 120.0, 240.0]},
        ),
        BASE,
        "m2 s-2",
        valid_time=np.datetime64("2024-01-01") + np.timedelta64(hour, "h"),
        source="atmos",
    )


def test_accumulation_mediator_emits_dataarray_mean() -> None:
    mediator = AccumulationMediator("mean", [DERIVED])
    mediator.import_state.add(_field(2.0, 0))
    mediator.import_state.add(_field(4.0, 6))
    mediator.compute(np.datetime64("2024-01-03"))
    output = mediator.export_state[DERIVED]
    assert isinstance(output.array, xr.DataArray)
    assert np.allclose(output.array, 3.0)
    assert mediator.samples_last_window[DERIVED] == 2


def test_duplicate_valid_time_is_ignored() -> None:
    mediator = AccumulationMediator("mean", [DERIVED])
    mediator.import_state.add(_field(2.0, 0))
    mediator.import_state.add(_field(10.0, 0))
    mediator.compute(np.datetime64("2024-01-03"))
    assert np.allclose(mediator.export_state[DERIVED].array, 2.0)


def test_compute_requires_samples() -> None:
    mediator = AccumulationMediator("mean", [DERIVED])
    with pytest.raises(CouplingError, match="no samples"):
        mediator.compute(np.datetime64("2024-01-03"))


def test_mediator_requires_no_initial_condition() -> None:
    mediator = AccumulationMediator("mean", [DERIVED])
    assert mediator.requires_ic is False
    mediator.initialize()
