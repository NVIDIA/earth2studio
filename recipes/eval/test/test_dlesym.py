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

"""Tests for DLESyMPipeline.

Uses a stub model that mimics DLESyM's shape without loading a real
checkpoint.  Covers:

* :func:`_unique_forecast_valid_times` — the flattened-lead-time math
  that drives DLESyM's verification store.
* ``DLESyMPipeline.run_item`` — confirms step 0 (IC window) is skipped
  and the caller sees exactly ``nsteps`` yields.
* ``DLESyMPipeline._mask_invalid_ocean`` — confirms ocean-variable
  values at non-valid lead times are NaN'd while atmos values stay
  untouched.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pytest
import torch
from src.pipeline import DLESyMPipeline, _unique_forecast_valid_times
from src.work import WorkItem

from earth2studio.utils.coords import CoordSystem


# ---------------------------------------------------------------------------
# _unique_forecast_valid_times
# ---------------------------------------------------------------------------


def _hours(*vals: int) -> np.ndarray:
    return np.array([np.timedelta64(v, "h") for v in vals]).astype("timedelta64[ns]")


class TestUniqueForecastValidTimes:
    def test_single_step_dlesym_like(self):
        # Simulates DLESyM: 16 output lead times [6h..96h] per iterator step
        step_lead = _hours(*range(6, 97, 6))  # 6, 12, …, 96
        ic = np.datetime64("2024-01-01T00")
        result = _unique_forecast_valid_times([ic], step_lead, nsteps=1)
        # Expect: [0, 6, 12, …, 96] = 17 ticks
        assert len(result) == 17
        assert result[0] == ic
        assert result[-1] == ic + np.timedelta64(96, "h")

    def test_two_steps_dlesym_like(self):
        step_lead = _hours(*range(6, 97, 6))
        ic = np.datetime64("2024-01-01T00")
        result = _unique_forecast_valid_times([ic], step_lead, nsteps=2)
        # Expect 2*16 + 1 = 33 ticks
        assert len(result) == 33
        assert result[-1] == ic + np.timedelta64(192, "h")

    def test_single_lead_time_per_step(self):
        # Degenerates to compute_verification_times-style output for DLWP-style models
        step_lead = _hours(6)
        ic = np.datetime64("2024-01-01T00")
        result = _unique_forecast_valid_times([ic], step_lead, nsteps=3)
        # Expect [0, 6, 12, 18] = 4 ticks
        assert len(result) == 4
        assert result == [ic + _hours(h)[0] for h in (0, 6, 12, 18)]

    def test_deduplicates_across_ics(self):
        step_lead = _hours(12, 24)
        # Two ICs 24h apart — output ticks overlap at 24h from first / 0h from second.
        ic1 = np.datetime64("2024-01-01T00")
        ic2 = np.datetime64("2024-01-02T00")
        result = _unique_forecast_valid_times([ic1, ic2], step_lead, nsteps=1)
        # From ic1: [0, 12, 24].  From ic2: [0, 12, 24] shifted → [24, 36, 48].
        # Union: {0, 12, 24, 36, 48}.
        expected = [ic1 + _hours(h)[0] for h in (0, 12, 24, 36, 48)]
        assert result == expected


# ---------------------------------------------------------------------------
# Stub model mimicking the DLESyM surface area needed by the pipeline
# ---------------------------------------------------------------------------


class _StubDLESyM:
    """Minimal duck-typed stand-in for DLESyMLatLon.

    Reproduces the shape contract relied on by DLESyMPipeline:

    * ``input_coords()`` — 9 input lead_times, base variables, lat/lon grid.
    * ``output_coords(input_coords)`` — 16 output lead_times, base + derived
      variables, same spatial dims.
    * ``create_iterator(x, coords)`` — yields the IC first (negative lead
      times), then ``nmax`` forward steps with shifted absolute lead times.
    * ``ocean_variables`` — list of variables for which only a subset of
      output lead times is valid.
    * ``retrieve_valid_ocean_outputs`` — returns the ocean-valid subset.
    """

    atmos_variables = ["t2m", "z500"]
    ocean_variables = ["sst"]

    # 4 lat × 8 lon grid so dims are explicit but cheap.
    _lat = np.linspace(90, -90, 4)
    _lon = np.linspace(0, 360, 8, endpoint=False)

    # Input window: [-18h, -12h, -6h, 0h] is close to the real atmosphere
    # lead times.  Using 4 (not 9) keeps tensors small — the code under
    # test is independent of the exact count.
    _input_lead = _hours(-18, -12, -6, 0)
    # Output window per step: [6h, 12h, ..., 96h] — 16 values at 6h stride.
    _output_lead = _hours(*range(6, 97, 6))
    # Ocean is only valid at 48h and 96h within each step (pattern: mod 48).
    _ocean_lead = _hours(48, 96)

    def __init__(self) -> None:
        self._all_vars = self.atmos_variables + self.ocean_variables

    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            [
                ("batch", np.empty(0)),
                ("time", np.empty(0)),
                ("lead_time", self._input_lead),
                ("variable", np.array(self._all_vars)),
                ("lat", self._lat),
                ("lon", self._lon),
            ]
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        # Shift lead_time by the last input lead_time, exactly like DLESyM does.
        anchor = input_coords["lead_time"][-1]
        return OrderedDict(
            [
                ("batch", input_coords["batch"]),
                ("time", input_coords["time"]),
                ("lead_time", self._output_lead + anchor),
                ("variable", np.array(self._all_vars)),
                ("lat", self._lat),
                ("lon", self._lon),
            ]
        )

    def retrieve_valid_ocean_outputs(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        valid_lt = np.array(
            [lt for lt in coords["lead_time"] if lt % self._ocean_lead[0] == 0]
        )
        out_coords = coords.copy()
        out_coords["lead_time"] = valid_lt
        out_coords["variable"] = np.array(self.ocean_variables)
        # The tensor value isn't inspected by the pipeline — just return something.
        return x[..., :1, :, :], out_coords

    def create_iterator(self, x: torch.Tensor, coords: CoordSystem):
        """Yields IC first, then ``inf`` forward steps.

        Output tensor dimensionality mirrors the caller-provided ``x``:
        after ``fetch_data`` + ``map_coords`` the pipeline may have 5
        dims (no batch) or 6 dims (batch), depending on how the source
        normalizes.  Using ``x.shape`` as the template keeps the stub
        matched to whatever the pipeline actually passes in.
        """
        # Step 0: IC window, with input lead_times.  Pass through exactly
        # what the caller gave us.
        yield x, coords

        # Subsequent steps: shift anchor forward by 96h each time and
        # resize the lead_time / variable axes to the output shape.
        stride = self._output_lead[-1]
        anchor = coords["lead_time"][-1]
        ordered_keys = list(coords.keys())
        lt_axis = ordered_keys.index("lead_time")
        var_axis = ordered_keys.index("variable")
        step = 0
        while True:
            step += 1
            new_coords = coords.copy()
            new_coords["lead_time"] = self._output_lead + anchor + stride * (step - 1)
            new_coords["variable"] = np.array(self._all_vars)

            new_shape = list(x.shape)
            new_shape[lt_axis] = len(new_coords["lead_time"])
            new_shape[var_axis] = len(new_coords["variable"])

            t = torch.arange(float(np.prod(new_shape))).reshape(new_shape)
            yield t, new_coords

    # DLESyMPipeline never mutates the model, so to() is a no-op.
    def to(self, device: Any) -> "_StubDLESyM":
        return self


def _make_stub_pipeline() -> DLESyMPipeline:
    """Build a DLESyMPipeline instance wired to the stub model."""
    pipeline = DLESyMPipeline()
    model = _StubDLESyM()
    pipeline.prognostic = model
    pipeline.diagnostics = []
    pipeline.perturbation = None
    pipeline.nsteps = 3
    pipeline._prognostic_ic = model.input_coords()
    pipeline._spatial_ref = model.output_coords(pipeline._prognostic_ic)
    pipeline._dx_input_coords = {}
    pipeline._ocean_variables = list(model.ocean_variables)
    return pipeline


# ---------------------------------------------------------------------------
# run_item — step-0 skip + yield count
# ---------------------------------------------------------------------------


class TestDLESyMRunItem:
    @pytest.fixture()
    def pipeline(self):
        return _make_stub_pipeline()

    @pytest.fixture()
    def fake_source(self, pipeline):
        """DataSource stub that returns zeros of the right shape for fetch_data."""
        import xarray as xr

        class _Src:
            def __call__(self, time, variable):
                # fetch_data handles normalization into (time, variable, ...).
                # Return an xr.DataArray with the expected shape and coords.
                n_t = len(time) if isinstance(time, (list, np.ndarray)) else 1
                n_v = len(variable) if isinstance(variable, (list, np.ndarray)) else 1
                data = np.zeros(
                    (n_t, n_v, 4, 8),
                    dtype="float32",
                )
                return xr.DataArray(
                    data,
                    dims=("time", "variable", "lat", "lon"),
                    coords={
                        "time": np.atleast_1d(time),
                        "variable": np.atleast_1d(variable),
                        "lat": pipeline._prognostic_ic["lat"],
                        "lon": pipeline._prognostic_ic["lon"],
                    },
                )

            async def fetch(self, time, variable):
                return self(time, variable)

        return _Src()

    def test_yields_nsteps_outputs_after_ic_skip(self, pipeline, fake_source):
        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0)
        outputs = list(pipeline.run_item(item, fake_source, torch.device("cpu")))
        assert len(outputs) == pipeline.nsteps, (
            f"DLESyM should yield exactly nsteps ({pipeline.nsteps}) outputs; "
            f"IC yield must be skipped"
        )

    def test_yielded_lead_times_cover_forecast_window(self, pipeline, fake_source):
        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0)
        outputs = list(pipeline.run_item(item, fake_source, torch.device("cpu")))

        for step_idx, (_, coords) in enumerate(outputs):
            lead_times = coords["lead_time"]
            # Each yield should have 16 lead times, in the range
            # [step * 96 + 6, (step + 1) * 96].
            assert len(lead_times) == 16
            expected_first = np.timedelta64(step_idx * 96 + 6, "h").astype(
                "timedelta64[ns]"
            )
            expected_last = np.timedelta64((step_idx + 1) * 96, "h").astype(
                "timedelta64[ns]"
            )
            assert lead_times[0] == expected_first
            assert lead_times[-1] == expected_last


# ---------------------------------------------------------------------------
# _mask_invalid_ocean
# ---------------------------------------------------------------------------


class TestMaskInvalidOcean:
    def test_masks_ocean_at_invalid_lead_times(self):
        pipeline = _make_stub_pipeline()

        # Build a (batch, time, lead, var, lat, lon) tensor of ones so any
        # NaN clearly came from masking.
        coords = pipeline._spatial_ref.copy()
        coords["lead_time"] = coords["lead_time"]  # 16 output lead times
        x = torch.ones(
            1,
            1,
            len(coords["lead_time"]),
            len(coords["variable"]),
            len(coords["lat"]),
            len(coords["lon"]),
        )

        masked = pipeline._mask_invalid_ocean(x, coords)

        var_idx = list(coords.keys()).index("variable")
        lead_idx = list(coords.keys()).index("lead_time")

        var_names = list(coords["variable"])
        sst_v = var_names.index("sst")

        valid_lt_set = {
            np.timedelta64(48, "h").astype("timedelta64[ns]"),
            np.timedelta64(96, "h").astype("timedelta64[ns]"),
        }
        for i, lt in enumerate(coords["lead_time"]):
            slicer = [slice(None)] * x.ndim
            slicer[lead_idx] = i
            slicer[var_idx] = sst_v
            slice_vals = masked[tuple(slicer)]
            if lt in valid_lt_set:
                assert torch.all(slice_vals == 1.0), f"sst at {lt} should stay valid"
            else:
                assert torch.all(torch.isnan(slice_vals)), f"sst at {lt} should be NaN"

    def test_atmos_variables_untouched(self):
        pipeline = _make_stub_pipeline()

        coords = pipeline._spatial_ref.copy()
        x = torch.ones(
            1,
            1,
            len(coords["lead_time"]),
            len(coords["variable"]),
            len(coords["lat"]),
            len(coords["lon"]),
        )

        masked = pipeline._mask_invalid_ocean(x, coords)

        var_idx = list(coords.keys()).index("variable")
        var_names = list(coords["variable"])
        for v in ("t2m", "z500"):
            vi = var_names.index(v)
            slicer = [slice(None)] * x.ndim
            slicer[var_idx] = vi
            assert not torch.any(torch.isnan(masked[tuple(slicer)])), (
                f"atmos variable {v} must not be masked"
            )

    def test_no_ocean_variables_is_noop(self):
        pipeline = _make_stub_pipeline()
        pipeline._ocean_variables = []  # simulate an atmos-only variant

        coords = pipeline._spatial_ref.copy()
        x = torch.ones(
            1,
            1,
            len(coords["lead_time"]),
            len(coords["variable"]),
            len(coords["lat"]),
            len(coords["lon"]),
        )
        out = pipeline._mask_invalid_ocean(x, coords)
        assert torch.equal(out, x)
