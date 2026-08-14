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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import CallableComponent
from earth2studio.nvcoupler.connector import Connector
from earth2studio.nvcoupler.dictionary import (
    DEFAULT_DICTIONARY,
    FieldDictionary,
    FieldEntry,
)
from earth2studio.nvcoupler.errors import (
    CouplingError,
    IncompatibleFieldError,
    VerticalMismatchError,
)
from earth2studio.nvcoupler.testing import (
    atmos_ic,
    fake_atmos,
    fake_ocean,
    ocean_ic,
)
from earth2studio.nvcoupler.vertical import HybridLevels, PressureLevels

T0 = np.datetime64("2024-01-01")


@pytest.fixture
def caplog(caplog):
    """Extend caplog to also capture loguru log records (the connector logs
    via loguru, which does not propagate to stdlib logging by default)."""
    from loguru import logger as loguru_logger

    handler_id = loguru_logger.add(caplog.handler, format="{message}", level=0)
    yield caplog
    loguru_logger.remove(handler_id)


def _realized_pair(with_mask=False):
    atmos, ocean = fake_atmos(), fake_ocean(with_mask=with_mask)
    clock = Clock(T0, "2024-01-05", "6h")
    atmos.realize(clock)
    ocean.realize(clock)
    atmos.initialize(*atmos_ic())
    ocean.initialize(*ocean_ic(sst0=3.0))
    return atmos, ocean, clock


def test_match_and_units():
    atmos, ocean, _ = _realized_pair()
    conn = Connector(ocean, atmos)
    assert conn.match() == ["sea_surface_temperature"]
    # no overlap the other way (atmos exports z1000, ocean imports 48h mean)
    with pytest.raises(IncompatibleFieldError, match="no fields match"):
        Connector(atmos, ocean).match()
    # explicit fields not present on both sides
    with pytest.raises(IncompatibleFieldError, match="not in both"):
        Connector(ocean, atmos, fields=["air_temperature_2m"]).match()


def test_execute_regrids_to_destination_grid():
    atmos, ocean, _ = _realized_pair()
    Connector(ocean, atmos).execute(T0)
    sst = atmos.import_state["sea_surface_temperature"]
    assert sst.data.shape == (32, 64)  # ocean 16x32 -> atmos grid
    assert torch.allclose(
        sst.data, torch.full((32, 64), 3.0)
    )  # constant stays constant
    assert sst.source == "ocean"
    assert list(sst.coords) == ["lat", "lon"]
    assert np.array_equal(sst.coords["lat"], atmos.grid_coords()["lat"])


def test_identity_when_grids_match():
    a1, a2 = fake_atmos(), fake_atmos()
    a2.name = "atmos2"
    a2.import_names = ["geopotential_at_1000hpa"]
    a2.export_names = ["sea_surface_temperature"]
    clock = Clock(T0, "2024-01-02", "6h")
    for c in (a1, a2):
        c.realize(clock)
        c.initialize(*atmos_ic())
    conn = Connector(a1, a2)  # a1 exports z1000, a2 imports it; same grid
    conn.execute(T0)
    z = a2.import_state["geopotential_at_1000hpa"]
    assert torch.equal(z.data, a1.export_state["geopotential_at_1000hpa"].data)
    assert conn._regridders == {}  # identity fast path built no regridder


def test_regrid_matches_direct_kernel_call():
    from earth2studio.utils.interp import latlon_interpolation_regular

    atmos, ocean, _ = _realized_pair()
    # put a non-constant field in the ocean export
    lat = torch.as_tensor(ocean.grid_coords()["lat"])
    lon = torch.as_tensor(ocean.grid_coords()["lon"])
    data = lat.view(-1, 1) + 0.1 * lon.view(1, -1)
    f = ocean.export_state["sea_surface_temperature"]
    f.data = data.to(torch.float32)
    Connector(ocean, atmos).execute(T0)
    got = atmos.import_state["sea_surface_temperature"].data

    lat1, lon1 = np.meshgrid(
        atmos.grid_coords()["lat"], atmos.grid_coords()["lon"], indexing="ij"
    )
    expected = latlon_interpolation_regular(
        torch.flip(data.to(torch.float32), dims=(-2,)),  # ascending lat
        torch.as_tensor(np.asarray(ocean.grid_coords()["lat"])[::-1].copy()).float(),
        torch.as_tensor(ocean.grid_coords()["lon"]).float(),
        torch.as_tensor(lat1).float(),
        torch.as_tensor(lon1).float(),
    )
    assert torch.allclose(got, expected)


def test_mask_fill_nearest_and_zero():
    atmos, ocean, _ = _realized_pair(with_mask=True)
    # poison the invalid (northern land) half; valid half stays 3.0
    f = ocean.export_state["sea_surface_temperature"]
    f.data = f.data.clone()
    f.data[:8, :] = 999.0
    Connector(ocean, atmos, fill="nearest").execute(T0)
    sst = atmos.import_state["sea_surface_temperature"]
    assert torch.all(sst.data < 4.0)  # no 999 leaked through regrid
    assert torch.allclose(sst.data, torch.full((32, 64), 3.0))
    assert sst.mask is None  # consumed

    zero_conn = Connector(ocean, atmos, fill="zero")
    zero_conn.execute(T0)
    sst0 = atmos.import_state["sea_surface_temperature"]
    assert sst0.data.min() == 0.0  # land became zero (then regridded)


def test_time_policy_linear_extrapolates():
    atmos, ocean, clock = _realized_pair()
    conn = Connector(ocean, atmos, time_policy="linear")
    f0 = ocean.export_state["sea_surface_temperature"]
    conn.execute(T0)  # first transfer: only one export, constant fallback
    assert torch.allclose(
        atmos.import_state["sea_surface_temperature"].data,
        torch.full((32, 64), 3.0),
    )
    # ocean produces a new export at +48h with value 5.0
    f1 = f0.clone()
    f1.data.fill_(5.0)
    f1.valid_time = T0 + np.timedelta64(48, "h")
    ocean.export_state.add(f1)
    # driver asks at +72h: linear extrapolation = 5 + (5-3) * 24/48 = 6
    conn.execute(T0 + np.timedelta64(72, "h"))
    got = atmos.import_state["sea_surface_temperature"]
    assert torch.allclose(got.data, torch.full((32, 64), 6.0))
    # constant policy would have given 5.0
    const = Connector(ocean, atmos, time_policy="constant")
    const.execute(T0 + np.timedelta64(72, "h"))
    assert torch.allclose(
        atmos.import_state["sea_surface_temperature"].data,
        torch.full((32, 64), 5.0),
    )


def test_time_policy_linear_holds_slope_across_repeated_executes():
    """Regression: repeated executes between source updates must keep
    extrapolating along the (prev, latest) slope instead of silently
    degrading to constant after the first extrapolated step."""
    atmos, ocean, _ = _realized_pair()
    conn = Connector(ocean, atmos, time_policy="linear")
    f0 = ocean.export_state["sea_surface_temperature"]
    f0.data = f0.data.clone().fill_(1.0)
    conn.execute(T0)  # seed history with 1.0 @ t0
    f1 = f0.clone()
    f1.data.fill_(2.0)
    f1.valid_time = T0 + np.timedelta64(48, "h")
    ocean.export_state.add(f1)
    # slope is (2-1)/48h; every 6h step past 48h adds 0.125
    for hours, expected in [(54, 2.125), (60, 2.25), (66, 2.375), (72, 2.5)]:
        conn.execute(T0 + np.timedelta64(hours, "h"))
        got = atmos.import_state["sea_surface_temperature"]
        assert torch.allclose(
            got.data, torch.full((32, 64), expected)
        ), f"at +{hours}h expected {expected}, got {got.data.flatten()[0]}"


def test_time_policy_linear_falls_back_for_lead_time_fields(caplog):
    from earth2studio.nvcoupler.field import Field

    atmos, ocean, _ = _realized_pair()
    conn = Connector(ocean, atmos, time_policy="linear")

    def lead_field(value, hours):
        return Field(
            torch.full((2, 16, 32), value),
            OrderedDict(
                {
                    "lead_time": np.array(
                        [np.timedelta64(48, "h"), np.timedelta64(96, "h")]
                    ),
                    **{k: v for k, v in ocean.grid_coords().items()},
                }
            ),
            "sea_surface_temperature",
            "K",
            valid_time=T0 + np.timedelta64(hours, "h"),
        )

    ocean.export_state.add(lead_field(3.0, 0))
    conn.execute(T0)
    ocean.export_state.add(lead_field(5.0, 48))
    conn.execute(T0 + np.timedelta64(72, "h"))
    # linear would extrapolate to 6.0; lead_time fields must hold constant
    got = atmos.import_state["sea_surface_temperature"]
    assert torch.allclose(got.data, torch.full((2, 32, 64), 5.0))
    assert any("undefined" in r.message for r in caplog.records)


def test_probe_last_transfer():
    atmos, ocean, _ = _realized_pair()
    conn = Connector(ocean, atmos)
    conn.execute(T0)
    assert "sea_surface_temperature" in conn.last_transfer
    assert conn.name == "ocean->atmos"


def test_unproduced_export_raises():
    atmos, ocean, _ = _realized_pair()
    del ocean.export_state["sea_surface_temperature"]
    with pytest.raises(CouplingError, match="has not produced"):
        Connector(ocean, atmos).execute(T0)


# ---------------------------------------------------------------------------
# Vertical stage through the connector (chemistry-style coupling)
# ---------------------------------------------------------------------------
def _chem_dictionary():
    d = FieldDictionary(DEFAULT_DICTIONARY)
    d.register(FieldEntry("ozone_mixing_ratio", "kg kg-1", aliases=frozenset({"o3"})))
    return d


def _vertical_pair(export_ps=True):
    """met (hybrid levels) -> chem (pressure levels) toy pair."""
    d = _chem_dictionary()
    hybrid = HybridLevels((30000.0, 20000.0, 0.0), (0.0, 0.5, 1.0))
    pressure = PressureLevels((500.0, 850.0))
    nlat, nlon = 4, 8

    def met_step(x, coords):
        return x, coords

    met = CallableComponent(
        "met",
        met_step,
        "6h",
        exports=["ozone_mixing_ratio"],
        dictionary=d,
        export_vertical={"ozone_mixing_ratio": hybrid},
    )
    chem = CallableComponent(
        "chem",
        met_step,
        "6h",
        imports=["ozone_mixing_ratio"],
        dictionary=d,
        import_vertical={"ozone_mixing_ratio": pressure},
    )
    clock = Clock(T0, "2024-01-02", "6h")
    met.realize(clock)
    chem.realize(clock)

    grid = OrderedDict(
        {
            "lat": np.linspace(90, -90, nlat),
            "lon": np.linspace(0, 360, nlon, endpoint=False),
        }
    )
    ps_value = 100000.0
    p_src = np.array([30000.0, 70000.0, 100000.0])
    o3 = (
        torch.tensor(np.log(p_src), dtype=torch.float64)
        .view(1, 3, 1, 1)
        .expand(1, 3, nlat, nlon)
        .clone()
    )
    # sp has no level dim so it cannot share o3's tensor; it is hand-added to
    # the export state below (the connector reads export_state directly)
    met_coords = OrderedDict(
        {"variable": np.array(["o3"]), "level": np.arange(3.0), **grid}
    )
    met.initialize(o3, met_coords)
    if export_ps:
        from earth2studio.nvcoupler.field import Field

        met.export_state.add(
            Field(
                torch.full((nlat, nlon), ps_value, dtype=torch.float64),
                OrderedDict(grid),
                "surface_pressure",
                "Pa",
                valid_time=T0,
                source="met",
            )
        )
    chem_coords = OrderedDict(
        {"variable": np.array(["o3"]), "level": np.array([500.0, 850.0]), **grid}
    )
    chem.initialize(torch.zeros(1, 2, nlat, nlon, dtype=torch.float64), chem_coords)
    return met, chem


def test_vertical_stage_hybrid_to_pressure():
    met, chem = _vertical_pair(export_ps=True)
    Connector(met, chem, fields=["ozone_mixing_ratio"]).execute(T0)
    o3 = chem.import_state["ozone_mixing_ratio"]
    assert list(o3.coords["level"]) == [500.0, 850.0]
    expected = np.log(np.array([50000.0, 85000.0]))
    assert o3.data.shape == (2, 4, 8)  # (level, lat, lon) after variable split
    assert np.allclose(o3.data[:, 0, 0].numpy(), expected)
    assert o3.vertical == PressureLevels((500.0, 850.0))


def test_vertical_missing_ps_raises():
    met, chem = _vertical_pair(export_ps=False)
    with pytest.raises(VerticalMismatchError, match="surface_pressure"):
        Connector(met, chem, fields=["ozone_mixing_ratio"]).execute(T0)


# ---------------------------------------------------------------------------
# HEALPix-style face grids: identity when identical, error when different
# ---------------------------------------------------------------------------
def _hpx_pair(dst_nside=4, src_nside=4):
    """src/dst CallableComponents on (face, height, width) HEALPix-style
    grids; dst_nside controls whether the grids match (4) or differ."""

    def step(x, coords):
        return x, coords

    src = CallableComponent("hpx_src", step, "6h", exports=["sea_surface_temperature"])
    dst = CallableComponent(
        "hpx_dst",
        step,
        "6h",
        imports=["sea_surface_temperature"],
        exports=["geopotential_at_1000hpa"],
    )
    clock = Clock(T0, "2024-01-02", "6h")

    def hpx_coords(nside):
        return OrderedDict(
            {
                "variable": np.array(["sst"]),
                "face": np.arange(12),
                "height": np.arange(nside),
                "width": np.arange(nside),
            }
        )

    src.realize(clock)
    src.initialize(
        torch.full((1, 12, src_nside, src_nside), 3.0), hpx_coords(src_nside)
    )
    dst.realize(clock)
    dst_coords = hpx_coords(dst_nside)
    dst_coords["variable"] = np.array(["z1000"])
    dst.initialize(torch.zeros(1, 12, dst_nside, dst_nside), dst_coords)
    return src, dst


def test_identical_face_grids_pass_through_identity():
    src, dst = _hpx_pair(dst_nside=4)
    conn = Connector(src, dst)
    conn.execute(T0)
    got = dst.import_state["sea_surface_temperature"]
    assert torch.equal(got.data, src.export_state["sea_surface_temperature"].data)
    assert list(got.coords) == ["face", "height", "width"]
    assert torch.allclose(got.data, torch.full((12, 4, 4), 3.0))
    assert conn._regridders == {}  # identity path built no regridder


def test_differing_face_grids_without_regridder_raise():
    src, dst = _hpx_pair(dst_nside=8)
    with pytest.raises(IncompatibleFieldError, match="regridder"):
        Connector(src, dst).execute(T0)


def test_user_regridder_on_differing_face_grids():
    """Regression: a user-supplied regridder= must work for non-latlon
    (face, height, width) grids and rebuild coords from the destination."""
    src, dst = _hpx_pair(dst_nside=4, src_nside=8)

    def pool(x: torch.Tensor) -> torch.Tensor:
        # mean-pool the trailing (height, width) dims 8x8 -> 4x4
        return x.reshape(*x.shape[:-2], 4, 2, 4, 2).mean(dim=(-3, -1))

    conn = Connector(src, dst, regridder=pool)
    conn.execute(T0)
    got = dst.import_state["sea_surface_temperature"]
    assert got.data.shape == (12, 4, 4)
    assert torch.allclose(got.data, torch.full((12, 4, 4), 3.0))
    # coords carry the destination's face/height/width grid
    assert list(got.coords) == ["face", "height", "width"]
    dst_grid = dst.grid_coords()
    for k in ("face", "height", "width"):
        assert np.array_equal(got.coords[k], np.asarray(dst_grid[k]))
