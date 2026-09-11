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

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from earth2studio.models.da.healda import E2S_CHANNELS
from earth2studio.models.da.healda_v2 import (
    N_WINDOW,
    HealDAv2,
)
from earth2studio.models.da.healda_v2_utils import (
    CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL,
    PLATFORM_NAME_TO_ID,
    PRESSURE_LEVELS_HPA,
    SENSOR_OFFSET,
    PCACodec,
    build_conv_plevel_channel_stats,
    build_raw_to_local_lut,
    compute_unified_metadata,
    nearest_pressure_level_index,
)

# ---------- Constants ----------

NVAR = len(E2S_CHANNELS)  # 74
LEVEL_MODEL = 1
NPIX_MODEL = 12 * 4**LEVEL_MODEL  # 48
NPIX_OUT = 192
NLAT = 5
NLON = 10

REQUEST_TIME = np.array([np.datetime64("2024-01-01T12:00:00")])

# Expanded conv-plevel channel bookkeeping: vertical channel groups follow the
# base conv order with surface pressure removed:
#   gps_angle -> 0, gps_t -> 1, gps_q -> 2, q -> 3, t -> 4, u -> 5, v -> 6
T_GROUP = 4
U_GROUP = 5
CONV_PLEVEL_OFFSET = SENSOR_OFFSET["conv-plevel"]

# Synthetic level stats used for the t channel at 600 hPa
T_600_MEAN = 250.0
T_600_STD = 10.0


# ---------- Mock neural network / grid ----------


class PhooVideoHealDAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 2
        self.out_channels = NVAR
        self.npix = NPIX_OUT
        self.level_model = LEVEL_MODEL
        self.time_length = N_WINDOW
        self.last_call: dict = {}

    def forward(self, x, t, second_of_day, day_of_year, obs_ctx, class_labels=None):
        self.last_call = {
            "x": x,
            "t": t,
            "second_of_day": second_of_day,
            "day_of_year": day_of_year,
            "obs_ctx": obs_ctx,
        }
        return torch.zeros(
            x.shape[0], self.out_channels, self.time_length, self.npix, device=x.device
        )


class MockGrid:
    def ang2pix(self, lon, lat):
        return torch.zeros(lon.shape[0], dtype=torch.long, device=lon.device)


class MockRegridder:
    def __init__(self, nlat, nlon):
        self.nlat = nlat
        self.nlon = nlon

    def __call__(self, x):
        return torch.randn(
            *x.shape[:-1], self.nlat, self.nlon, dtype=x.dtype, device=x.device
        )


# ---------- Fixtures / builders ----------


def _build_channel_stats() -> pd.DataFrame:
    """Channel stats for atms, the PCA sensors, and conv-plevel."""
    conv_offset = SENSOR_OFFSET["conv"]
    base_conv = pd.DataFrame(
        {
            "Global_Channel_ID": np.arange(8) + conv_offset,
            "mean": [0.05, 250.0, 0.005, 950.0, 0.005, 260.0, 0.0, 0.0],
            "stddev": [0.02, 20.0, 0.002, 40.0, 0.002, 20.0, 5.0, 5.0],
        }
    )
    # min/max valid for the base conv channels (order matches CONV_CHANNELS)
    base_conv["min_valid"] = [0.0, 150.0, 0.0, 500.0, 0.0, 150.0, -100.0, -100.0]
    base_conv["max_valid"] = [0.1, 350.0, 1.0, 1100.0, 1.0, 350.0, 100.0, 100.0]

    level_stats = pd.DataFrame(
        {
            "Global_Channel_ID": [conv_offset + 5],  # base conv channel "t"
            "Level_hPa": [600],
            "obs_mean": [T_600_MEAN],
            "obs_std": [T_600_STD],
        }
    )
    plevel = build_conv_plevel_channel_stats(level_stats, base_conv)

    atms = pd.DataFrame(
        {
            "Global_Channel_ID": np.arange(22) + SENSOR_OFFSET["atms"],
            "mean": 250.0,
            "stddev": 50.0,
            "min_valid": 0.0,
            "max_valid": 400.0,
        }
    )
    pca_parts = [
        pd.DataFrame(
            {
                "Global_Channel_ID": np.arange(32) + SENSOR_OFFSET[sensor],
                "mean": 0.0,
                "stddev": 1.0,
                "min_valid": -np.inf,
                "max_valid": np.inf,
            }
        )
        for sensor in ("iasi-pca", "cris-fsr-pca", "airs-pca")
    ]
    return pd.concat([atms, *pca_parts, plevel], ignore_index=True)


def _build_codec() -> PCACodec:
    """Tiny synthetic AIRS codec: 3 channels -> 2 latents."""
    airs_offset = SENSOR_OFFSET["airs"]
    return PCACodec(
        bt_mean=torch.tensor([250.0, 260.0, 270.0]),
        bt_std=torch.tensor([10.0, 10.0, 10.0]),
        components=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        channel_gcids=torch.tensor([airs_offset, airs_offset + 1, airs_offset + 2]),
        sensor_chan=torch.tensor([1, 5, 9]),  # sparse GSI channel ids
    )


def _build_model(device="cpu", lat_lon=False):
    with patch("earth2studio.models.da.healda_v2.earth2grid") as mock_e2g:
        mock_e2g.healpix.Grid.return_value = MockGrid()
        mock_e2g.healpix.HEALPIX_PAD_XY = 0
        if lat_lon:
            mock_e2g.get_regridder.return_value = MockRegridder(NLAT, NLON)
        model = HealDAv2(
            model=PhooVideoHealDAModel(),
            condition=torch.zeros(1, 2, 1, NPIX_OUT),
            era5_mean=torch.zeros(1, NVAR, 1, 1),
            era5_std=torch.ones(1, NVAR, 1, 1),
            channel_stats=_build_channel_stats(),
            raw_to_local={"atms": build_raw_to_local_lut(np.arange(1, 23))},
            codecs={"airs-pca": _build_codec()},
            lat_lon=lat_lon,
            output_resolution=(NLAT, NLON),
        )
    model._grid = MockGrid()
    return model.to(device)


def _build_raw_conv_df(
    n_obs=10,
    request_time=None,
    variable="t",
    observation=260.0,
    obs_type=120,
    elev=100.0,
    pres_pa=50000.0,
):
    if request_time is None:
        request_time = REQUEST_TIME
    t = request_time[0].astype("datetime64[ns]")
    df = pd.DataFrame(
        {
            "time": np.full(n_obs, t),
            "lat": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "lon": np.random.uniform(0, 360, n_obs).astype(np.float32),
            "observation": np.full(n_obs, observation, dtype=np.float32),
            "variable": variable,
            "type": np.full(n_obs, obs_type, dtype=np.uint16),
            "elev": np.full(n_obs, elev, dtype=np.float32),
            "pres": np.full(n_obs, pres_pa, dtype=np.float32),
        }
    )
    df.attrs = {"request_time": request_time}
    return df


def _build_raw_sat_df(n_obs=10, request_time=None, sensor="atms", satellite="n20"):
    if request_time is None:
        request_time = REQUEST_TIME
    t = request_time[0].astype("datetime64[ns]")
    df = pd.DataFrame(
        {
            "time": np.full(n_obs, t),
            "lat": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "lon": np.random.uniform(0, 360, n_obs).astype(np.float32),
            "observation": np.random.uniform(200, 300, n_obs).astype(np.float32),
            "variable": sensor,
            "sensor_index": np.ones(n_obs, dtype=np.uint16),
            "satellite": satellite,
            "scan_angle": np.zeros(n_obs, dtype=np.float32),
            "satellite_za": np.full(n_obs, 30.0, dtype=np.float32),
            "solza": np.full(n_obs, 45.0, dtype=np.float32),
        }
    )
    df.attrs = {"request_time": request_time}
    return df


def _mock_forward(inputs):
    return torch.zeros(1, NVAR, N_WINDOW, NPIX_OUT)


# ---------- Utils tests ----------


def test_nearest_pressure_level_index():
    pressures = np.array([1000.0, 612.0, 25.0, 1200.0, 962.6, 962.4])
    idx = nearest_pressure_level_index(pressures)
    levels = PRESSURE_LEVELS_HPA[idx]
    # 612 -> 600; 25 clips to 50; 1200 clips to 1000; 962.5 is the 1000/925
    # midpoint: strictly above -> 1000, at/below -> 925
    assert list(levels) == [1000, 600, 50, 1000, 1000, 925]


def test_build_conv_plevel_channel_stats():
    stats = _build_channel_stats()
    plevel = stats[stats["Global_Channel_ID"] >= CONV_PLEVEL_OFFSET].reset_index(
        drop=True
    )
    assert len(plevel) == 92

    # t @ 600 hPa (group 4, level index 4) uses the level stats
    level_600 = int(np.where(PRESSURE_LEVELS_HPA == 600)[0][0])
    t_600 = plevel.iloc[T_GROUP * 13 + level_600]
    assert t_600["mean"] == T_600_MEAN
    assert t_600["stddev"] == T_600_STD
    # t @ 500 hPa has no level stats -> falls back to the base t normalization
    level_500 = int(np.where(PRESSURE_LEVELS_HPA == 500)[0][0])
    t_500 = plevel.iloc[T_GROUP * 13 + level_500]
    assert t_500["mean"] == 260.0
    assert t_500["stddev"] == 20.0
    # QC bounds inherited from the base channel
    assert t_600["min_valid"] == 150.0
    assert t_600["max_valid"] == 350.0
    # Surface pressure row uses the base ps normalization and bounds
    ps = plevel.iloc[CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL]
    assert ps["mean"] == 950.0
    assert ps["min_valid"] == 500.0


def test_compute_unified_metadata_branches():
    n = 2
    target = torch.full((n,), 1700000000, dtype=torch.int64)
    time_ns = target * 1_000_000_000
    lon = torch.tensor([0.0, 90.0])
    lat = torch.tensor([30.0, -45.0])
    # Row 0 is conventional (finite height), row 1 satellite (NaN height)
    height = torch.tensor([100.0, float("nan")])
    pressure = torch.tensor([500.0, float("nan")])
    scan = torch.tensor([float("nan"), 10.0])
    sat_za = torch.tensor([float("nan"), 30.0])
    sol_za = torch.tensor([float("nan"), 45.0])

    meta = compute_unified_metadata(
        target, time_ns, lon, lat, height, pressure, scan, sat_za, sol_za
    )
    assert meta.shape == (n, 50)
    # Shared latitude features
    assert torch.allclose(meta[0, 8], torch.sin(torch.deg2rad(lat[0])))
    assert torch.allclose(meta[1, 9], torch.cos(torch.deg2rad(lat[1])))
    # Zero relative time -> dt features are [0, 0, sin 0, cos 0]
    assert torch.allclose(meta[:, 4:7], torch.zeros(n, 3))
    assert torch.allclose(meta[:, 7], torch.ones(n))
    # Branch exclusivity: conv rows zero the sat block and vice versa
    assert torch.all(meta[0, 10:30] == 0)
    assert torch.any(meta[0, 30:50] != 0)
    assert torch.any(meta[1, 10:30] != 0)
    assert torch.all(meta[1, 30:50] == 0)


# ---------- Preprocessing tests ----------


def test_prep_conv_plevel_channel_and_normalization():
    model = _build_model()
    # t observation at 612 hPa -> binned to 600 hPa
    df = _build_raw_conv_df(n_obs=4, observation=265.0, pres_pa=61200.0)
    obs = model.filter_and_normalize(df, None, REQUEST_TIME[0])

    assert len(obs) == 4
    level_600 = int(np.where(PRESSURE_LEVELS_HPA == 600)[0][0])
    expected_gid = CONV_PLEVEL_OFFSET + T_GROUP * 13 + level_600
    assert (obs["global_channel"] == expected_gid).all()
    assert (obs["global_platform"] == PLATFORM_NAME_TO_ID["t"]).all()
    # z-scored with the 600 hPa level stats
    np.testing.assert_allclose(
        obs["observation"].to_numpy(), (265.0 - T_600_MEAN) / T_600_STD, rtol=1e-6
    )
    # Pressure converted Pa -> hPa for the metadata path
    np.testing.assert_allclose(obs["pressure"].to_numpy(), 612.0, rtol=1e-6)
    # Final frame of the window
    assert (obs["frame"] == N_WINDOW - 1).all()


def test_prep_conv_qc():
    model = _build_model()

    # NaN pressure -> dropped
    df = _build_raw_conv_df(n_obs=3, pres_pa=np.nan)
    assert len(model.prep_conv(df)) == 0

    # Non-GPS pressure floor is 1 hPa: 0.8 hPa dropped, GPS 0.8 hPa kept
    df = _build_raw_conv_df(n_obs=3, pres_pa=80.0)  # 0.8 hPa
    assert len(model.prep_conv(df)) == 0
    df = _build_raw_conv_df(n_obs=3, variable="gps", observation=0.05, pres_pa=80.0)
    assert len(model.prep_conv(df)) == 3

    # GPS level-2 retrievals dropped
    df = _build_raw_conv_df(n_obs=3, variable="gps_t", observation=250.0)
    assert len(model.prep_conv(df)) == 0

    # Satellite-derived UV dropped, in-situ UV kept
    df = _build_raw_conv_df(n_obs=3, variable="u", observation=10.0, obs_type=240)
    assert len(model.prep_conv(df)) == 0
    df = _build_raw_conv_df(n_obs=3, variable="u", observation=10.0, obs_type=220)
    assert len(model.prep_conv(df)) == 3

    # Height out of physical bounds dropped
    df = _build_raw_conv_df(n_obs=3, elev=70000.0)
    assert len(model.prep_conv(df)) == 0

    # Unknown variable raises
    df = _build_raw_conv_df(n_obs=3, variable="bogus")
    with pytest.raises(ValueError, match="Unknown conventional"):
        model.prep_conv(df)


def test_prep_conv_surface_pressure():
    model = _build_model()
    # Station pressure: observation in Pa -> hPa, valid range (500, 1100)
    df = _build_raw_conv_df(n_obs=2, variable="pres", observation=90000.0)
    obs = model.filter_and_normalize(df, None, REQUEST_TIME[0])
    assert len(obs) == 2
    expected_gid = CONV_PLEVEL_OFFSET + CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL
    assert (obs["global_channel"] == expected_gid).all()
    np.testing.assert_allclose(
        obs["observation"].to_numpy(), (900.0 - 950.0) / 40.0, rtol=1e-6
    )

    # 300 hPa station pressure violates the (500, 1100) valid range
    df = _build_raw_conv_df(n_obs=2, variable="pres", observation=30000.0)
    obs = model.filter_and_normalize(df, None, REQUEST_TIME[0])
    assert len(obs) == 0


def test_prep_mw():
    model = _build_model()
    df = _build_raw_sat_df(n_obs=5, sensor="atms")
    out = model.prep_mw(df, "atms")
    assert len(out) == 5
    # Raw channel 1 -> local 0 -> global SENSOR_OFFSET["atms"]
    assert (out["global_channel"] == SENSOR_OFFSET["atms"]).all()
    assert (out["global_platform"] == PLATFORM_NAME_TO_ID["n20"]).all()
    assert np.isnan(out["height"]).all()

    with pytest.raises(ValueError, match="Unknown satellite"):
        model.prep_mw(_build_raw_sat_df(n_obs=2, satellite="bogus"), "atms")


def test_prep_ir_pca():
    model = _build_model()
    codec = _build_codec()
    # Two footprints x three channels in long format (channels 1, 5, 9)
    t = REQUEST_TIME[0].astype("datetime64[ns]")
    rows = []
    for fp, (latv, lonv) in enumerate([(10.0, 20.0), (30.0, 40.0)]):
        for chan, bt in [(1, 260.0), (5, 270.0), (9, 280.0)]:
            rows.append(
                {
                    "time": t,
                    "lat": latv,
                    "lon": lonv,
                    "observation": bt,
                    "variable": "airs",
                    "sensor_index": chan,
                    "satellite": "aqua",
                    "scan_angle": 5.0,
                    "satellite_za": 20.0,
                    "solza": 60.0,
                }
            )
    df = pd.DataFrame(rows)

    out = model.prep_ir_pca(df, "airs-pca")
    # 2 footprints x 2 latents
    assert len(out) == 4
    assert list(out["global_channel"][:2]) == [
        SENSOR_OFFSET["airs-pca"],
        SENSOR_OFFSET["airs-pca"] + 1,
    ]
    assert (out["global_platform"] == PLATFORM_NAME_TO_ID["aqua"]).all()
    # Hand-computed latents: standardized bt = [1, 1, 1] @ components
    expected = codec.encode(torch.tensor([[260.0, 270.0, 280.0]])).numpy().ravel()
    np.testing.assert_allclose(out["observation"].to_numpy()[:2], expected, rtol=1e-6)
    # Footprint metadata repeated per latent
    np.testing.assert_allclose(out["lat"].to_numpy(), [10.0, 10.0, 30.0, 30.0])


def test_frame_bucketing():
    model = _build_model()
    analysis_time = REQUEST_TIME[0]
    # Frame g valid time = analysis - 6h * (7 - g); window is (valid-3h, valid+3h]
    frame3_valid = pd.Timestamp(analysis_time) - pd.Timedelta(hours=6 * 4)

    df = _build_raw_conv_df(n_obs=4)
    df["time"] = np.array(
        [
            frame3_valid + pd.Timedelta(hours=3),  # inclusive end -> frame 3
            frame3_valid - pd.Timedelta(hours=3),  # exclusive start -> frame 2
            frame3_valid,
            pd.Timestamp(analysis_time) + pd.Timedelta(hours=4),  # outside window
        ],
        dtype="datetime64[ns]",
    )
    obs = model.filter_and_normalize(df, None, analysis_time)
    frames = obs.sort_values("obs_time_ns")["frame"].to_list()
    assert len(obs) == 3
    assert frames == [2, 3, 3]
    # Each frame's target_sec matches its valid time
    target = obs[obs["frame"] == 3]["target_sec"].unique()
    assert len(target) == 1
    assert target[0] == int(frame3_valid.timestamp())


# ---------- build_input / call tests ----------


def test_build_input_obs_ctx():
    model = _build_model()
    conv_df = _build_raw_conv_df(15)
    sat_df = _build_raw_sat_df(10)
    obs = model.filter_and_normalize(conv_df, sat_df, REQUEST_TIME[0])
    inputs = model.build_input(obs, REQUEST_TIME[0])

    obs_ctx = inputs["obs_ctx"]
    total_pixels = N_WINDOW * NPIX_MODEL
    assert obs_ctx.cu_seqlens_k.numel() == total_pixels + 1
    assert int(obs_ctx.cu_seqlens_k[-1]) == len(obs)
    assert obs_ctx.obs.shape[0] == len(obs)
    assert obs_ctx.float_metadata.shape == (len(obs), 50)
    assert inputs["second_of_day"].shape == (1, N_WINDOW)
    assert inputs["day_of_year"].shape == (1, N_WINDOW)
    assert inputs["condition"].shape == (1, 2, N_WINDOW, NPIX_OUT)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
@pytest.mark.parametrize("lat_lon", [False, True])
def test_healda_v2_call(device, lat_lon):
    model = _build_model(device=device, lat_lon=lat_lon)
    conv_df = _build_raw_conv_df(15)
    sat_df = _build_raw_sat_df(10)

    out = model(conv_df, sat_df)

    assert isinstance(out, xr.DataArray)
    if lat_lon:
        assert out.dims == ("time", "lead_time", "variable", "lat", "lon")
        assert out.shape == (1, N_WINDOW, NVAR, NLAT, NLON)
    else:
        assert out.dims == ("time", "lead_time", "variable", "npix")
        assert out.shape == (1, N_WINDOW, NVAR, NPIX_OUT)
    assert np.all(out.coords["time"].values == REQUEST_TIME)
    # Analysis frame is the final lead_time == 0
    assert out.coords["lead_time"].values[-1] == np.timedelta64(0, "ns")
    assert out.coords["lead_time"].values[0] == np.timedelta64(-42, "h").astype(
        "timedelta64[ns]"
    )


def test_healda_v2_call_missing_request_time():
    model = _build_model()
    df = _build_raw_conv_df(5)
    df.attrs = {}
    with pytest.raises(ValueError, match="request_time"):
        model(df)


def test_healda_v2_call_multiple_request_times():
    model = _build_model()
    request_time = np.array(
        [np.datetime64("2024-01-01T12:00:00"), np.datetime64("2024-01-01T18:00:00")]
    )
    df = _build_raw_conv_df(5, request_time=request_time)
    with pytest.raises(ValueError, match="single analysis time"):
        model(df)


def test_healda_v2_call_no_obs():
    model = _build_model()
    with pytest.raises(ValueError, match="At least one"):
        model(None, None)


def test_healda_v2_call_empty_obs():
    model = _build_model()
    df = _build_raw_conv_df(5, observation=9999.0)  # violates t valid range
    out = model(df)
    assert isinstance(out, xr.DataArray)
    assert out.shape == (1, N_WINDOW, NVAR, NPIX_OUT)
    assert np.all(np.isnan(out.values))


def test_healda_v2_generator():
    model = _build_model()
    conv_df = _build_raw_conv_df(10)
    sat_df = _build_raw_sat_df(10)

    gen = model.create_generator()
    assert gen.send(None) is None

    with patch.object(model, "_forward", _mock_forward):
        da = gen.send((conv_df, None))
        assert da.shape == (1, N_WINDOW, NVAR, NPIX_OUT)
        da = gen.send((None, sat_df))
        assert da.shape == (1, N_WINDOW, NVAR, NPIX_OUT)
        da = gen.send((conv_df, sat_df))
        assert da.shape == (1, N_WINDOW, NVAR, NPIX_OUT)

    with pytest.raises(ValueError, match="At least one"):
        gen.send((None, None))
    gen.close()


# ---------- Coords tests ----------


def test_healda_v2_init_coords():
    model = _build_model()
    assert model.init_coords() is None


def test_healda_v2_input_coords():
    model = _build_model()
    conv_schema, sat_schema = model.input_coords()
    for field in ("time", "lat", "lon", "observation", "variable", "elev", "pres"):
        assert field in conv_schema
    for field in ("time", "lat", "lon", "observation", "sensor_index", "satellite"):
        assert field in sat_schema
    # IR sensors are part of the sat schema variables
    sat_vars = set(sat_schema["variable"])
    assert {"iasi", "crisfsr", "airs"} <= sat_vars


def test_healda_v2_output_coords():
    model = _build_model()
    (coords,) = model.output_coords(model.input_coords(), request_time=REQUEST_TIME)
    assert list(coords.keys()) == ["time", "lead_time", "variable", "npix"]
    assert len(coords["lead_time"]) == N_WINDOW
    assert len(coords["variable"]) == NVAR
    assert len(coords["npix"]) == NPIX_OUT


def test_healda_v2_default_package_unpublished():
    with pytest.raises(NotImplementedError, match="not been published"):
        HealDAv2.load_default_package()
