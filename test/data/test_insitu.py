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

"""Offline tests for the insitubatch initial-condition / verification feed.

The feed takes an injected zarr ``Store``, so the whole surface is exercised against a
small synthetic store on disk -- no network, no live bucket. The fixture mirrors the
public WB2 / ARCO layout the adapter targets: a fat time-chunk (several sample-axis steps
per stored chunk) plus 1-D ``time`` (CF-encoded) / ``latitude`` / ``longitude`` coordinate
arrays.
"""

from importlib.metadata import version

import numpy as np
import pytest

pytest.importorskip("insitubatch", reason="insitubatch is an optional dependency")

from insitubatch import ensure_local_dir, obstore_store  # noqa: E402

from earth2studio.data.insitu import (  # noqa: E402
    InSituForecastFeed,
    decode_cf_time,
)

pytestmark = pytest.mark.skipif(
    int(version("zarr").split(".")[0]) < 3, reason="Requires zarr v3"
)

TIME_UNITS = "hours since 1959-01-01"
STEP_H = 6  # store sample-axis step (6-hourly, like WB2)


def write_store(tmp_path, *, n=48, spc=8, lat=4, lon=5, inner_latlon=True, seed=0):
    """Write a synthetic analysis store; return ``(store, {array_name: source_ndarray})``.

    ``spc`` steps per stored chunk is the fat time-chunk that lets an overlapping
    ``(init, lead)`` grid collapse onto few decodes. ``inner_latlon`` lays fields out
    ``(lat, lon)`` (the contract order); set it ``False`` for the ``(lon, lat)`` layout that
    ``transpose_inner=True`` is meant to fix.
    """
    import zarr

    url = f"file://{tmp_path}/analysis.zarr"
    ensure_local_dir(url)
    store = obstore_store(url, read_only=False)
    group = zarr.open_group(store=store, mode="w")

    # CF-encoded time coordinate: 6-hourly integers since a reanalysis epoch.
    t = group.create_array("time", shape=(n,), chunks=(n,), dtype="i8")
    t[:] = (np.arange(n) * STEP_H).astype("i8")
    t.attrs["units"] = TIME_UNITS

    inner = (lat, lon) if inner_latlon else (lon, lat)
    group.create_array("latitude", shape=(lat,), chunks=(lat,), dtype="f4")[:] = (
        np.linspace(90.0, -90.0, lat, dtype="f4")
    )
    group.create_array("longitude", shape=(lon,), chunks=(lon,), dtype="f4")[:] = (
        np.linspace(0.0, 360.0, lon, endpoint=False, dtype="f4")
    )

    rng = np.random.default_rng(seed)
    srcs: dict[str, np.ndarray] = {}
    for name in ("2m_temperature", "10m_u_component_of_wind"):
        arr = group.create_array(
            name, shape=(n, *inner), chunks=(spc, *inner), dtype="f4"
        )
        data = rng.standard_normal((n, *inner)).astype("f4")
        arr[:] = data
        srcs[name] = data
    return store, srcs


def read_store_time(store):
    import zarr

    g = zarr.open_group(store=store, mode="r")
    attrs = dict(g["time"].attrs)
    return decode_cf_time(np.asarray(g["time"][:]), attrs["units"])


def test_decode_cf_time_matches_manual_offset():
    # "hours since 1959-01-01" -> datetime64[ns]; index 4 is +24h.
    values = np.array([0, 6, 12, 18, 24], dtype="i8")
    out = decode_cf_time(values, TIME_UNITS)
    assert out.dtype == np.dtype("datetime64[ns]")
    assert out[0] == np.datetime64("1959-01-01T00:00")
    assert out[4] == np.datetime64("1959-01-02T00:00")


def test_feed_contract(tmp_path):
    """One batch: tensor layout, coord keys/order/dtypes, and lead-axis values."""
    store, _ = write_store(tmp_path)
    variables = ["t2m", "u10m"]
    var_map = {"t2m": "2m_temperature", "u10m": "10m_u_component_of_wind"}
    leads = np.array([np.timedelta64(h, "h") for h in (0, 6, 12)])

    feed = InSituForecastFeed(
        store,
        variables=variables,
        var_map=var_map,
        lead_times=leads,
        sample_range=(0, 8),
        batch_size=8,
    )
    batches = list(feed)
    feed.dataset.close()

    assert len(batches) == 1
    x, coords = batches[0]
    # (time, lead_time, variable, lat, lon)
    assert x.shape == (8, len(leads), len(variables), 4, 5)
    assert x.dtype.is_floating_point

    assert list(coords.keys()) == ["time", "lead_time", "variable", "lat", "lon"]
    assert coords["time"].dtype == np.dtype("datetime64[ns]")
    assert coords["lead_time"].dtype == np.dtype("timedelta64[ns]")
    assert np.array_equal(coords["variable"], np.array(variables))
    assert coords["lat"].dtype == np.float32 and coords["lon"].dtype == np.float32
    assert np.array_equal(coords["lead_time"], leads.astype("timedelta64[ns]"))
    # init times are the first 8 store steps
    assert np.array_equal(coords["time"], read_store_time(store)[:8])


def test_feed_values_are_shift_views(tmp_path):
    """Each (lead, variable) cell is a sample-axis shift view of the stored array."""
    store, srcs = write_store(tmp_path)
    variables = ["t2m", "u10m"]
    var_map = {"t2m": "2m_temperature", "u10m": "10m_u_component_of_wind"}
    lead_steps = (0, 1, 2)  # in units of the 6-h store step
    leads = np.array([np.timedelta64(k * STEP_H, "h") for k in lead_steps])

    feed = InSituForecastFeed(
        store,
        variables=variables,
        var_map=var_map,
        lead_times=leads,
        sample_range=(0, 8),
        batch_size=8,
    )
    ((x, _coords),) = list(feed)
    feed.dataset.close()
    x = x.numpy()

    for li, k in enumerate(lead_steps):
        for vi, vid in enumerate(variables):
            src = srcs[var_map[vid]]
            for t in range(8):  # init index t -> valid index t + k
                np.testing.assert_array_equal(x[t, li, vi], src[t + k])


def test_decode_once_dedup(tmp_path):
    """The thesis: an overlapping (init, lead) grid decodes each shared chunk once.

    48 requested field-reads (8 inits x 3 leads x 2 vars) touch sample indices 0..9, which
    span two fat chunks (spc=8) per variable -> exactly 4 unique decodes.
    """
    store, _ = write_store(tmp_path, n=48, spc=8)
    variables = ["t2m", "u10m"]
    var_map = {"t2m": "2m_temperature", "u10m": "10m_u_component_of_wind"}
    leads = np.array([np.timedelta64(h, "h") for h in (0, 6, 12)])

    feed = InSituForecastFeed(
        store,
        variables=variables,
        var_map=var_map,
        lead_times=leads,
        sample_range=(0, 8),
        batch_size=8,
    )
    list(feed)
    decodes = feed.dataset.cache_misses
    feed.dataset.close()

    requested = 8 * len(leads) * len(variables)
    # touched sample indices 0..9 -> chunks {0, 1} (spc=8), per each of 2 variables
    touched = {i + k for i in range(8) for k in (0, 1, 2)}
    unique_chunks = len({idx // 8 for idx in touched}) * len(variables)
    assert decodes == unique_chunks == 4
    assert decodes < requested


def test_transpose_inner_swaps_field_axes(tmp_path):
    """A store laid out (lon, lat) yields (lat, lon) fields under transpose_inner=True."""
    store, srcs = write_store(tmp_path, lat=4, lon=5, inner_latlon=False)
    feed = InSituForecastFeed(
        store,
        variables=["t2m"],
        var_map={"t2m": "2m_temperature"},
        sample_range=(0, 8),
        batch_size=8,
        transpose_inner=True,
    )
    x, coords = list(feed)[0]
    feed.dataset.close()

    assert x.shape[-2:] == (4, 5)  # (lat, lon)
    assert coords["lat"].shape[0] == 4 and coords["lon"].shape[0] == 5
    # value at (0,0) is the stored (lon, lat) field transposed
    np.testing.assert_array_equal(x.numpy()[0, 0, 0], srcs["2m_temperature"][0].T)


def test_persistent_cache_across_runs(tmp_path):
    """cache_dir persists decoded chunks: a second run over the same store hits the cache."""
    store, _ = write_store(tmp_path, n=48, spc=8)
    kw = {
        "variables": ["t2m"],
        "var_map": {"t2m": "2m_temperature"},
        "lead_times": np.array([np.timedelta64(h, "h") for h in (0, 6, 12)]),
        "sample_range": (0, 8),
        "batch_size": 8,
        "cache_dir": str(tmp_path / "cache"),
    }

    cold = InSituForecastFeed(store, **kw)
    list(cold)
    cold_misses, cold_hits = cold.dataset.cache_misses, cold.dataset.cache_hits
    cold.dataset.close()

    warm = InSituForecastFeed(store, **kw)
    list(warm)
    warm_misses, warm_hits = warm.dataset.cache_misses, warm.dataset.cache_hits
    warm.dataset.close()

    assert cold_misses > 0 and cold_hits == 0  # cold run fetches + populates the cache
    assert warm_hits == cold_misses  # warm run serves every chunk from disk
    assert warm_misses == 0  # ... and fetches nothing


def test_lead_not_multiple_of_store_step_raises(tmp_path):
    store, _ = write_store(tmp_path)  # 6-h store step
    with pytest.raises(ValueError, match="integer multiple of the store step"):
        InSituForecastFeed(
            store,
            variables=["t2m"],
            var_map={"t2m": "2m_temperature"},
            lead_times=np.array(
                [np.timedelta64(90, "m")]
            ),  # 1.5 h, not a multiple of 6 h
            sample_range=(0, 4),
        )


def test_verification_lead_past_store_end_raises(tmp_path):
    """A positive lead whose read leaves the store end is rejected, not silently dropped."""
    store, _ = write_store(tmp_path, n=48)  # 6-h step; 240 h = 40 steps
    with pytest.raises(ValueError, match=r"outside the store|valid init range"):
        InSituForecastFeed(
            store,
            variables=["t2m"],
            var_map={"t2m": "2m_temperature"},
            lead_times=np.array([np.timedelta64(h, "h") for h in (0, 240)]),
            sample_range=(0, 20),  # init 20 + 40-step lead -> index 60 >> 48
        )


def test_history_lead_before_store_start_raises(tmp_path):
    """A negative (history) lead whose read precedes index 0 is rejected."""
    store, _ = write_store(tmp_path, n=48)
    with pytest.raises(ValueError, match=r"outside the store|valid init range"):
        InSituForecastFeed(
            store,
            variables=["t2m"],
            var_map={"t2m": "2m_temperature"},
            lead_times=np.array(
                [np.timedelta64(h, "h") for h in (-240, 0)]
            ),  # -40 steps
            sample_range=(0, 44),  # init 0 - 40-step history -> index -40
        )


def test_sample_range_none_defaults_to_valid_window(tmp_path):
    """With no sample_range, the feed covers exactly the inits whose leads all fit the store."""
    store, _ = write_store(tmp_path, n=48)  # spc=8
    feed = InSituForecastFeed(
        store,
        variables=["t2m"],
        var_map={"t2m": "2m_temperature"},
        lead_times=np.array(
            [np.timedelta64(h, "h") for h in (0, 6, 12)]
        ),  # steps 0,1,2
        batch_size=8,
    )
    n_inits = sum(x.shape[0] for x, _ in feed)
    feed.dataset.close()
    # valid_anchor_range([0,1,2], 48) = [0, 46): the last two inits would read past the end
    assert n_inits == 46
