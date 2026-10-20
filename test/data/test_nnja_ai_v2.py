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

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest

from earth2studio.data import NNJAAIv2ObsConv, NNJAAIv2ObsSat
from earth2studio.data import nnja_ai_v2 as module

DAY = date(2025, 7, 10)
T0 = datetime(2025, 7, 10, 6, 0)


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def _ts(values: list[datetime]) -> pa.Array:
    return pa.array(values, pa.timestamp("ms", tz="UTC"))


def _write(root: Path, key: str, table: pa.Table) -> None:
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _sat_common(times: list[datetime], said: list[int], fovn: list[int]) -> dict:
    n = len(times)
    return {
        "MSG_DATE": _ts([_utc(2025, 7, 10, 5)] * n),
        "SRC_FILENAME": pa.array(["gdas.20250710.t06z.sensor.tm00.bufr_d"] * n),
        "OBS_TIME": _ts(times),
        "LAT": pa.array([10.0, -20.0, 30.0][:n], pa.float32()),
        "LON": pa.array([-100.0, 20.0, 170.0][:n], pa.float32()),
        "SAID": pa.array(said, pa.int32()),
        "FOVN": pa.array(fovn, pa.int16()),
        "SAZA": pa.array([1.0, 2.0, 3.0][:n], pa.float64()),
        "SOZA": pa.array([40.0, 50.0, 60.0][:n], pa.float64()),
        "SOLAZI": pa.array([100.0, 110.0, 120.0][:n], pa.float64()),
        "BEARAZ": pa.array([200.0, 210.0, 220.0][:n], pa.float64()),
    }


def _write_amsua(root: Path) -> None:
    # Footprints: metop-b in window, n19 in window, metop-b 2 h before window.
    times = [_utc(2025, 7, 10, 6, 2), _utc(2025, 7, 10, 5, 55), _utc(2025, 7, 10, 4)]
    columns = _sat_common(times, said=[3, 223, 3], fovn=[1, 15, 30])
    columns["HOLS"] = pa.array([5, 100, 0], pa.int16())
    chnm = [list(range(1, 16))] * 3
    tmbr = [[200.0 + c for c in range(15)] for _ in range(3)]
    tmbr[0][4] = None  # one missing channel on the first footprint
    columns["BRITCSTC"] = pa.StructArray.from_arrays(
        [
            pa.array(chnm, pa.list_(pa.int16(), 15)),
            pa.array(tmbr, pa.list_(pa.float64(), 15)),
            pa.array([[0.5] * 15] * 3, pa.list_(pa.float64(), 15)),
        ],
        names=["CHNM", "TMBR", "CSTC"],
    )
    _write(
        root,
        "amsua/1bamua/NC021023/OBS_DATE=2025-07-10/1bamua.NC021023.parquet",
        pa.table(columns),
    )


def _write_atms(root: Path) -> None:
    times = [_utc(2025, 7, 10, 6, 1), _utc(2025, 7, 10, 6, 3)]
    columns = _sat_common(times, said=[224, 225], fovn=[48, 96])
    columns["SLNM"] = pa.array([7, 8], pa.int16())
    columns["ATMSSQ"] = pa.array([0, 4], pa.int64())
    columns["ATMSGQ"] = pa.array([2, 2], pa.int64())
    channels = [1, 2, 3]
    columns["ATMSCH"] = pa.StructArray.from_arrays(
        [
            pa.array([channels] * 2, pa.list_(pa.int16())),
            pa.array([[23.8e9, 31.4e9, 50.3e9]] * 2, pa.list_(pa.float64())),
            pa.array([[240.0, 241.0, 242.0]] * 2, pa.list_(pa.float64())),
            pa.array(
                [[250.0, 251.0, 252.0], [260.0, None, 262.0]], pa.list_(pa.float64())
            ),
            pa.array([[0, 0, 1]] * 2, pa.list_(pa.int64())),
        ],
        names=["CHNM", "SCCF", "TMANT", "TMBR", "ATMSCHQ"],
    )
    _write(
        root,
        "atms/atms/NC021203/OBS_DATE=2025-07-10/atms.NC021203.parquet",
        pa.table(columns),
    )


def _events(values: list[list[list[int | None]]]) -> pa.Array:
    return pa.array(values, pa.list_(pa.list_(pa.int32())))


def _write_adpupa(root: Path) -> None:
    """Two soundings, two levels each; the second sounding is out of window."""
    prslevel = pa.StructArray.from_arrays(
        [
            pa.array([[0, 1], [0, 1]], pa.list_(pa.int32())),
            pa.StructArray.from_arrays(
                [
                    pa.StructArray.from_arrays(
                        [_events([[[2], [2]], [[2], [2]]])], names=["PQM"]
                    )
                ],
                names=["P__EVENT"],
            ),
            pa.StructArray.from_arrays(
                [
                    pa.StructArray.from_arrays(
                        [_events([[[9, 2], [2]], [[2], []]])], names=["TQM"]
                    )
                ],
                names=["T__EVENT"],
            ),
            pa.StructArray.from_arrays(
                [
                    pa.StructArray.from_arrays(
                        [_events([[[1], [1]], [[1], [1]]])], names=["QQM"]
                    )
                ],
                names=["Q__EVENT"],
            ),
            pa.StructArray.from_arrays(
                [
                    pa.StructArray.from_arrays(
                        [_events([[[2], [2]], [[2], [2]]])], names=["WQM"]
                    )
                ],
                names=["W__EVENT"],
            ),
            pa.StructArray.from_arrays(
                [
                    pa.array([[None, 11.5], [None, None]], pa.list_(pa.float64())),
                    pa.array([[None, 46.0], [None, None]], pa.list_(pa.float64())),
                    pa.array([[None, 0.5], [None, None]], pa.list_(pa.float64())),
                ],
                names=["XDR", "YDR", "HRDR"],
            ),
        ],
        names=["CAT", "P___INFO", "T___INFO", "Q___INFO", "W___INFO", "DRFTINFO"],
    )
    table = pa.table(
        {
            "MSG_DATE": _ts([_utc(2025, 7, 10, 6)] * 2),
            "SRC_FILENAME": pa.array(["gdas.20250710.t06z.prepbufr.nr"] * 2),
            "OBS_TIME": _ts([_utc(2025, 7, 10, 6), _utc(2025, 7, 10, 9)]),
            "LAT": pa.array([45.0, 0.0], pa.float32()),
            "LON": pa.array([10.0, -50.0], pa.float32()),
            "SID": pa.array(["10954", "99999"]),
            "TYP": pa.array([120, 120], pa.int32()),
            "ELV": pa.array([761, 5], pa.int32()),
            "PRSLEVEL": prslevel,
            "POB_QC": pa.array(
                [[1000.0, 850.0], [1000.0, 850.0]], pa.list_(pa.float64())
            ),
            "POB_RAW": pa.array(
                [[1000.0, 850.0], [1000.0, 850.0]], pa.list_(pa.float64())
            ),
            "TOB_QC": pa.array([[20.0, 5.0], [20.0, None]], pa.list_(pa.float64())),
            "TOB_RAW": pa.array([[19.0, 5.0], [20.0, None]], pa.list_(pa.float64())),
            "QOB_QC": pa.array([[10000, 5000], [1, 1]], pa.list_(pa.int32())),
            "QOB_RAW": pa.array([[10000, 5000], [1, 1]], pa.list_(pa.int32())),
            "UOB_QC": pa.array([[None, 3.0], [1.0, 1.0]], pa.list_(pa.float64())),
            "UOB_RAW": pa.array([[None, 3.0], [1.0, 1.0]], pa.list_(pa.float64())),
            "VOB_QC": pa.array([[None, -4.0], [1.0, 1.0]], pa.list_(pa.float64())),
            "VOB_RAW": pa.array([[None, -4.0], [1.0, 1.0]], pa.list_(pa.float64())),
            "ZOB_QC": pa.array([[761, 1500], [5, 1500]], pa.list_(pa.int32())),
            "ZOB_RAW": pa.array([[761, 1500], [5, 1500]], pa.list_(pa.int32())),
        }
    )
    _write(
        root, "conv/prepbufr/ADPUPA/OBS_DATE=2025-07-10/prepbufr.ADPUPA.parquet", table
    )


def _write_adpsfc(root: Path) -> None:
    """Three surface reports: mass, wind and one below the pressure cut."""

    def event(name: str, values: list[list[int | None]]) -> pa.StructArray:
        return pa.StructArray.from_arrays(
            [
                pa.StructArray.from_arrays(
                    [pa.array(values, pa.list_(pa.int32()))], names=[name]
                )
            ],
            names=[f"{name[0]}__EVENT"],
        )

    table = pa.table(
        {
            "MSG_DATE": _ts([_utc(2025, 7, 10, 6)] * 3),
            "SRC_FILENAME": pa.array(["gdas.20250710.t06z.prepbufr.nr"] * 3),
            "OBS_TIME": _ts([_utc(2025, 7, 10, 6, 5)] * 3),
            "LAT": pa.array([-41.5, -41.5, 60.0], pa.float32()),
            "LON": pa.array([147.2, 147.2, -1.0], pa.float32()),
            "SID": pa.array(["YMLT", "YMLT", "HIGH"]),
            "TYP": pa.array([187, 287, 181], pa.int32()),
            "ELV": pa.array([178, 178, 5000], pa.int32()),
            "CAT": pa.array([0, 6, 0], pa.int32()),
            "P___INFO": event("PQM", [[2], [2], [2]]),
            "T___INFO": event("TQM", [[9, 2], [], [2]]),
            "Q___INFO": event("QQM", [[2], [], [2]]),
            "W___INFO": event("WQM", [[], [2], []]),
            "POB_QC": pa.array([974.1, 974.1, 450.0], pa.float64()),
            "POB_RAW": pa.array([974.1, 974.1, 450.0], pa.float64()),
            "TOB_QC": pa.array([12.1, None, -30.0], pa.float64()),
            "TOB_RAW": pa.array([12.0, None, -30.0], pa.float64()),
            "QOB_QC": pa.array([8000, None, 100], pa.int32()),
            "QOB_RAW": pa.array([8000, None, 100], pa.int32()),
            "UOB_QC": pa.array([None, 2.0, None], pa.float64()),
            "UOB_RAW": pa.array([None, 2.0, None], pa.float64()),
            "VOB_QC": pa.array([None, -1.0, None], pa.float64()),
            "VOB_RAW": pa.array([None, -1.0, None], pa.float64()),
            "ZOB_QC": pa.array([178, 178, 5000], pa.int32()),
            "ZOB_RAW": pa.array([178, 178, 5000], pa.int32()),
        }
    )
    _write(
        root, "conv/prepbufr/ADPSFC/OBS_DATE=2025-07-10/prepbufr.ADPSFC.parquet", table
    )


@pytest.fixture
def archive(tmp_path: Path) -> str:
    _write_amsua(tmp_path)
    _write_atms(tmp_path)
    _write_adpupa(tmp_path)
    _write_adpsfc(tmp_path)
    return f"file://{tmp_path}"


def test_nnja_ai_v2_layout_helpers():
    assert module._split_root("gs://nnja-ai/data/v2/") == ("gs://nnja-ai", "data/v2")
    assert module._split_root("file:///data/archive") == ("file:///data/archive", "")
    with pytest.raises(ValueError):
        module._split_root("/data/archive")

    windows = module._merge_windows(
        [T0, T0 + timedelta(hours=1), T0 + timedelta(days=2)],
        timedelta(minutes=-40),
        timedelta(minutes=40),
    )
    assert windows == [
        (T0 - timedelta(minutes=40), T0 + timedelta(minutes=100)),
        (T0 + timedelta(days=2, minutes=-40), T0 + timedelta(days=2, minutes=40)),
    ]
    assert module._partition_days(windows) == [DAY, date(2025, 7, 12)]
    assert module._partition_days(
        [(datetime(2025, 7, 9, 23), datetime(2025, 7, 11, 1))]
    ) == [date(2025, 7, 9), DAY, date(2025, 7, 11)]

    product = module._SAT_PRODUCTS["amsua"]
    assert product.partition_key(DAY) == (
        "amsua/1bamua/NC021023/OBS_DATE=2025-07-10/1bamua.NC021023.parquet"
    )


def test_nnja_ai_v2_column_helpers():
    column = module._uint_column(np.array([1.0, np.nan, -1.0, 70000.0]), pa.uint16())
    assert column.to_pylist() == [1, None, None, None]

    struct_of_arrays = pa.schema(
        [pa.field("PRSLEVEL", pa.struct([pa.field("CAT", pa.list_(pa.int32()))]))]
    )
    assert module._has_path(struct_of_arrays, ("PRSLEVEL", "CAT"))
    assert not module._has_path(struct_of_arrays, ("PRSLEVEL", "TQM"))
    list_of_structs = pa.schema(
        [pa.field("PRSLEVEL", pa.list_(pa.struct([pa.field("CAT", pa.int32())])))]
    )
    with pytest.raises(ValueError, match="struct of arrays"):
        module._has_path(list_of_structs, ("PRSLEVEL", "CAT"))

    # The time filter adopts the column's own type, tz-aware or naive.
    for data_type in (pa.timestamp("ms", tz="UTC"), pa.timestamp("ms")):
        table = pa.table(
            {
                "OBS_TIME": pa.array(
                    [_utc(2025, 7, 10, 1), _utc(2025, 7, 10, 5)],
                    pa.timestamp("ms", tz="UTC"),
                ).cast(data_type),
                "v": [1, 2],
            }
        )
        kept = pads.dataset(table).to_table(
            filter=module._window_filter(
                [(datetime(2025, 7, 10, 4), datetime(2025, 7, 10, 6))], data_type
            )
        )
        assert kept["v"].to_pylist() == [2]


def test_nnja_ai_v2_event_edge():
    events = pa.array([[9, 2], [], [3]], pa.list_(pa.int32()))
    np.testing.assert_array_equal(
        module._event_edge(events, newest=True), [9.0, np.nan, 3.0]
    )
    np.testing.assert_array_equal(
        module._event_edge(events, newest=False), [2.0, np.nan, 3.0]
    )


def test_nnja_ai_v2_obs_sat_fetch_amsua(archive: str):
    ds = NNJAAIv2ObsSat(
        time_tolerance=timedelta(minutes=10), root=archive, cache=False, verbose=False
    )
    df = ds(T0, "amsua")
    assert list(df.columns) == ds.SCHEMA.names
    # Two footprints in the window, 15 channels, one missing value.
    assert len(df) == 2 * 15 - 1
    assert set(df["satellite"]) == {"metop-b", "n19"}
    assert set(df["variable"]) == {"amsua"}
    first = df[(df["satellite"] == "metop-b") & (df["sensor_index"] == 1)].iloc[0]
    assert first["observation"] == pytest.approx(200.0)
    assert first["scan_position"] == 1
    assert first["scan_angle"] == pytest.approx(-48.0 - 1.0 / 3.0, abs=1e-4)
    assert first["elev"] == pytest.approx(5.0)
    assert first["lon"] == pytest.approx(260.0)
    assert first["time"] == pd.Timestamp("2025-07-10 06:02")
    assert first["cycle_time"] == pd.Timestamp("2025-07-10 06:00")
    assert pd.isna(first["wavenumber"]) and pd.isna(first["quality"])
    assert 5 not in set(df.loc[df["satellite"] == "metop-b", "sensor_index"])

    df = ds(T0, "amsua", fields=["time", "observation"])
    assert list(df.columns) == ["time", "observation"]

    ds = NNJAAIv2ObsSat(
        time_tolerance=timedelta(minutes=10),
        satellites=["n19"],
        root=archive,
        cache=False,
        verbose=False,
    )
    assert set(ds(T0, "amsua")["satellite"]) == {"n19"}


def test_nnja_ai_v2_obs_sat_fetch_atms(archive: str):
    ds = NNJAAIv2ObsSat(
        time_tolerance=timedelta(minutes=10), root=archive, cache=False, verbose=False
    )
    df = ds(T0, ["atms", "atms_antenna_temperature"])
    assert list(df.columns) == ds.SCHEMA.names
    assert df["variable"].value_counts().to_dict() == {
        "atms_antenna_temperature": 6,
        "atms": 5,
    }
    tb = df[df["variable"] == "atms"]
    assert tb["observation"].tolist() == [250.0, 251.0, 252.0, 260.0, 262.0]
    assert tb["wavenumber"].iloc[0] == pytest.approx(23.8e9 / 2.99792458e10)
    assert tb["quality"].tolist() == [0, 0, 1, 0, 1]
    assert tb["scan_line"].tolist() == [7, 7, 7, 8, 8]
    assert tb["scan_quality"].tolist() == [0, 0, 0, 4, 4]
    assert tb["granule_quality"].tolist() == [2] * 5
    assert set(tb["satellite"]) == {"npp", "n20"}
    assert tb["elev"].isna().all()
    assert tb["scan_angle"].iloc[0] == pytest.approx(-52.725 + 47 * 1.11, abs=1e-3)


def test_nnja_ai_v2_obs_sat_validation(archive: str):
    with pytest.raises(ValueError, match="Invalid satellite"):
        NNJAAIv2ObsSat(satellites=["voyager"], root=archive)
    ds = NNJAAIv2ObsSat(root=archive, cache=False, verbose=False)
    with pytest.raises(ValueError, match="Unsupported"):
        ds(T0, "amsub")
    with pytest.raises(ValueError, match="MIN_DATE"):
        ds(datetime(1990, 1, 1), "amsua")
    with pytest.raises(ValueError, match="starts in 2008"):
        ds(datetime(2007, 6, 1), "amsua")
    with pytest.raises(module.NNJAAIv2PartitionMissingError):
        ds(datetime(2025, 7, 11, 6), "amsua")
    assert ds.available(T0)
    assert not ds.available(np.datetime64("1990-01-01"))


def test_nnja_ai_v2_obs_conv_fetch(archive: str):
    ds = NNJAAIv2ObsConv(
        time_tolerance=timedelta(minutes=10), root=archive, cache=False, verbose=False
    )
    df = ds(T0, ["t", "q", "u", "v", "pres"])
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["class"]) == {"ADPUPA", "ADPSFC"}

    upper = df[df["class"] == "ADPUPA"].set_index(["variable", "pres"])
    # Only the first sounding is in the window; two levels, 1000 and 850 hPa.
    assert set(upper.index.get_level_values("pres")) == {100000.0, 85000.0}
    assert upper.loc[("t", 100000.0), "observation"] == pytest.approx(293.15)
    assert upper.loc[("t", 100000.0), "quality"] == 9  # newest event
    assert upper.loc[("q", 85000.0), "observation"] == pytest.approx(5e-3)
    assert upper.loc[("u", 85000.0), "observation"] == pytest.approx(3.0)
    assert upper.loc[("v", 85000.0), "observation"] == pytest.approx(-4.0)
    assert ("u", 100000.0) not in upper.index
    assert upper.loc[("t", 85000.0), "elev"] == pytest.approx(1500.0)
    assert upper.loc[("t", 85000.0), "level_cat"] == 1
    assert upper.loc[("t", 85000.0), "station_elev"] == pytest.approx(761.0)
    assert upper.loc[("t", 85000.0), "station"] == "10954"
    # Drift: the 850 hPa level is 0.5 h after the cycle (outside the +-10 min
    # window) and displaced, yet kept because the report header is in window.
    assert upper.loc[("t", 85000.0), "time"] == pd.Timestamp("2025-07-10 06:30")
    assert upper.loc[("t", 85000.0), "report_time"] == pd.Timestamp("2025-07-10 06:00")
    assert upper.loc[("t", 85000.0), "lat"] == pytest.approx(46.0)
    assert upper.loc[("t", 85000.0), "report_lat"] == pytest.approx(45.0)
    assert upper.loc[("t", 100000.0), "lat"] == pytest.approx(45.0)
    assert upper.loc[("t", 100000.0), "cycle_time"] == pd.Timestamp("2025-07-10 06:00")
    assert len(upper.loc["pres"]) == 1  # CAT 0 level of a type-120 report only
    assert upper.loc[("pres", 100000.0), "observation"] == pytest.approx(100000.0)

    surface = df[df["class"] == "ADPSFC"]
    assert surface["variable"].value_counts().to_dict() == {
        "t": 2,
        "q": 2,
        "u": 1,
        "v": 1,
        "pres": 1,
    }
    pres = surface[surface["variable"] == "pres"].iloc[0]
    assert pres["station"] == "YMLT" and pres["type"] == 187
    assert pres["observation"] == pytest.approx(97410.0)
    wind = surface[surface["variable"] == "u"].iloc[0]
    assert wind["type"] == 287 and wind["quality"] == 2 and wind["level_cat"] == 6
    assert surface[surface["variable"] == "t"]["quality"].tolist() == [9, 2]

    original = NNJAAIv2ObsConv(
        time_tolerance=timedelta(minutes=10),
        original_event=True,
        root=archive,
        cache=False,
        verbose=False,
    )(T0, "t")
    first = original[(original["class"] == "ADPUPA") & (original["pres"] == 100000.0)]
    assert first["observation"].iloc[0] == pytest.approx(292.15)
    assert first["quality"].iloc[0] == 2  # oldest event


def test_nnja_ai_v2_obs_conv_exclude_and_validation(archive: str):
    ds = NNJAAIv2ObsConv(
        time_tolerance=timedelta(minutes=10),
        exclude_message_types=("ADPUPA",),
        root=archive,
        cache=False,
        verbose=False,
    )
    assert set(ds(T0, "t")["class"]) == {"ADPSFC"}
    assert ds(datetime(2025, 7, 12), "t").empty

    with pytest.raises(ValueError, match="Unknown PrepBUFR message type"):
        NNJAAIv2ObsConv(exclude_message_types=("ADPUPA", "NOPE"), root=archive)
    with pytest.raises(ValueError, match="GPS"):
        ds(T0, "gps")
    with pytest.raises(ValueError, match="Unsupported"):
        ds(T0, "t2m")
    with pytest.raises(ValueError, match="MIN_DATE"):
        ds(datetime(1999, 1, 1), "t")
    assert ds.available(T0)
    assert not ds.available(datetime(1999, 1, 1))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(600)
def test_nnja_ai_v2_obs_sat_remote():
    ds = NNJAAIv2ObsSat(
        time_tolerance=timedelta(minutes=5), satellites=["metop-c"], verbose=False
    )
    df = ds(T0, "amsua")
    assert list(df.columns) == ds.SCHEMA.names
    assert not df.empty
    assert set(df["satellite"]) == {"metop-c"}
    assert df["sensor_index"].max() == 15


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(600)
def test_nnja_ai_v2_obs_conv_remote():
    ds = NNJAAIv2ObsConv(
        time_tolerance=timedelta(minutes=30),
        exclude_message_types=("AIRCAR", "SATWND", "ASCATW", "VADWND"),
        verbose=False,
    )
    df = ds(T0, ["t", "pres"])
    assert list(df.columns) == ds.SCHEMA.names
    assert {"ADPUPA", "ADPSFC"} <= set(df["class"])
    assert df.loc[df["variable"] == "t", "observation"].between(170, 340).all()
