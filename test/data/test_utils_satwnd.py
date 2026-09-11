# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NCEP satwnd (AMV) BUFR decoder and NNJAObsSatwnd source."""

import math
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from earth2studio.data import NNJAObsSatwnd, utils_ncep, utils_satwnd
from earth2studio.lexicon import NNJAObsSatwndLexicon

pytest.importorskip("pybufrkit", reason="pybufrkit not installed")

# Mnemonic ids as they appear in a real 2015 dump: SWQM and EHAM are NCEP-local.
TABLE_B = {
    1007: ("SAID     SATELLITE IDENTIFIER",),
    1032: ("GNAP     GENERATING APPLICATION",),
    1044: ("GNAPS    Standard generating app",),
    2023: ("SWCM     SATELLITE DERIVED WIND",),
    2163: ("HAMD     HEIGHT ASSIGNMENT METHO",),
    2162: ("EHAM     EXTENDED HEIGHT ASSIGNM",),
    2164: ("TCMD     TRACER CORRELATION METH",),
    5002: ("CLAT     LATITUDE(COARSE ACCURAC",),
    6002: ("CLON     LONGITUDE (COARSE ACCUR",),
    5001: ("CLATH    LATITUDE (HIGH ACCURACY",),
    6001: ("CLONH    LONGITUDE (HIGH ACCURAC",),
    7004: ("PRLC     PRESSURE",),
    7024: ("SAZA     SATELLITE ZENITH ANGLE",),
    11001: ("WDIR     WIND DIRECTION",),
    11002: ("WSPD     WIND SPEED",),
    33007: ("PCCF     PERCENT CONFIDENCE",),
    33216: ("SWQM     SDMEDIT SATELLITE WIND",),
}
IDS = utils_satwnd.resolve_mnemonics(TABLE_B)
PLAN = {
    "u": ("u", NNJAObsSatwndLexicon.get_item("u")[1]),
    "v": ("v", NNJAObsSatwndLexicon.get_item("v")[1]),
}
WANTED = {"u": "u", "v": "v"}
BOUNDS = (datetime(2015, 1, 1, 0), datetime(2015, 1, 1, 3))


def _stream(pairs: list[tuple[str, object]]):
    descriptors = [SimpleNamespace(id=IDS[name]) for name, _ in pairs]
    values = [value for _, value in pairs]
    return descriptors, values


def _time_block():
    return [
        ("YEAR", 2015),
        ("MNTH", 1),
        ("DAYS", 1),
        ("HOUR", 1),
        ("MINU", 30),
        ("SECO", 6.0),
    ]


def _legacy_nesdis(said=257, swcm=1, swqm=2, wdir=260.0, wspd=53.5, prlc=32500.0):
    """NC005010-style layout: SWQM, HAMD, GNAP triplets after WDIR and WSPD."""
    qc = [
        ("GNAP", 1),
        ("PCCF", 99),
        ("GNAP", 2),
        ("PCCF", 62),
        ("GNAP", 3),
        ("PCCF", 97),
        ("GNAP", 4),
        ("PCCF", 44),
    ]
    return _stream(
        [
            ("SAID", said),
            ("SAZA", 45.43),
            *_time_block(),
            ("CLAT", 36.96),
            ("CLON", -60.73),
            ("SWCM", swcm),
            ("SWQM", swqm),
            ("HAMD", 4),
            ("PRLC", prlc),
            ("WDIR", wdir),
            *qc,
            ("WSPD", wspd),
            *qc,
        ]
    )


def _goes_r(said=270, swcm=1, subset_pccf=(("GNAPS", 4), ("PCCF", 89), ("GNAPS", 5), ("PCCF", 89), ("GNAPS", 6), ("PCCF", None), ("GNAPS", 7), ("PCCF", 61))):  # fmt: skip
    """NC005030-style layout: CLATH/CLONH, EHAM, first PRLC/WDIR/WSPD then
    per-level replications with missing PRLC, AMVQIC GNAPS block last."""
    return _stream(
        [
            ("SAID", said),
            ("SWCM", swcm),
            ("CLATH", 61.10906),
            ("CLONH", -90.3736),
            *_time_block(),
            ("EHAM", 15),
            ("PRLC", 41780.0),
            ("WDIR", 313.0),
            ("WSPD", 35.2),
            ("SAZA", 70.43),
            ("EHAM", None),
            ("PRLC", None),
            ("SAID", said),
            ("SAZA", None),
            ("PRLC", None),
            *subset_pccf,
            ("PRLC", None),
        ]
    )


def test_type_table_matches_gsi_sattabin_spot_checks():
    table = utils_satwnd.SATWND_TYPE_TABLE
    assert table[(10, 257, 1)] == (245, 7)
    assert table[(19, 257, 1)] == (240, 11)
    assert table[(30, 270, 1)] == (245, 15)
    assert table[(31, 270, 4)] == (247, 19)
    assert table[(31, 270, 5)] == (247, 19)
    assert table[(44, 172, 1)] == (252, 4)
    assert table[(46, 174, 5)] == (250, 4)
    assert table[(47, 173, 1)] == (253, 5)
    assert table[(64, 54, 1)] == (253, 1)
    assert table[(66, 57, 3)] == (254, 1)
    assert table[(67, 70, 1)] == (253, 2)
    assert table[(71, 784, 3)] == (258, 8)
    assert table[(71, 784, 5)] == (259, 8)
    assert table[(72, 854, 1)] == (255, 12)
    assert table[(81, 3, 1)] == (244, 10)
    assert table[(91, 225, 1)] == (260, 14)
    assert table[(99, 731, 6)] == (241, 20)
    assert table[(24, 470, 1)] == (256, -1)
    # Not typed by GSI: wrong method for the subset, or a SAID outside the
    # platform list (INSAT-3DR 473 in 2024 dumps).
    assert (10, 257, 3) not in table
    assert (24, 473, 1) not in table
    assert (99, 470, 1) not in table


def test_resolve_mnemonics_prefers_file_table_over_wmo_fallbacks():
    ids = utils_satwnd.resolve_mnemonics(
        {2164: ("EHAM     EXTENDED HEIGHT ASSIGNM",), 33222: ("SWQM     QM",)}
    )
    assert ids["EHAM"] == 2164
    assert ids["SWQM"] == 33222
    assert ids["PRLC"] == 7004
    assert utils_satwnd.resolve_mnemonics({})["SWQM"] == 33216


def test_bufr_local_subcategory_by_edition():
    ed3 = bytearray(24)
    ed3[7] = 3
    ed3[16] = 5
    ed3[17] = 30
    assert utils_satwnd.bufr_local_subcategory(bytes(ed3)) == 30
    ed4 = bytearray(24)
    ed4[7] = 4
    ed4[18] = 5
    ed4[20] = 67
    assert utils_satwnd.bufr_local_subcategory(bytes(ed4)) == 67


def test_pressure_to_height_m_inverts_standard_atmosphere():
    pressure_pa = np.array([101_325.0, 22_632.06, 5_474.89, 868.02, 50_000.0])
    height = utils_satwnd.pressure_to_height_m(pressure_pa)
    assert height[:4] == pytest.approx([0.0, 11_000.0, 20_000.0, 32_000.0], abs=1.0)
    assert 5_400.0 < height[4] < 5_700.0
    # Above-standard surface pressure floors at 0, garbage is NaN.
    out = utils_satwnd.pressure_to_height_m(np.array([110_000.0, np.nan, -5.0, 0.01]))
    assert out[0] == 0.0
    assert np.isnan(out[1:]).all()


def test_legacy_nesdis_subset_rows():
    descriptors, values = _legacy_nesdis()
    rows = utils_satwnd._extract_satwnd_subset(
        descriptors, values, 10, IDS, WANTED, *BOUNDS
    )
    assert [r["variable"] for r in rows] == ["u", "v"]
    u, v = rows
    # Meteorological direction 260 deg at 53.5 m/s.
    assert u["observation"] == pytest.approx(-53.5 * math.sin(math.radians(260.0)))
    assert v["observation"] == pytest.approx(-53.5 * math.cos(math.radians(260.0)))
    assert u["time"] == datetime(2015, 1, 1, 1, 30, 6)
    assert u["lat"] == pytest.approx(np.float32(36.96))
    assert u["lon"] == pytest.approx(np.float32(360.0 - 60.73))
    assert u["type"] == 245 and u["gsi_case"] == 7
    assert u["class"] == "SATWND" and u["station"] == "IR257"
    assert u["quality"] == 2 and u["height_method"] == 4
    assert u["pres"] == pytest.approx(32500.0)
    assert 8_000.0 < u["elev"] < 9_500.0
    assert u["satellite_id"] == 257 and u["subset"] == "NC005010"
    assert u["wind_method"] == 1 and u["satellite_za"] == pytest.approx(45.43)
    # NESDIS legacy: GNAP 1 = qifn, 3 = qify, 4 = ee.
    assert u["qi"] == 99 and u["qi_forecast"] == 97 and u["expected_error"] == 44


def test_goes_r_subset_uses_first_level_and_amvqic_gnaps():
    descriptors, values = _goes_r()
    rows = utils_satwnd._extract_satwnd_subset(
        descriptors, values, 30, IDS, WANTED, *BOUNDS
    )
    assert len(rows) == 2
    u = rows[0]
    assert u["type"] == 245 and u["gsi_case"] == 15
    assert u["pres"] == pytest.approx(41780.0)
    assert u["height_method"] == 15
    assert u["satellite_za"] == pytest.approx(70.43)
    assert u["quality"] is None  # no SWQM in GOES-R BUFR
    assert u["lat"] == pytest.approx(np.float32(61.10906))
    # AMVQIC: GNAPS 5 -> qi, 6 -> qi with forecast (missing), 7 -> ee.
    assert u["qi"] == 89 and math.isnan(u["qi_forecast"]) and u["expected_error"] == 61

    # EUMETSAT 310077 layout orders GNAPS 6,5,4,2: ee falls back to GNAPS 2.
    descriptors, values = _goes_r(
        said=57,
        subset_pccf=(("GNAPS", 6), ("PCCF", 56), ("GNAPS", 5), ("PCCF", 69), ("GNAPS", 4), ("PCCF", 58), ("GNAPS", 2), ("PCCF", 0)),  # fmt: skip
    )
    rows = utils_satwnd._extract_satwnd_subset(
        descriptors, values, 67, IDS, WANTED, *BOUNDS
    )
    assert rows[0]["type"] == 253 and rows[0]["gsi_case"] == 2
    assert rows[0]["qi"] == 69 and rows[0]["qi_forecast"] == 56
    assert rows[0]["expected_error"] == 0


def test_jma_gnap_codes_and_untyped_subsets_dropped():
    descriptors, values = _stream(
        [
            ("SAID", 172),
            ("SAZA", 44.51),
            *_time_block(),
            ("CLAT", 37.1),
            ("CLON", 155.92),
            ("SWCM", 1),
            ("PRLC", 39330.0),
            ("GNAP", 101),
            ("PCCF", 76),
            ("GNAP", 102),
            ("PCCF", 88),
            ("GNAP", 103),
            ("PCCF", 0),
            ("WDIR", 270.0),
            ("WSPD", 10.0),
        ]
    )
    rows = utils_satwnd._extract_satwnd_subset(
        descriptors, values, 44, IDS, WANTED, *BOUNDS
    )
    assert rows[0]["type"] == 252 and rows[0]["gsi_case"] == 4
    assert rows[0]["qi"] == 88 and rows[0]["qi_forecast"] == 76
    assert rows[0]["observation"] == pytest.approx(10.0)  # westerly -> +u
    assert rows[1]["observation"] == pytest.approx(0.0, abs=1e-6)

    # INSAT is typed 256 with no GSI case; a SAID outside the table is dropped.
    assert (
        utils_satwnd._extract_satwnd_subset(
            descriptors, values, 24, IDS, WANTED, *BOUNDS
        )
        == []
    )
    descriptors, values = _legacy_nesdis(said=470)
    rows = utils_satwnd._extract_satwnd_subset(
        descriptors, values, 24, IDS, WANTED, *BOUNDS
    )
    assert rows[0]["type"] == 256 and rows[0]["gsi_case"] is None


def test_subset_filters_time_window_and_missing_wind():
    descriptors, values = _legacy_nesdis()
    late = (datetime(2015, 1, 1, 2), datetime(2015, 1, 1, 3))
    assert utils_satwnd._extract_satwnd_subset(descriptors, values, 10, IDS, WANTED, *late) == []  # fmt: skip
    descriptors, values = _legacy_nesdis(wspd=None)
    assert utils_satwnd._extract_satwnd_subset(descriptors, values, 10, IDS, WANTED, *BOUNDS) == []  # fmt: skip
    only_u = utils_satwnd._extract_satwnd_subset(
        *_legacy_nesdis(), 10, IDS, {"u": "u"}, *BOUNDS
    )
    assert [r["variable"] for r in only_u] == ["u"]


class _Message:
    def __init__(self, subsets):
        self.n_subsets = SimpleNamespace(value=len(subsets))
        self.template_data = SimpleNamespace(
            value=SimpleNamespace(
                decoded_descriptors_all_subsets=[d for d, _ in subsets],
                decoded_values_all_subsets=[v for _, v in subsets],
            )
        )


class _Decoder:
    def __init__(self, messages):
        self.messages = messages

    def process(self, message):
        return self.messages[message]


def _message_bytes(subset: int) -> bytes:
    raw = bytearray(b"BUFR" + bytes(20))
    raw[7] = 3
    raw[16] = 5
    raw[17] = subset
    return bytes(raw)


def test_decode_satwnd_end_to_end(tmp_path, monkeypatch):
    local = tmp_path / "gdas.satwnd.bufr_d"
    local.write_bytes(b"satwnd-bytes")
    nesdis = _message_bytes(10)
    goes_r = _message_bytes(30)
    monkeypatch.setattr(
        utils_satwnd,
        "_parse_prepbufr_messages",
        lambda data, *, silence_noise: (TABLE_B, {}, [(nesdis, 5), (goes_r, 5)]),
    )
    monkeypatch.setattr(
        utils_satwnd,
        "_create_decoder",
        lambda tb, td: _Decoder(
            {
                nesdis: _Message([_legacy_nesdis(), _legacy_nesdis(swcm=3)]),
                goes_r: _Message([_goes_r()]),
            }
        ),
    )
    df = utils_satwnd.decode_satwnd(str(local), PLAN, *BOUNDS, decode_workers=1)
    # Two typed NESDIS/GOES-R winds x (u, v); SWCM 3 under NC005010 is untyped.
    assert len(df) == 4
    assert list(df.columns) == utils_satwnd.NCEP_SATWND_PUBLIC_SCHEMA.names
    assert sorted(df["type"].unique().tolist()) == [245]
    # Rows are grouped by variable (all u, then all v) by _finalize_rows.
    assert df["variable"].tolist() == ["u", "u", "v", "v"]
    assert sorted(df["subset"].tolist()) == ["NC005010", "NC005010", "NC005030", "NC005030"]  # fmt: skip
    assert df["pres"].dtype == np.float32
    assert str(df["satellite_id"].dtype) == "uint16[pyarrow]"
    assert df.loc[df["subset"] == "NC005030", "quality"].isna().all()

    source = NNJAObsSatwnd(cache=False, verbose=False, decode_workers=1)
    task = utils_ncep.NCEPObsTask(
        uri="s3://example/satwnd",
        route="satwnd",
        datetime_file=BOUNDS[0],
        datetime_min=BOUNDS[0],
        datetime_max=BOUNDS[1],
        var_plan=PLAN,
    )
    public = source._decode_file(str(local), task)
    pd.testing.assert_frame_equal(public, df[source.SCHEMA.names])


def test_nnja_obs_satwnd_uri_and_tasks():
    source = NNJAObsSatwnd(cache=False, verbose=False)
    uri = source._build_uri("satwnd", datetime(2024, 1, 1, 6))
    assert uri.endswith("amv/satwnd/2024/01/bufr/gdas.20240101.t06z.satwnd.tm00.bufr_d")
    with pytest.raises(ValueError):
        source._build_uri("prepbufr", datetime(2024, 1, 1))
    tasks = source._create_tasks([datetime(2024, 1, 1, 6)], ["u", "v"])
    assert len(tasks) == 1 and tasks[0].route == "satwnd"
    assert set(tasks[0].var_plan) == {"u", "v"}
    assert tasks[0].var_plan["u"][0] == "u"
    assert NNJAObsSatwnd.available(datetime(1990, 6, 1))
    assert not NNJAObsSatwnd.available(datetime(1970, 6, 1))
    assert source.resolve_fields(["time", "qi"]).names == ["time", "qi"]
