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

from earth2studio.data import NNJAObsSatwnd, utils_ncep
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
IDS = utils_ncep.resolve_mnemonics(TABLE_B)
TABLE_IDS = {entry[0].split()[0]: key for key, entry in TABLE_B.items()}
PLAN = {
    "u": ("u", NNJAObsSatwndLexicon.get_item("u")[1]),
    "v": ("v", NNJAObsSatwndLexicon.get_item("v")[1]),
}
BOUNDS = (datetime(2015, 1, 1, 0), datetime(2015, 1, 1, 3))


def _stream(pairs: list[tuple[str, object]]):
    descriptors = [
        SimpleNamespace(id=IDS.get(name, TABLE_IDS.get(name))) for name, _ in pairs
    ]
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


def test_resolve_mnemonics_prefers_file_table_over_wmo_fallbacks():
    ids = utils_ncep.resolve_mnemonics(
        {33222: ("SWQM     QM",), 2250: ("CMCM     COMPUTATION METHOD",)}
    )
    assert ids["SWQM"] == 33222
    assert ids["CMCM"] == 2250
    assert ids["PRLC"] == 7004
    fallback = utils_ncep.resolve_mnemonics({})
    assert fallback["SWQM"] == 33216
    assert "CMCM" not in fallback


def test_bufr_local_subcategory_by_edition():
    ed3 = bytearray(24)
    ed3[7] = 3
    ed3[16] = 5
    ed3[17] = 30
    assert utils_ncep.bufr_local_subcategory(bytes(ed3)) == 30
    ed4 = bytearray(24)
    ed4[7] = 4
    ed4[18] = 5
    ed4[20] = 67
    assert utils_ncep.bufr_local_subcategory(bytes(ed4)) == 67


def _decode(subsets, subset_number, bounds=BOUNDS):
    message = _message_bytes(subset_number)
    decoder = _Decoder({message: _Message(subsets)})
    return utils_ncep._decode_satwnd_message(decoder, message, IDS, *bounds)


def test_legacy_nesdis_wind_is_raw():
    winds = _decode([_legacy_nesdis()], 10)
    assert len(winds["time"]) == 1
    # Meteorological direction 260 deg at 53.5 m/s.
    assert winds["u"][0] == pytest.approx(-53.5 * math.sin(math.radians(260.0)))
    assert winds["v"][0] == pytest.approx(-53.5 * math.cos(math.radians(260.0)))
    assert winds["time"][0] == datetime(2015, 1, 1, 1, 30, 6)
    assert winds["lat"][0] == pytest.approx(36.96)
    assert winds["lon"][0] == pytest.approx(360.0 - 60.73)
    assert winds["pres"][0] == pytest.approx(32500.0)
    assert winds["quality"][0] == 2 and winds["height_method"][0] == 4
    assert winds["satellite_id"][0] == 257 and winds["subset"][0] == "NC005010"
    assert winds["wind_method"][0] == 1 and winds["wind_method_local"][0] is None
    assert winds["satellite_za"][0] == pytest.approx(45.43)
    assert winds["quality_indicators"][0] == [
        {"application": code, "confidence": value}
        for code, value in ((1, 99), (2, 62), (3, 97), (4, 44))
    ]
    assert winds["amv_quality_indicators"][0] is None


def test_goes_r_wind_reads_first_slot_and_amvqic_map():
    winds = _decode([_goes_r()], 30)
    assert winds["pres"][0] == pytest.approx(41780.0)
    assert winds["satellite_za"][0] == pytest.approx(70.43)
    assert winds["quality"][0] is None  # no SWQM in GOES-R BUFR
    assert winds["lat"][0] == pytest.approx(61.10906)
    # GNAPS 6 has a missing PCCF, so it has no entry.
    assert winds["amv_quality_indicators"][0] == [
        {"application": code, "confidence": value}
        for code, value in ((4, 89), (5, 89), (7, 61))
    ]
    assert winds["quality_indicators"][0] is None


def test_repeated_field_uses_first_slot_populated_in_the_message():
    def wind(first_prlc, second_prlc):
        return _stream(
            [
                ("SAID", 57),
                *_time_block(),
                ("CLAT", 10.0),
                ("CLON", 20.0),
                ("SWCM", 1),
                ("PRLC", first_prlc),
                ("WDIR", 90.0),
                ("WSPD", 5.0),
                ("PRLC", second_prlc),
            ]
        )

    # Slot 1 is populated in the message, so a wind missing it stays missing
    # instead of borrowing slot 2.
    winds = _decode([wind(50000.0, 60000.0), wind(None, 70000.0)], 67)
    assert winds["pres"][0] == pytest.approx(50000.0)
    assert math.isnan(winds["pres"][1])
    # Slot 1 empty for every wind: the template carries the field in slot 2.
    winds = _decode([wind(None, 60000.0), wind(None, 70000.0)], 67)
    assert winds["pres"] == pytest.approx([60000.0, 70000.0])


def test_every_producer_is_kept_and_unlocatable_winds_skipped():
    # INSAT-3DR (SAID 473) and deep-layer WV (SWCM 5) are not typed by GSI but
    # are real winds; the source keeps them.
    winds = _decode([_legacy_nesdis(said=473, swcm=5)], 24)
    assert winds["satellite_id"] == [473] and winds["wind_method"] == [5]
    late = (datetime(2015, 1, 1, 2), datetime(2015, 1, 1, 3))
    assert _decode([_legacy_nesdis()], 10, late)["time"] == []
    assert _decode([_legacy_nesdis(wspd=None)], 10)["time"] == []


def test_decode_satwnd_end_to_end(tmp_path, monkeypatch):
    local = tmp_path / "gdas.satwnd.bufr_d"
    local.write_bytes(b"satwnd-bytes")
    nesdis = _message_bytes(10)
    goes_r = _message_bytes(30)
    monkeypatch.setattr(
        utils_ncep,
        "_parse_prepbufr_messages",
        lambda data, *, silence_noise: (TABLE_B, {}, [(nesdis, 5), (goes_r, 5)]),
    )
    decoder = _Decoder(
        {
            nesdis: _Message([_legacy_nesdis(), _legacy_nesdis(swcm=3)]),
            goes_r: _Message([_goes_r()]),
        }
    )
    monkeypatch.setattr(utils_ncep, "_init_decode_worker", lambda tb, td: None)
    monkeypatch.setattr(utils_ncep, "_worker_decoder", decoder)
    df = utils_ncep.decode_satwnd(str(local), PLAN, *BOUNDS, decode_workers=1)
    # Three winds x (u, v), all u rows then all v rows, source order within each.
    assert len(df) == 6
    assert list(df.columns) == utils_ncep.NCEP_SATWND_PUBLIC_SCHEMA.names
    assert df["variable"].tolist() == ["u", "u", "u", "v", "v", "v"]
    assert df["subset"].tolist()[:3] == ["NC005010", "NC005010", "NC005030"]
    assert df["wind_method"].tolist()[:3] == [1, 3, 1]
    assert df["type"].isna().all() and df["station"].isna().all()
    assert (df["class"] == "SATWND").all()
    assert df["pres"].dtype == np.float32
    assert str(df["satellite_id"].dtype) == "uint16[pyarrow]"
    assert df.loc[df["subset"] == "NC005030", "quality"].isna().all()
    # Round-trips through parquet with the declared nested type.
    df.to_parquet(tmp_path / "winds.parquet")
    back = pd.read_parquet(tmp_path / "winds.parquet")
    assert back["quality_indicators"].iloc[0][0]["application"] == 1

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
    uri = source._build_uri("satwnd", datetime(2019, 12, 31, 18))
    assert uri.endswith("amv/merged/2019/12/bufr/gdas.20191231.t18z.satwnd.tm00.bufr_d")
    with pytest.raises(ValueError):
        source._build_uri("prepbufr", datetime(2024, 1, 1))
    tasks = source._create_tasks([datetime(2024, 1, 1, 6)], ["u", "v"])
    assert len(tasks) == 1 and tasks[0].route == "satwnd"
    assert set(tasks[0].var_plan) == {"u", "v"}
    assert tasks[0].var_plan["u"][0] == "u"
    assert NNJAObsSatwnd.available(datetime(1990, 6, 1))
    assert not NNJAObsSatwnd.available(datetime(1970, 6, 1))
    names = source.resolve_fields(["time", "quality_indicators"]).names
    assert names == ["time", "quality_indicators"]
