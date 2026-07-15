# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for shared NCEP conventional format adapters."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import earth2studio.data.nnja as nnja_data
from earth2studio.data import NNJAObsConv, NomadsGDASObsConv, utils_ncep
from earth2studio.data.utils_bufr import OBS_TOB, OBS_TQM
from earth2studio.lexicon import GDASObsConvLexicon, NNJAObsConvLexicon

pytest.importorskip("pybufrkit", reason="pybufrkit not installed")


def _message(desc_values: list[tuple[int, object]]) -> SimpleNamespace:
    descriptors = [
        SimpleNamespace(id=descriptor_id) for descriptor_id, _ in desc_values
    ]
    values = [value for _, value in desc_values]
    return SimpleNamespace(
        edition=SimpleNamespace(value=4),
        year=SimpleNamespace(value=2024),
        month=SimpleNamespace(value=1),
        day=SimpleNamespace(value=1),
        hour=SimpleNamespace(value=0),
        minute=SimpleNamespace(value=0),
        second=SimpleNamespace(value=0),
        n_subsets=SimpleNamespace(value=1),
        template_data=SimpleNamespace(
            value=SimpleNamespace(
                decoded_descriptors_all_subsets=[descriptors],
                decoded_values_all_subsets=[values],
            )
        ),
    )


class _Decoder:
    def __init__(self, expected_message: bytes, message: SimpleNamespace) -> None:
        self.expected_message = expected_message
        self.message = message

    def process(self, message: bytes) -> SimpleNamespace:
        assert message == self.expected_message
        return self.message


def _prepbufr_plan(lexicon: type, variable: str) -> dict:
    source_key, modifier = lexicon.get_item(variable)
    _, separator, key = source_key.partition("::")
    return {variable: (key if separator else source_key, modifier)}


def _gpsro_plan(lexicon: type, variable: str, descriptor: int) -> dict:
    _source_key, modifier = lexicon.get_item(variable)
    return {variable: (descriptor, modifier)}


def _modifiers(lexicon: type, *variables: str) -> dict:
    return {variable: lexicon.get_item(variable)[1] for variable in variables}


def test_same_local_prepbufr_bytes_are_adapter_exact(tmp_path, monkeypatch):
    local_path = tmp_path / "same.prepbufr.nr"
    local_path.write_bytes(b"same-local-prepbufr-bytes")
    message_bytes = b"prepbufr-data-message"
    decoded = _message(
        [
            (utils_ncep.HDR_SID, b"72469   "),
            (utils_ncep.HDR_XOB, -105.25),
            (utils_ncep.HDR_YOB, 39.75),
            (utils_ncep.HDR_DHR, 0.25),
            (utils_ncep.HDR_ELV, 1_655.0),
            (utils_ncep.HDR_TYP, 120),
            (utils_ncep.OBS_CAT, 1),
            (utils_ncep.OBS_HRDR, 0.25001),
            (utils_ncep.OBS_XDR, -104.5),
            (utils_ncep.OBS_YDR, 40.25),
            (utils_ncep.OBS_POB, 850.0),
            (utils_ncep.OBS_PQM, 4),
            (utils_ncep.OBS_ZOB, 1_500.0),
            (OBS_TOB, 5.5),
            (OBS_TQM, 2),
            # Later event-stack values must not become physical levels.
            (utils_ncep.OBS_POB, 849.0),
            (OBS_TOB, 6.5),
        ]
    )
    table_b = {
        utils_ncep.HDR_DHR: ("DHR", "HR", 5, 0, 0),
        utils_ncep.OBS_HRDR: ("HRDR", "HR", 5, 0, 0),
    }

    def parse(file_data: bytes, *, silence_noise: bool):
        assert file_data == b"same-local-prepbufr-bytes"
        assert silence_noise
        return table_b, {}, [(message_bytes, 102)]

    monkeypatch.setattr(utils_ncep, "_parse_prepbufr_messages", parse)
    monkeypatch.setattr(
        utils_ncep,
        "_create_decoder",
        lambda _table_b, _table_d: _Decoder(message_bytes, decoded),
    )

    nnja = NNJAObsConv(cache=False, verbose=False, decode_workers=1)
    gdas = NomadsGDASObsConv(cache=False, verbose=False, decode_workers=1)
    assert isinstance(nnja._prepbufr_adapter, utils_ncep._NCEPPrepbufrAdapter)
    assert isinstance(gdas._prepbufr_adapter, utils_ncep._NCEPPrepbufrAdapter)

    bounds = (datetime(2024, 1, 1), datetime(2024, 1, 1, 1))
    nnja_df = nnja._prepbufr_adapter.decode_file(
        str(local_path),
        _prepbufr_plan(NNJAObsConvLexicon, "t"),
        *bounds,
    )
    gdas_df = gdas._prepbufr_adapter.decode_file(
        str(local_path),
        _prepbufr_plan(GDASObsConvLexicon, "t"),
        *bounds,
    )

    pd.testing.assert_frame_equal(nnja_df, gdas_df, check_exact=True)
    assert len(nnja_df) == 1
    assert nnja_df.loc[0, "observation"] == pytest.approx(278.65)
    assert nnja_df.loc[0, "time"] == pd.Timestamp("2024-01-01 00:15:00.036")
    assert nnja_df.loc[0, "lat"] == pytest.approx(40.25)
    assert nnja_df.loc[0, "lon"] == pytest.approx(255.5)
    assert nnja_df.loc[0, "elev"] == pytest.approx(1_500.0)
    assert nnja_df.loc[0, "level_cat"] == 1
    assert nnja_df.loc[0, "pressure_quality"] == 4
    for schema in (NNJAObsConv.SCHEMA, NomadsGDASObsConv.SCHEMA):
        assert schema.field("level_cat").type == pa.uint16()
        assert schema.field("pressure_quality").type == pa.uint16()

    nnja_task = nnja_data._NNJAConvTask(
        s3_uri="s3://example/same.prepbufr.nr",
        datetime_file=datetime(2024, 1, 1),
        datetime_min=bounds[0],
        datetime_max=bounds[1],
        var_plan=_prepbufr_plan(NNJAObsConvLexicon, "t"),
    )
    nnja_public = nnja._decode_prepbufr_file(str(local_path), nnja_task)
    gdas_public = gdas._decode_prepbufr(str(local_path), ["t"], *bounds)
    pd.testing.assert_frame_equal(nnja_public, gdas_public, check_exact=True)


def test_same_local_gpsro_bytes_preserve_default_product(tmp_path, monkeypatch):
    local_path = tmp_path / "same.gpsro.bufr"
    local_path.write_bytes(b"same-local-gpsro-bytes")
    message_bytes = b"gpsro-data-message"
    decoded = _message(
        [
            (utils_ncep.GPSRO_SAID, 3),
            (utils_ncep.GPSRO_PTID, 27),
            (utils_ncep.GPSRO_QFRO, 12),
            (utils_ncep.GPSRO_ELRC, 6_371_000.0),
            (utils_ncep.GPSRO_LAT, -10.5),
            (utils_ncep.GPSRO_LON, -70.25),
            (utils_ncep.GPSRO_YEAR, 2024),
            (utils_ncep.GPSRO_MONTH, 1),
            (utils_ncep.GPSRO_DAY, 1),
            (utils_ncep.GPSRO_HOUR, 0),
            (utils_ncep.GPSRO_MIN, 30),
            (utils_ncep.GPSRO_SEC, 15.25),
            (utils_ncep.GPSRO_LAT, -9.75),
            (utils_ncep.GPSRO_LON, -69.5),
            (utils_ncep.GPSRO_MEFR, 1_500_000_000.0),
            (utils_ncep.GPSRO_IMPP, 6_373_000.0),
            (utils_ncep.GPSRO_BNDA, 0.00999),
            (utils_ncep.GPSRO_BNDA, 0.00888),
            (utils_ncep.GPSRO_MEFR, 0.0),
            (utils_ncep.GPSRO_IMPP, 6_373_000.0),
            (utils_ncep.GPSRO_BNDA, 0.00123),
            (utils_ncep.GPSRO_BNDA, 0.00045),
        ]
    )

    def parse(file_data: bytes, *, silence_noise: bool):
        assert file_data == b"same-local-gpsro-bytes"
        assert silence_noise
        return {}, {}, [(message_bytes, 0)]

    monkeypatch.setattr(utils_ncep, "_parse_prepbufr_messages", parse)
    monkeypatch.setattr(
        utils_ncep,
        "_create_decoder",
        lambda _table_b, _table_d: _Decoder(message_bytes, decoded),
    )

    nnja = NNJAObsConv(cache=False, verbose=False, decode_workers=1)
    gdas = NomadsGDASObsConv(cache=False, verbose=False, decode_workers=1)
    assert isinstance(nnja._gpsro_adapter, utils_ncep._NCEPGpsroAdapter)
    assert isinstance(gdas._gpsro_adapter, utils_ncep._NCEPGpsroAdapter)

    bounds = (datetime(2024, 1, 1), datetime(2024, 1, 1, 1))
    nnja_df = nnja._gpsro_adapter.decode_file(
        str(local_path),
        _gpsro_plan(NNJAObsConvLexicon, "gps", utils_ncep.GPSRO_BNDA),
        *bounds,
    )
    gdas_df = gdas._gpsro_adapter.decode_file(
        str(local_path),
        _gpsro_plan(GDASObsConvLexicon, "gps", utils_ncep.GPSRO_BNDA),
        *bounds,
    )

    pd.testing.assert_frame_equal(nnja_df, gdas_df, check_exact=True)
    assert nnja_df["variable"].tolist() == ["gps"]
    assert nnja_df["observation"].tolist() == pytest.approx([0.00123])
    assert nnja_df.loc[0, "time"] == pd.Timestamp("2024-01-01 00:30:15.250")
    assert pd.isna(nnja_df.loc[0, "pres"])
    assert nnja_df.loc[0, "elev"] == pytest.approx(2_000.0)

    nnja_task = nnja_data._NNJAGpsRoTask(
        s3_uri="s3://example/same.gpsro.bufr",
        datetime_file=datetime(2024, 1, 1),
        datetime_min=bounds[0],
        datetime_max=bounds[1],
        var_plan=_gpsro_plan(NNJAObsConvLexicon, "gps", utils_ncep.GPSRO_BNDA),
    )
    nnja_public = nnja._decode_gpsro_file(str(local_path), nnja_task)
    gdas_public = gdas._decode_gpsro(str(local_path), ["gps"], *bounds)
    pd.testing.assert_frame_equal(nnja_public, gdas_public, check_exact=True)


def test_cat_delimits_levels_and_repeated_pob_remains_an_event_slot():
    desc_values = [
        (utils_ncep.HDR_SID, b"72469   "),
        (utils_ncep.HDR_XOB, -105.25),
        (utils_ncep.HDR_YOB, 39.75),
        (utils_ncep.HDR_DHR, 0.0),
        (utils_ncep.HDR_TYP, 120),
        (utils_ncep.OBS_CAT, 1),
        (utils_ncep.OBS_POB, 1000.0),
        (utils_ncep.OBS_PQM, 1),
        (utils_ncep.OBS_ZOB, 100.0),
        (OBS_TOB, 1.0),
        (utils_ncep.OBS_POB, 999.0),
        (utils_ncep.OBS_PQM, 6),
        (utils_ncep.OBS_ZOB, 200.0),
        (OBS_TOB, 2.0),
        (utils_ncep.OBS_CAT, 2),
        (utils_ncep.OBS_POB, 800.0),
        (OBS_TOB, 3.0),
        (utils_ncep.OBS_CAT, 3),
        (utils_ncep.OBS_POB, 700.0),
        (OBS_TOB, None),
        # A missing first event still owns the slot; do not promote history.
        (OBS_TOB, 4.0),
    ]
    descriptors = [SimpleNamespace(id=descriptor) for descriptor, _ in desc_values]
    values = [value for _, value in desc_values]

    rows = utils_ncep._extract_prepbufr_subset(
        descriptors,
        values,
        datetime(2024, 1, 1),
        "ADPUPA",
        [("t", "TOB")],
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 1),
    )

    assert [row["observation"] for row in rows] == pytest.approx([1.0, 3.0])
    assert [row["pres"] for row in rows] == pytest.approx([1000.0, 800.0])
    assert rows[0]["elev"] == pytest.approx(100.0)
    assert rows[0]["pressure_quality"] == 1


def test_aircraft_profile_mapping_is_variable_specific():
    frame = pd.DataFrame(
        {
            "variable": ["t", "q", "pres", "u", "v"],
            "type": pd.Series([330, 331, 333, 334, 335], dtype="uint16[pyarrow]"),
        }
    )

    mapped = utils_ncep.map_aircraft_profile_types(frame)

    assert mapped["type"].tolist() == [130, 131, 133, 234, 235]


def test_public_facades_share_canonical_schema():
    expected = utils_ncep.NCEP_CONVENTIONAL_PUBLIC_SCHEMA

    assert NomadsGDASObsConv.SCHEMA is expected
    assert NNJAObsConv.SCHEMA is expected
    assert expected.field("level_cat").type == pa.uint16()
    assert expected.field("pressure_quality").type == pa.uint16()


def test_empty_public_facades_use_shared_schema_dtypes():
    gdas = NomadsGDASObsConv(cache=False, verbose=False, decode_workers=1)
    nnja = NNJAObsConv(cache=False, verbose=False, decode_workers=1)

    gdas_empty = gdas._compile_dataframe([], NomadsGDASObsConv.SCHEMA)
    nnja_empty = nnja._compile_dataframe([], NNJAObsConv.SCHEMA)

    assert gdas_empty.dtypes.astype(str).to_dict() == {
        name: str(dtype)
        for name, dtype in utils_ncep._empty_dataframe(
            NomadsGDASObsConv.SCHEMA
        ).dtypes.items()
    }
    assert nnja_empty.dtypes.astype(str).to_dict() == {
        name: str(dtype)
        for name, dtype in utils_ncep._empty_dataframe(
            NNJAObsConv.SCHEMA
        ).dtypes.items()
    }


def test_extract_gpsro_subset_bending_angle_rows_and_metadata():
    subset_stream = [
        (utils_ncep.GPSRO_SAID, 3),
        (utils_ncep.GPSRO_PTID, 27),
        (utils_ncep.GPSRO_QFRO, 12),
        (utils_ncep.GPSRO_ELRC, 6_371_000.0),
        (utils_ncep.GPSRO_LAT, -10.5),
        (utils_ncep.GPSRO_LON, -70.25),
        (utils_ncep.GPSRO_YEAR, 2024),
        (utils_ncep.GPSRO_MONTH, 1),
        (utils_ncep.GPSRO_DAY, 1),
        (utils_ncep.GPSRO_HOUR, 0),
        (utils_ncep.GPSRO_MIN, 30),
        (utils_ncep.GPSRO_SEC, 15.25),
        (utils_ncep.GPSRO_LAT, -9.75),
        (utils_ncep.GPSRO_LON, -69.5),
        (utils_ncep.GPSRO_MEFR, 1_500_000_000.0),
        (utils_ncep.GPSRO_IMPP, 6_373_000.0),
        (utils_ncep.GPSRO_BNDA, 0.00999),
        (utils_ncep.GPSRO_BNDA, 0.00888),
        (utils_ncep.GPSRO_MEFR, 0.0),
        (utils_ncep.GPSRO_IMPP, 6_373_000.0),
        (utils_ncep.GPSRO_BNDA, 0.00123),
        (utils_ncep.GPSRO_BNDA, 0.00045),
    ]
    descs = [SimpleNamespace(id=descriptor_id) for descriptor_id, _ in subset_stream]
    values = [value for _, value in subset_stream]

    rows = utils_ncep._extract_gpsro_subset(
        descs,
        values,
        {utils_ncep.GPSRO_BNDA: "gps"},
        datetime(2024, 1, 1, 0),
        datetime(2024, 1, 1, 1),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["time"] == datetime(2024, 1, 1, 0, 30, 15, 250000)
    assert row["lat"] == pytest.approx(np.float32(-9.75))
    assert row["lon"] == pytest.approx(np.float32(290.5))
    assert row["pres"] is None
    assert row["elev"] == pytest.approx(np.float32(2_000.0))
    assert row["type"] == np.uint16(3)
    assert row["class"] == "GPSRO"
    assert row["station"] == "00030027"
    assert row["quality"] == np.uint16(12)
    assert row["observation"] == pytest.approx(np.float32(0.00123))
    assert row["variable"] == "gps"


def test_extract_gpsro_subset_missing_bending_angle_does_not_emit_error():
    subset_stream = [
        (utils_ncep.GPSRO_SAID, 3),
        (utils_ncep.GPSRO_PTID, 27),
        (utils_ncep.GPSRO_QFRO, 12),
        (utils_ncep.GPSRO_ELRC, 6_371_000.0),
        (utils_ncep.GPSRO_LAT, -10.5),
        (utils_ncep.GPSRO_LON, -70.25),
        (utils_ncep.GPSRO_YEAR, 2024),
        (utils_ncep.GPSRO_MONTH, 1),
        (utils_ncep.GPSRO_DAY, 1),
        (utils_ncep.GPSRO_HOUR, 0),
        (utils_ncep.GPSRO_MIN, 30),
        (utils_ncep.GPSRO_SEC, 15.25),
        (utils_ncep.GPSRO_MEFR, 0.0),
        (utils_ncep.GPSRO_IMPP, 6_373_000.0),
        (utils_ncep.GPSRO_BNDA, None),
        (utils_ncep.GPSRO_BNDA, 0.006),
    ]
    descs = [SimpleNamespace(id=descriptor_id) for descriptor_id, _ in subset_stream]
    values = [value for _, value in subset_stream]

    rows = utils_ncep._extract_gpsro_subset(
        descs,
        values,
        {utils_ncep.GPSRO_BNDA: "gps"},
        datetime(2024, 1, 1, 0),
        datetime(2024, 1, 1, 1),
    )

    assert rows == []


def test_extract_gpsro_subset_missing_level_lat_lon_does_not_reuse_stale_values():
    subset_stream = [
        (utils_ncep.GPSRO_SAID, 3),
        (utils_ncep.GPSRO_PTID, 27),
        (utils_ncep.GPSRO_QFRO, 12),
        (utils_ncep.GPSRO_ELRC, 6_371_000.0),
        (utils_ncep.GPSRO_LAT, -10.5),
        (utils_ncep.GPSRO_LON, -70.25),
        (utils_ncep.GPSRO_YEAR, 2024),
        (utils_ncep.GPSRO_MONTH, 1),
        (utils_ncep.GPSRO_DAY, 1),
        (utils_ncep.GPSRO_HOUR, 0),
        (utils_ncep.GPSRO_MIN, 30),
        (utils_ncep.GPSRO_SEC, 15.25),
        (utils_ncep.GPSRO_LAT, -9.75),
        (utils_ncep.GPSRO_LON, -69.5),
        (utils_ncep.GPSRO_MEFR, 0.0),
        (utils_ncep.GPSRO_IMPP, 6_373_000.0),
        (utils_ncep.GPSRO_BNDA, 0.00123),
        (utils_ncep.GPSRO_BNDA, 0.00045),
        # A missing per-level latitude must clear state so the next observation
        # does not reuse -9.75 from the previous bending-angle block.
        (utils_ncep.GPSRO_LAT, None),
        (utils_ncep.GPSRO_LON, -68.5),
        (utils_ncep.GPSRO_MEFR, 0.0),
        (utils_ncep.GPSRO_IMPP, 6_374_000.0),
        (utils_ncep.GPSRO_BNDA, 0.00234),
        (utils_ncep.GPSRO_BNDA, 0.00056),
    ]
    descs = [SimpleNamespace(id=descriptor_id) for descriptor_id, _ in subset_stream]
    values = [value for _, value in subset_stream]

    rows = utils_ncep._extract_gpsro_subset(
        descs,
        values,
        {utils_ncep.GPSRO_BNDA: "gps"},
        datetime(2024, 1, 1, 0),
        datetime(2024, 1, 1, 1),
    )

    assert len(rows) == 1
    assert rows[0]["lat"] == pytest.approx(np.float32(-9.75))
    assert rows[0]["lon"] == pytest.approx(np.float32(290.5))
    assert rows[0]["observation"] == pytest.approx(np.float32(0.00123))


def test_finalize_rows_filters_and_converts_pressure():
    result = utils_ncep._finalize_rows(
        [], _modifiers(NNJAObsConvLexicon, "t"), convert_pres_mb_to_pa=True
    )
    assert result.empty

    # Rows whose variable is not requested are dropped.
    rows = [
        {
            "time": datetime(2024, 1, 1, 0),
            "lat": 40.0,
            "lon": 250.0,
            "pres": 850.0,
            "elev": None,
            "type": 120,
            "class": "ADPUPA",
            "station": "72469",
            "station_elev": 1000.0,
            "observation": 273.15,
            "variable": "other_var",
        }
    ]
    result = utils_ncep._finalize_rows(
        rows, _modifiers(NNJAObsConvLexicon, "t"), convert_pres_mb_to_pa=True
    )
    assert result.empty

    rows_match = [{**rows[0], "variable": "t"}]
    result = utils_ncep._finalize_rows(
        rows_match, _modifiers(NNJAObsConvLexicon, "t"), convert_pres_mb_to_pa=True
    )
    assert len(result) == 1
    # 850 mb -> 85000 Pa
    assert result["pres"].iloc[0] == pytest.approx(85000.0)

    result_no_conv = utils_ncep._finalize_rows(
        rows_match, _modifiers(NNJAObsConvLexicon, "t"), convert_pres_mb_to_pa=False
    )
    assert result_no_conv["pres"].iloc[0] == pytest.approx(850.0)


def test_finalize_rows_gpsro_preserves_pressure_and_elevation_units():
    rows = [
        {
            "time": datetime(2024, 1, 1, 0),
            "lat": 40.0,
            "lon": 250.0,
            "pres": 25_000.0,
            "elev": 2_000.0,
            "type": 3,
            "class": "GPSRO",
            "station": "00030027",
            "station_elev": None,
            "quality": 12,
            "observation": 0.00123,
            "variable": "gps",
        }
    ]

    result = utils_ncep._finalize_rows(
        rows, _modifiers(NNJAObsConvLexicon, "gps"), convert_pres_mb_to_pa=False
    )

    assert list(result.columns) == list(
        utils_ncep.NCEP_CONVENTIONAL_PUBLIC_SCHEMA.names
    )
    assert len(result) == 1
    assert result["pres"].iloc[0] == pytest.approx(25_000.0)
    assert result["elev"].iloc[0] == pytest.approx(2_000.0)
    assert result["type"].iloc[0] == 3
    assert result["quality"].iloc[0] == 12
    assert result["observation"].iloc[0] == pytest.approx(0.00123)
    assert result["variable"].iloc[0] == "gps"


def test_finalize_rows_pres_filters_level_cat():
    base_row = {
        "time": datetime(2024, 1, 1, 0),
        "lat": 40.0,
        "lon": 250.0,
        "pres": 850.0,
        "elev": None,
        "type": 120,
        "class": "ADPUPA",
        "station": "72469",
        "station_elev": 1000.0,
        "quality": 2,
        "variable": "pres",
    }
    rows = [
        {**base_row, "observation": 1000.0, "level_cat": 0},
        {**base_row, "observation": 850.0, "level_cat": 1},
    ]

    result = utils_ncep._finalize_rows(
        rows, _modifiers(NNJAObsConvLexicon, "pres"), convert_pres_mb_to_pa=True
    )

    assert len(result) == 1
    assert result["observation"].iloc[0] == pytest.approx(100000.0)
    assert result["pres"].iloc[0] == pytest.approx(85000.0)
    assert result["level_cat"].iloc[0] == 0


def test_finalize_rows_adds_missing_columns():
    rows = [
        {
            "time": datetime(2024, 1, 1, 0),
            "lat": 40.0,
            "lon": 250.0,
            "pres": 850.0,
            "type": 120,
            "class": "ADPUPA",
            "observation": 273.15,
            "variable": "t",
        }
    ]
    result = utils_ncep._finalize_rows(
        rows, _modifiers(NNJAObsConvLexicon, "t"), convert_pres_mb_to_pa=True
    )

    assert list(result.columns) == list(
        utils_ncep.NCEP_CONVENTIONAL_PUBLIC_SCHEMA.names
    )
    assert pd.isna(result["elev"].iloc[0])
    assert result["station"].iloc[0] is None
    assert pd.isna(result["station_elev"].iloc[0])
