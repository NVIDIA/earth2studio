"""Unit tests for FourCastNet hosted API client and artifact processor."""

import pytest

from od_experiments.earth2_001.api.artifact_processor import (
    ForecastArtifactProcessor,
    calculate_sha256,
)
from od_experiments.earth2_001.api.fourcastnet_hosted import (
    ForecastRequest,
    ForecastResponse,
    FourCastNetHostedClient,
    NvidiaValidationError,
)


def test_forecast_request_validation_valid():
    req = ForecastRequest(
        input_id=0,
        variables="t2m",
        simulation_length=1,
        ensemble_size=1,
        noise_amplitude=0.0,
    )
    req.validate()
    d = req.to_dict()
    assert d["input_id"] == 0
    assert d["variables"] == "t2m"


def test_forecast_request_invalid_input_id():
    with pytest.raises(NvidiaValidationError):
        req = ForecastRequest(input_id=-1, variables="t2m")
        req.validate()

    with pytest.raises(NvidiaValidationError):
        req = ForecastRequest(input_id=5, variables="t2m")
        req.validate()


def test_forecast_request_invalid_variable():
    with pytest.raises(NvidiaValidationError):
        req = ForecastRequest(input_id=0, variables="invalid_var")
        req.validate()


def test_forecast_request_multiple_variables_valid():
    req = ForecastRequest(input_id=0, variables="w10m, t2m, msl")
    req.validate()
    assert req.simulation_length == 1


def test_artifact_processor_saves_json_response(tmp_path):
    proc = ForecastArtifactProcessor(tmp_path)
    req = ForecastRequest(input_id=0, variables="t2m")
    resp = ForecastResponse(
        status_code=200,
        req_id="test-req-123",
        nvcf_status="fulfilled",
        content_type="application/json",
        raw_data=b'{"message":"Large asset written"}',
        json_payload={"message": "Large asset written"},
        duration_seconds=1.23,
    )
    artifacts = proc.process_response(req, resp, test_id="T01")
    assert len(artifacts) >= 1
    assert artifacts[0].artifact_type == "json_response"
    assert (tmp_path / "t01_response.json").exists()


def test_calculate_sha256():
    data = b"origin_datum_verification"
    digest = calculate_sha256(data)
    assert len(digest) == 64
    assert digest == calculate_sha256(data)
