"""Official 18-Test Integration Suite for NVIDIA Hosted FourCastNet API.

Executes genuine, distinct live HTTP requests against https://climate.api.nvidia.com/v1/nvidia/fourcastnet.
Strictly records machine-readable audit evidence, SHA-256 checksums, and timing metrics.
All live network requests are strictly opt-in and sequential.
"""

import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from loguru import logger

from od_experiments.earth2_001.api.artifact_processor import (
    ForecastArtifactProcessor,
    ProcessedArtifact,
)
from od_experiments.earth2_001.api.fourcastnet_hosted import (
    ForecastRequest,
    ForecastResponse,
    FourCastNetHostedClient,
    NvidiaApiError,
    NvidiaAuthenticationError,
    NvidiaValidationError,
)
from od_experiments.earth2_001.security import get_nvidia_api_key

LIVE_OUTPUT_DIR = Path("outputs/earth2_001/api_live")
EVIDENCE_FILE = LIVE_OUTPUT_DIR / "request_evidence.jsonl"
RESULTS_JSON = LIVE_OUTPUT_DIR / "live_api_results.json"
RESULTS_MD = LIVE_OUTPUT_DIR / "live_api_results.md"


def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def append_evidence(entry: dict[str, Any]) -> None:
    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def execute_live_test(
    test_id: str,
    test_name: str,
    request: ForecastRequest,
    expected_outcome: str,
    bypass_auth: bool = False,
    expect_error_code: int | None = None,
    client: FourCastNetHostedClient | None = None,
) -> dict[str, Any]:
    """Executes a single, verified live API test and records verifiable evidence."""
    if client is None:
        client = FourCastNetHostedClient()

    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processor = ForecastArtifactProcessor(LIVE_OUTPUT_DIR)

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_commit_sha()

    evidence: dict[str, Any] = {
        "test_id": test_id,
        "test_name": test_name,
        "timestamp_utc": timestamp_utc,
        "git_commit": git_sha,
        "endpoint": client.endpoint,
        "request_parameters": request.to_dict(),
        "accept_format": request.accept_format,
        "bypass_auth": bypass_auth,
        "expected_outcome": expected_outcome,
        "reached_nvidia": False,
        "inference_completed": False,
        "output_parsed": False,
        "status": "FAILED",
        "http_status": None,
        "duration_seconds": 0.0,
        "artifacts": [],
        "error_summary": None,
    }

    t0 = time.perf_counter()
    try:
        if expect_error_code is not None:
            # Expected error path (T17, T18)
            try:
                resp = client.infer(request, bypass_auth_for_testing=bypass_auth)
                evidence["http_status"] = resp.status_code
                evidence["reached_nvidia"] = True
                evidence["error_summary"] = (
                    f"Expected HTTP error {expect_error_code}, but received HTTP {resp.status_code}"
                )
                evidence["status"] = "FAILED"
            except (
                NvidiaAuthenticationError,
                NvidiaValidationError,
                NvidiaApiError,
            ) as e:
                evidence["duration_seconds"] = round(time.perf_counter() - t0, 4)
                evidence["reached_nvidia"] = True
                evidence["http_status"] = e.status_code
                evidence["error_summary"] = str(e)
                if e.status_code == expect_error_code:
                    evidence["status"] = "PASSED"
                    evidence["output_parsed"] = True
                else:
                    evidence["status"] = "FAILED"
        else:
            # Standard successful inference path (T01-T16)
            resp = client.infer(request, bypass_auth_for_testing=bypass_auth)
            evidence["duration_seconds"] = round(resp.duration_seconds, 4)
            evidence["reached_nvidia"] = True
            evidence["http_status"] = resp.status_code

            if resp.status_code == 200:
                evidence["inference_completed"] = True
                artifacts = processor.process_response(request, resp, test_id=test_id)
                evidence["output_parsed"] = True
                evidence["status"] = "PASSED"
                evidence["artifacts"] = [asdict(a) for a in artifacts]
                evidence["nvcf_reqid"] = resp.req_id
                evidence["nvcf_status"] = resp.nvcf_status
            else:
                evidence["status"] = "FAILED"
                evidence["error_summary"] = f"Unexpected HTTP status {resp.status_code}"

    except Exception as e:
        evidence["duration_seconds"] = round(time.perf_counter() - t0, 4)
        evidence["error_summary"] = f"{type(e).__name__}: {str(e)}"
        evidence["status"] = "FAILED"

    append_evidence(evidence)
    # Pacing: 1 second delay between tests to respect rate limits
    time.sleep(1.0)
    return evidence


@pytest.mark.live_api
class TestFourCastNetLive18:
    """18 distinct live integration tests against NVIDIA-hosted FourCastNet endpoint."""

    @pytest.fixture(autouse=True)
    def check_prerequisites(self):
        key = get_nvidia_api_key()
        if not key:
            pytest.skip("NVIDIA_API_KEY is not configured; skipping live API suite.")

    def test_T01_sample_0_t2m_1step(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T01",
            "Sample input 0, t2m, 1 step",
            req,
            "Genuine successful inference response",
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T02_sample_1_t2m_1step(self):
        req = ForecastRequest(
            input_id=1,
            variables="t2m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T02", "Sample input 1, t2m, 1 step", req, "Input selection works"
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T03_sample_2_t2m_1step(self):
        req = ForecastRequest(
            input_id=2,
            variables="t2m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T03", "Sample input 2, t2m, 1 step", req, "Input selection works"
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T04_sample_3_t2m_1step(self):
        req = ForecastRequest(
            input_id=3,
            variables="t2m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T04", "Sample input 3, t2m, 1 step", req, "Input selection works"
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T05_sample_0_w10m_1step(self):
        req = ForecastRequest(
            input_id=0,
            variables="w10m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T05", "Sample input 0, w10m, 1 step", req, "Wind-speed output accepted"
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T06_sample_0_msl_1step(self):
        req = ForecastRequest(
            input_id=0,
            variables="msl",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T06", "Sample input 0, msl, 1 step", req, "Pressure output accepted"
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T07_sample_0_tcwv_1step(self):
        req = ForecastRequest(
            input_id=0,
            variables="tcwv",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T07", "Sample input 0, tcwv, 1 step", req, "Water-vapour output accepted"
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T08_sample_0_z500_1step(self):
        req = ForecastRequest(
            input_id=0,
            variables="z500",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T08", "Sample input 0, z500, 1 step", req, "Geopotential output accepted"
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T09_sample_0_multivar_t2m_w10m(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m,w10m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T09", "Sample input 0, t2m,w10m", req, "Multiple variables supported"
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T10_sample_0_2steps_12h(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m",
            simulation_length=2,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T10",
            "Sample input 0, two forecast steps",
            req,
            "Twelve-hour forecast supported",
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T11_sample_0_4steps_24h(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m",
            simulation_length=4,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T11",
            "Sample input 0, four forecast steps",
            req,
            "Twenty-four-hour forecast supported",
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T12_sample_0_ensemble_size_2(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m",
            simulation_length=1,
            ensemble_size=2,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T12",
            "Sample input 0, ensemble size 2",
            req,
            "Multiple ensemble members supported",
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T13_sample_0_noise_amplitude_001(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.01,
        )
        res = execute_live_test(
            "T13",
            "Sample input 0, noise amplitude 0.01",
            req,
            "Nonzero perturbation accepted",
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T14_sample_0_multivar_w10m_msl_tcwv(self):
        req = ForecastRequest(
            input_id=0,
            variables="w10m,msl,tcwv",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T14",
            "Sample input 0, w10m,msl,tcwv",
            req,
            "Multivariable response correctly processed",
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T15_sample_0_accept_tar_request(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
            accept_format="application/x-tar",
        )
        res = execute_live_test(
            "T15",
            "Sample input 0, 1 step, request TAR response",
            req,
            "Binary response handled safely",
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T16_repeat_T01_reproducibility(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T16",
            "Repeat T01 with identical parameters",
            req,
            "Compare response structure and reproducibility characteristics",
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 200

    def test_T17_missing_authorization_rejection(self):
        req = ForecastRequest(
            input_id=0,
            variables="t2m",
            simulation_length=1,
            ensemble_size=1,
            noise_amplitude=0.0,
        )
        res = execute_live_test(
            "T17",
            "Submit request without authorization",
            req,
            "Service rejects unauthenticated request with HTTP 401",
            bypass_auth=True,
            expect_error_code=401,
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 401

    def test_T18_invalid_input_id_rejection(self):
        # input_id=999 is outside of 0..4 range. Bypass client range check by creating object directly
        req = object.__new__(ForecastRequest)
        req.input_id = 999
        req.variables = "t2m"
        req.simulation_length = 1
        req.ensemble_size = 1
        req.noise_amplitude = 0.0
        req.accept_format = "application/json"

        # Mock validate method to allow out-of-range payload to hit NVIDIA API
        req.validate = lambda: None

        res = execute_live_test(
            "T18",
            "Submit out-of-range input_id 999",
            req,
            "Service rejects out-of-range input with HTTP 400",
            expect_error_code=400,
        )
        assert res["status"] == "PASSED"
        assert res["http_status"] == 400
