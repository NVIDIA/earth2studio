import pytest
from pathlib import Path
from od_experiments.earth2_001.config import load_config, validate_config
from od_experiments.earth2_001.security import scrub_string, scrub_dict, sanitize_env


def test_default_config_loading():
    cfg = load_config()
    assert cfg["experiment_id"] == "OD-EARTH2-001"
    assert cfg["model"]["name"] == "persistence"
    assert "t2m" in cfg["simulation"]["variables"]
    valid, errors = validate_config(cfg)
    assert valid
    assert len(errors) == 0


def test_invalid_config():
    valid, errors = validate_config({"invalid": 123})
    assert not valid
    assert any("experiment_id" in e for e in errors)


def test_secret_scrubber_redaction():
    text = "Authorization: Bearer mysecrettoken123456789012345"
    assert "[REDACTED]" in scrub_string(text)
    assert "mysecrettoken123456789012345" not in scrub_string(text)

    d = {
        "api_key": "secret12345",
        "nested": {"AWS_SECRET_ACCESS_KEY": "supersecretkey"},
        "public": "regular_value",
    }
    scrubbed = scrub_dict(d)
    assert scrubbed["api_key"] == "[REDACTED]"
    assert scrubbed["nested"]["AWS_SECRET_ACCESS_KEY"] == "[REDACTED]"
    assert scrubbed["public"] == "regular_value"
