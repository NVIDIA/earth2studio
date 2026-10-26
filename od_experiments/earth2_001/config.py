import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from od_experiments.earth2_001.security import scrub_dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment_id": "OD-EARTH2-001",
    "description": "Reproducible baseline weather forecast and scientific verification for EO-OBSCURA.",
    "model": {
        "name": "persistence",
        "checkpoint": None,
        "device": "cpu",
    },
    "data_source": {
        "name": "gfs",
        "transport": "s3",
        "bucket": "noaa-gfs-bdp-pds",
        "skip_signature": True,
    },
    "simulation": {
        "initial_time": "2024-01-01T00:00:00",
        "lead_time_hours": 6,
        "n_steps": 1,
        "variables": ["t2m", "u10m", "v10m"],
    },
    "regions": {
        "primary": "Europe",
        "secondary": "Japan",
    },
    "output": {
        "directory": "outputs/earth2_001",
        "format": "zarr",
        "generate_plots": True,
    },
    "validation": {
        "metrics": ["mae", "rmse", "mean_bias"],
        "baseline_model": "persistence",
    },
}


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads configuration from a YAML file, falling back to defaults."""
    if not config_path or not Path(config_path).exists():
        return DEFAULT_CONFIG.copy()

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = DEFAULT_CONFIG.copy()
    for k, v in user_config.items():
        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
            config[k] = {**config[k], **v}
        else:
            config[k] = v

    return config


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates configuration parameters."""
    errors = []
    if "experiment_id" not in config:
        errors.append("Missing 'experiment_id'")
    if "model" not in config or "name" not in config["model"]:
        errors.append("Missing 'model.name'")
    if "simulation" not in config or "variables" not in config["simulation"]:
        errors.append("Missing 'simulation.variables'")
    return len(errors) == 0, errors
