import json
import sys
from typing import Any, Dict, Optional

import earth2studio
from earth2studio.data import GFS, Random
from earth2studio.models.px import DLWP, FCN, Persistence


def inspect_components(
    model_name: str | None = None,
    dataset_name: str | None = None,
    as_json: bool = False,
) -> dict[str, Any]:
    """Inspects available Earth2Studio models, datasets, and variables."""
    info = {
        "earth2studio_version": getattr(earth2studio, "__version__", "0.20.0a0"),
        "supported_models": {
            "persistence": {
                "type": "prognostic_baseline",
                "description": "Zero-parameter baseline holding atmospheric state constant.",
                "hardware_requirement": "CPU compatible (minimal memory)",
                "zero_cost": True,
                "variables": "Pass-through (all input variables supported)",
            },
            "dlwp": {
                "type": "prognostic_neural",
                "description": "Deep Learning Weather Prediction 1.4-degree model.",
                "hardware_requirement": "CPU / CUDA",
                "zero_cost": True,
                "weights_source": "Public NGC checkpoint",
                "variables": ["t2m", "t850", "z1000", "z700", "z500", "z300"],
            },
            "fourcastnet": {
                "type": "prognostic_neural",
                "description": "Adaptive Fourier Neural Operator global model (0.25 deg).",
                "hardware_requirement": "CUDA GPU (min 16GB VRAM) or Hosted API",
                "zero_cost": True,
                "variables": [
                    "t2m",
                    "u10m",
                    "v10m",
                    "msl",
                    "u500",
                    "v500",
                    "z500",
                    "t500",
                ],
            },
        },
        "supported_data_sources": {
            "gfs": {
                "description": "NOAA Global Forecast System historical analyses.",
                "transport": "Anonymous AWS S3 (noaa-gfs-bdp-pds)",
                "cost": "$0.00 (Public AWS Open Data)",
                "requires_api_key": False,
            },
            "random": {
                "description": "Synthetic random spatial field generator for offline testing.",
                "transport": "In-memory / Offline",
                "cost": "$0.00",
                "requires_api_key": False,
            },
        },
    }

    if as_json:
        print(json.dumps(info, indent=2))
    else:
        print("=" * 70)
        print("          ORIGIN DATUM EARTH-2 INSPECTION REGISTRY")
        print("=" * 70)
        print(f"Earth2Studio Version: {info['earth2studio_version']}\n")
        print("Available Model Architectures:")
        for name, details in info["supported_models"].items():
            print(f"  • {name.upper()}:")
            print(f"      Type: {details['type']}")
            print(f"      Description: {details['description']}")
            print(f"      Hardware: {details['hardware_requirement']}")
            print(f"      Zero Cost: {details['zero_cost']}")
        print("\nAvailable Data Sources:")
        for name, details in info["supported_data_sources"].items():
            print(f"  • {name.upper()}:")
            print(f"      Description: {details['description']}")
            print(f"      Transport: {details['transport']}")
            print(f"      Cost: {details['cost']}")
        print("=" * 70)

    return info


if __name__ == "__main__":
    as_json = "--json" in sys.argv
    inspect_components(as_json=as_json)
