import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SENSITIVE_PATTERNS = [
    re.compile(
        r"(?i)(api[_-]?key|access[_-]?key|secret[_-]?key|auth[_-]?token|password|credential)[\"']?\s*[:=]\s*[\"']?([a-zA-Z0-9_\-\.]{8,})[\"']?"
    ),
    re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}"),
]

SENSITIVE_ENV_KEYS = [
    "NGC_API_KEY",
    "NVIDIA_API_KEY",
    "NVIDIA_API",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "GITHUB_TOKEN",
]


class SecretScrubber:
    """Utility class to scrub credentials from strings and objects."""

    @staticmethod
    def scrub_text(text: str) -> str:
        return scrub_string(text)

    @staticmethod
    def scrub_data(data: Any) -> Any:
        if isinstance(data, dict):
            return scrub_dict(data)
        elif isinstance(data, str):
            return scrub_string(data)
        elif isinstance(data, list):
            return [SecretScrubber.scrub_data(x) for x in data]
        return data


def scrub_string(text: str) -> str:
    """Replaces sensitive tokens and keys with redacted placeholders."""
    if not isinstance(text, str):
        return text
    scrubbed = text
    for pattern in SENSITIVE_PATTERNS:
        scrubbed = pattern.sub("[REDACTED]", scrubbed)
    return scrubbed


def scrub_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub dictionaries to prevent secret leakage."""
    result = {}
    for k, v in d.items():
        if any(
            secret_key.lower() in k.lower() for secret_key in SENSITIVE_ENV_KEYS
        ) or any(term in k.lower() for term in ["key", "secret", "token", "password"]):
            result[k] = "[REDACTED]"
        elif isinstance(v, dict):
            result[k] = scrub_dict(v)
        elif isinstance(v, list):
            result[k] = [
                (
                    scrub_dict(item)
                    if isinstance(item, dict)
                    else (scrub_string(item) if isinstance(item, str) else item)
                )
                for item in v
            ]
        elif isinstance(v, str):
            result[k] = scrub_string(v)
        else:
            result[k] = v
    return result


def sanitize_env() -> dict[str, str]:
    """Return an environment dictionary with any active secrets safely scrubbed."""
    env = {}
    for k, v in os.environ.items():
        if any(secret_key.lower() in k.lower() for secret_key in SENSITIVE_ENV_KEYS):
            env[k] = "[REDACTED]"
        else:
            env[k] = scrub_string(v)
    return env


def get_nvidia_api_key(
    env_keys: Optional[List[str]] = None,
    search_files: bool = True,
) -> Optional[str]:
    """Retrieves the NVIDIA API key from environment or local .env files safely."""
    from od_experiments.earth2_001.nvidia_api import find_and_load_api_key
    return find_and_load_api_key(env_keys=env_keys, search_env_files=search_files)
