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

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, cast

from loguru import logger
from tqdm import tqdm

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
except ImportError:
    OptionalDependencyFailure("serve")
    compose = None
    initialize_config_dir = None
    GlobalHydra = None
    OmegaConf = None


class _InterceptHandler(logging.Handler):
    """Forward stdlib log records to loguru so uvicorn/FastAPI logs are unified."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def resolve_repo_root() -> Path:
    """Return the repository root by walking up until ``pyproject.toml`` is found.

    Falls back gracefully: if no marker is found, the caller should treat the
    result as best-effort and prefer an explicit env-var override.
    """
    current = Path(__file__).resolve().parent
    for ancestor in (current, *current.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return current.parent.parent.parent


def resolve_serve_path(env_var: str, relative_to_root: str) -> Path:
    """Resolve a ``serve/`` sub-path, preferring an env-var override.

    Parameters
    ----------
    env_var : str
        Environment variable name (e.g. ``CONFIG_DIR``).
    relative_to_root : str
        Path relative to the repository root used when *env_var* is unset
        (e.g. ``serve/server/conf``).

    Returns
    -------
    Path
        Absolute path to the requested directory/file.
    """
    value = os.environ.get(env_var)
    if value:
        return Path(value)
    return resolve_repo_root() / relative_to_root


@dataclass
class RedisConfig:
    """Redis connection configuration"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    socket_connect_timeout: int = 5
    socket_timeout: int = 5
    decode_responses: bool = True
    retention_ttl: int = 604800  # Redis key retention TTL in seconds (7 days)


@dataclass
class QueueConfig:
    """RQ queue configuration"""

    name: str = "inference"
    result_zip_queue_name: str = "result_zip"
    object_storage_queue_name: str = "object_storage"
    geocatalog_ingestion_queue_name: str = "geocatalog_ingestion"
    finalize_metadata_queue_name: str = "finalize_metadata"
    max_size: int = 10
    default_timeout: str = "1h"
    job_timeout: str = "2h"


@dataclass
class PathsConfig:
    """File system paths configuration"""

    default_output_dir: str = "/outputs"
    results_zip_dir: str = "/outputs"
    output_format: Literal["zarr", "netcdf4"] = "zarr"
    result_zip_enabled: bool = (
        False  # because output_format is defaulted to zarr, default zip creation to False
    )


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} - [{extra[execution_id]}] "
        "{name} - {level} - {message}"
    )


@dataclass
class ServerConfig:
    """FastAPI server configuration"""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    workers: int = 1
    title: str = "Earth2Studio REST API"
    description: str = "REST API for running Earth2Studio workflows"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    results_ttl_hours: int = 24
    cleanup_watchdog_sec: int = 900


@dataclass
class CORSConfig:
    """CORS middleware configuration"""

    allow_origins: list = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: list = field(default_factory=lambda: ["*"])
    allow_headers: list = field(default_factory=lambda: ["*"])


@dataclass
class ObjectStorageConfig:
    """Object storage configuration for S3/CloudFront and Azure Blob Storage"""

    enabled: bool = False
    storage_type: Literal["s3", "azure"] = "s3"  # Storage provider type
    # S3 configuration
    bucket: str | None = None
    region: str = "us-east-1"
    prefix: str = "outputs"  # Remote prefix for uploaded files
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    endpoint_url: str | None = None  # For S3-compatible services
    use_transfer_acceleration: bool = True
    # Transfer configuration
    max_concurrency: int = 16  # Maximum number of concurrent transfers
    multipart_chunksize: int = 8 * 1024 * 1024  # 8 MB chunk size for multipart uploads
    use_rust_client: bool = True  # Enable high-performance Rust client
    # CloudFront signed URL configuration
    cloudfront_domain: str | None = None
    cloudfront_key_pair_id: str | None = None
    cloudfront_private_key: str | None = None  # PEM private key content
    # Signed URL settings
    signed_url_expires_in: int = 86400  # Default 24 hours
    # Azure: storage account and container come from workflow request ``container_url``;
    # GeoCatalog base URL from request ``geo_catalog_url`` (see cpu_worker / object storage).


@dataclass
class WorkflowExposureConfig:
    """Configuration for controlling which workflows are exposed via API endpoints"""

    exposed_workflows: list[str] = field(
        default_factory=lambda: []
    )  # Empty list means all workflows are exposed
    warmup_workflows: list[str] = field(
        default_factory=lambda: ["example_user_workflow"]
    )  # Workflows accessible for warmup even if not exposed


@dataclass
class AppConfig:
    """Root configuration for the application"""

    redis: RedisConfig = field(default_factory=RedisConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)
    object_storage: ObjectStorageConfig = field(default_factory=ObjectStorageConfig)
    workflow_exposure: WorkflowExposureConfig = field(
        default_factory=WorkflowExposureConfig
    )


class ConfigManager:
    """
    Singleton configuration manager using Hydra.

    This class provides a centralized way to access configuration throughout
    the application. It supports:
    - Loading from YAML files
    - Environment variable overrides
    - Command-line argument overrides
    - Programmatic access to config values
    """

    _instance: Optional["ConfigManager"] = None
    _config: AppConfig | None = None
    _workflow_config: dict[str, Any] = {}

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._config is None:
            self._initialize_config()

    def _initialize_config(self) -> None:
        """Initialize configuration from Hydra"""
        try:
            config_dir = resolve_serve_path("CONFIG_DIR", "serve/server/conf")

            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            # Initialize Hydra with config directory
            initialize_config_dir(
                config_dir=str(config_dir.absolute()), version_base=None
            )

            # Compose configuration
            cfg = compose(config_name="config")
            workflow_cfg = compose(config_name="workflows")

            # Convert to structured config - manually construct from dict
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            self._config = self._dict_to_config(cfg_dict)

            # Workflow configs are validated when loaded
            self._workflow_config = OmegaConf.to_container(workflow_cfg)

            # Apply environment variable overrides
            self._apply_env_overrides()

            # Ensure paths exist
            self._ensure_paths_exist()

            logger.info("Configuration initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Hydra config: {e}")
            logger.info("Falling back to default configuration")
            self._config = self._create_default_config_object()
            self._apply_env_overrides()
            self._ensure_paths_exist()

    def _dict_to_config(self, cfg_dict: dict) -> AppConfig:
        """Convert dictionary to AppConfig object"""
        return AppConfig(
            redis=RedisConfig(**cfg_dict.get("redis", {})),
            queue=QueueConfig(**cfg_dict.get("queue", {})),
            paths=PathsConfig(**cfg_dict.get("paths", {})),
            logging=LoggingConfig(**cfg_dict.get("logging", {})),
            server=ServerConfig(**cfg_dict.get("server", {})),
            cors=CORSConfig(**cfg_dict.get("cors", {})),
            object_storage=ObjectStorageConfig(**cfg_dict.get("object_storage", {})),
            workflow_exposure=WorkflowExposureConfig(
                **cfg_dict.get("workflow_exposure", {})
            ),
        )

    def _create_default_config_object(self) -> AppConfig:
        """Create default configuration object"""
        return AppConfig(
            redis=RedisConfig(),
            queue=QueueConfig(),
            paths=PathsConfig(),
            logging=LoggingConfig(),
            server=ServerConfig(),
            cors=CORSConfig(),
            object_storage=ObjectStorageConfig(),
            workflow_exposure=WorkflowExposureConfig(),
        )

    # ------------------------------------------------------------------ #
    # Env-var override table.  Each entry is:
    #   (ENV_VAR, config_section, field_name, converter)
    #
    # *converter* is one of:
    #   str   – plain string assignment
    #   int   – int() conversion
    #   bool  – case-insensitive "true" check
    #   None  – handled by custom logic in _apply_env_special()
    # ------------------------------------------------------------------ #
    _ENV_OVERRIDES: list[tuple[str, str, str, type | None]] = [
        # Redis
        ("REDIS_HOST", "redis", "host", str),
        ("REDIS_PORT", "redis", "port", int),
        ("REDIS_DB", "redis", "db", int),
        ("REDIS_PASSWORD", "redis", "password", str),
        ("REDIS_RETENTION_TTL", "redis", "retention_ttl", int),
        # Queue
        ("MAX_QUEUE_SIZE", "queue", "max_size", int),
        # Paths
        ("DEFAULT_OUTPUT_DIR", "paths", "default_output_dir", str),
        ("RESULTS_ZIP_DIR", "paths", "results_zip_dir", str),
        ("OUTPUT_FORMAT", "paths", "output_format", None),
        # Logging
        ("LOG_LEVEL", "logging", "level", str),
        # Server
        ("SERVER_PORT", "server", "port", int),
        ("RESULTS_TTL_HOURS", "server", "results_ttl_hours", int),
        ("CLEANUP_WATCHDOG_SEC", "server", "cleanup_watchdog_sec", int),
        # Object storage – simple types
        ("OBJECT_STORAGE_ENABLED", "object_storage", "enabled", bool),
        ("OBJECT_STORAGE_TYPE", "object_storage", "storage_type", None),
        ("OBJECT_STORAGE_BUCKET", "object_storage", "bucket", str),
        ("OBJECT_STORAGE_REGION", "object_storage", "region", str),
        ("OBJECT_STORAGE_PREFIX", "object_storage", "prefix", str),
        ("OBJECT_STORAGE_ACCESS_KEY_ID", "object_storage", "access_key_id", str),
        (
            "OBJECT_STORAGE_SECRET_ACCESS_KEY",
            "object_storage",
            "secret_access_key",
            str,
        ),
        ("OBJECT_STORAGE_SESSION_TOKEN", "object_storage", "session_token", str),
        ("OBJECT_STORAGE_ENDPOINT_URL", "object_storage", "endpoint_url", str),
        (
            "OBJECT_STORAGE_TRANSFER_ACCELERATION",
            "object_storage",
            "use_transfer_acceleration",
            bool,
        ),
        ("OBJECT_STORAGE_MAX_CONCURRENCY", "object_storage", "max_concurrency", int),
        (
            "OBJECT_STORAGE_MULTIPART_CHUNKSIZE",
            "object_storage",
            "multipart_chunksize",
            int,
        ),
        ("OBJECT_STORAGE_USE_RUST_CLIENT", "object_storage", "use_rust_client", bool),
        ("CLOUDFRONT_DOMAIN", "object_storage", "cloudfront_domain", str),
        ("CLOUDFRONT_KEY_PAIR_ID", "object_storage", "cloudfront_key_pair_id", str),
        ("CLOUDFRONT_PRIVATE_KEY", "object_storage", "cloudfront_private_key", str),
        ("SIGNED_URL_EXPIRES_IN", "object_storage", "signed_url_expires_in", int),
        # Workflow exposure (custom)
        ("EXPOSED_WORKFLOWS", "workflow_exposure", "exposed_workflows", None),
    ]

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration.

        Iterates the ``_ENV_OVERRIDES`` table and, for each env var that is
        set, converts the value and assigns it to the matching config field.
        Entries with ``converter=None`` are routed to ``_apply_env_special``.
        """
        if not self._config:
            return

        for env_var, section, field_name, converter in self._ENV_OVERRIDES:
            raw = os.environ.get(env_var)
            if not raw:
                continue

            sub_cfg = getattr(self._config, section)

            if converter is None:
                self._apply_env_special(env_var, raw, sub_cfg, field_name)
            elif converter is bool:
                setattr(sub_cfg, field_name, raw.lower() == "true")
            elif converter is int:
                try:
                    setattr(sub_cfg, field_name, int(raw))
                except ValueError:
                    logger.warning(
                        "Ignoring non-integer value %r for env var %s", raw, env_var
                    )
            else:
                setattr(sub_cfg, field_name, raw)

        logger.debug("Environment variable overrides applied")

    @staticmethod
    def _apply_env_special(env_var: str, raw: str, sub_cfg: Any, field: str) -> None:
        """Handle env-var overrides that need validation beyond a simple cast."""
        if env_var == "OUTPUT_FORMAT":
            fmt = raw.lower()
            if fmt in ("zarr", "netcdf4"):
                setattr(sub_cfg, field, cast(Literal["zarr", "netcdf4"], fmt))
        elif env_var == "OBJECT_STORAGE_TYPE":
            st = raw.lower()
            if st in ("s3", "azure"):
                setattr(sub_cfg, field, cast(Literal["s3", "azure"], st))
        elif env_var == "EXPOSED_WORKFLOWS":
            setattr(sub_cfg, field, [w.strip() for w in raw.split(",") if w.strip()])

    def _ensure_paths_exist(self) -> None:
        """Ensure all configured paths exist"""
        if not self._config:
            return

        try:
            Path(self._config.paths.default_output_dir).mkdir(
                parents=True, exist_ok=True
            )
            Path(self._config.paths.results_zip_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create directories: {e}")

    @property
    def config(self) -> AppConfig:
        """Get the current configuration"""
        if self._config is None:
            self._initialize_config()
        return cast(AppConfig, self._config)

    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        cfg = self.config.redis
        password_part = f":{cfg.password}@" if cfg.password else ""
        return f"redis://{password_part}{cfg.host}:{cfg.port}/{cfg.db}"

    def setup_logging(self) -> None:
        """Configure logging using loguru."""
        logger.remove()
        logger.configure(extra={"execution_id": ""})
        logger.add(
            lambda msg: tqdm.write(msg, end="", file=sys.stderr),
            format=self.config.logging.format,
            level=self.config.logging.level.upper(),
            colorize=False,
        )

        # Intercept stdlib loggers (uvicorn, FastAPI, etc.) into loguru
        logging.basicConfig(
            handlers=[_InterceptHandler()],
            level=logging.getLevelName(self.config.logging.level.upper()),
            force=True,
        )

    @property
    def workflow_config(self) -> dict[str, Any]:
        """Get the current configuration"""
        if self._workflow_config is None:
            self._initialize_config()
        return self._workflow_config


def get_config() -> AppConfig:
    """
    Get the global configuration instance.

    Returns:
        AppConfig: The application configuration
    """
    return get_config_manager().config


def get_config_manager() -> ConfigManager:
    """
    Get the configuration manager instance (singleton).

    Returns:
        ConfigManager: The configuration manager

    Example:
        >>> from earth2studio.serve.server.config import get_config_manager
        >>> manager = get_config_manager()
        >>> manager.setup_logging()
    """
    return ConfigManager()


# Convenience function to reset config (mainly for testing)
@check_optional_dependencies()
def reset_config() -> None:
    """Reset the configuration manager singleton (mainly for testing)."""
    ConfigManager._instance = None
    GlobalHydra.instance().clear()


def get_workflow_config(name: str) -> dict[str, Any]:
    """
    Get the global workflow configuration.

    Returns:
        dict[str, Any]: The workflow configuration
    """
    return get_config_manager().workflow_config.get(name, {})
