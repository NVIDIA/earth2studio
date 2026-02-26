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

"""
Configuration Management for Earth2Studio API Server

This module provides centralized configuration management using Hydra.
All configuration values are defined here with sensible defaults and can be
overridden via YAML files or command-line arguments.
"""

import logging
import os
from dataclasses import dataclass, field
from logging import LogRecord
from pathlib import Path
from typing import Any, Literal, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


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
    finalize_metadata_queue_name: str = "finalize_metadata"
    max_size: int = 10
    default_timeout: str = "1h"
    job_timeout: str = "2h"


@dataclass
class PathsConfig:
    """File system paths configuration"""

    default_output_dir: str = "/outputs"
    results_zip_dir: str = "/workspace/earth2studio-project/examples/outputs"
    output_format: Literal["zarr", "netcdf4"] = "zarr"
    result_zip_enabled: bool = (
        False  # because output_format is defaulted to zarr, default zip creation to False
    )


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


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
    """Object storage configuration for S3/CloudFront"""

    enabled: bool = False
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
    _config: AppConfig = AppConfig()
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
            # Get config directory
            config_dir = Path(__file__).parent / "conf"

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
        )

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration"""
        if not self._config:
            return

        # Redis overrides
        if os.getenv("REDIS_HOST"):
            self._config.redis.host = os.getenv(
                "REDIS_HOST", default=self._config.redis.host
            )
        if os.getenv("REDIS_PORT"):
            self._config.redis.port = int(
                os.getenv("REDIS_PORT", default=self._config.redis.port)
            )
        if os.getenv("REDIS_DB"):
            self._config.redis.db = int(
                os.getenv("REDIS_DB", default=self._config.redis.db)
            )
        if os.getenv("REDIS_PASSWORD"):
            self._config.redis.password = os.getenv("REDIS_PASSWORD")
        if os.getenv("REDIS_RETENTION_TTL"):
            self._config.redis.retention_ttl = int(
                os.getenv(
                    "REDIS_RETENTION_TTL", default=self._config.redis.retention_ttl
                )
            )

        # Queue overrides
        if os.getenv("MAX_QUEUE_SIZE"):
            self._config.queue.max_size = int(
                os.getenv("MAX_QUEUE_SIZE", default=self._config.queue.max_size)
            )

        # Path overrides
        if os.getenv("DEFAULT_OUTPUT_DIR"):
            self._config.paths.default_output_dir = os.getenv(
                "DEFAULT_OUTPUT_DIR", default=self._config.paths.default_output_dir
            )
        if os.getenv("RESULTS_ZIP_DIR"):
            self._config.paths.results_zip_dir = os.getenv(
                "RESULTS_ZIP_DIR", default=self._config.paths.results_zip_dir
            )

        # Logging overrides
        if os.getenv("LOG_LEVEL"):
            self._config.logging.level = os.getenv(
                "LOG_LEVEL", default=self._config.logging.level
            )

        # Server overrides
        if os.getenv("SERVER_PORT"):
            self._config.server.port = int(
                os.getenv("SERVER_PORT", default=self._config.server.port)
            )
        if os.getenv("RESULTS_TTL_HOURS"):
            self._config.server.results_ttl_hours = int(
                os.getenv(
                    "RESULTS_TTL_HOURS", default=self._config.server.results_ttl_hours
                )
            )
        if os.getenv("CLEANUP_WATCHDOG_SEC"):
            self._config.server.cleanup_watchdog_sec = int(
                os.getenv(
                    "CLEANUP_WATCHDOG_SEC",
                    default=self._config.server.cleanup_watchdog_sec,
                )
            )

        # Object storage overrides
        if os.getenv("OBJECT_STORAGE_ENABLED"):
            self._config.object_storage.enabled = (
                os.getenv("OBJECT_STORAGE_ENABLED", "").lower() == "true"
            )
        if os.getenv("OBJECT_STORAGE_BUCKET"):
            self._config.object_storage.bucket = os.getenv("OBJECT_STORAGE_BUCKET")
        if os.getenv("OBJECT_STORAGE_REGION"):
            self._config.object_storage.region = os.getenv(
                "OBJECT_STORAGE_REGION", default=self._config.object_storage.region
            )
        if os.getenv("OBJECT_STORAGE_PREFIX"):
            self._config.object_storage.prefix = os.getenv(
                "OBJECT_STORAGE_PREFIX", default=self._config.object_storage.prefix
            )
        if os.getenv("OBJECT_STORAGE_ACCESS_KEY_ID"):
            self._config.object_storage.access_key_id = os.getenv(
                "OBJECT_STORAGE_ACCESS_KEY_ID"
            )
        if os.getenv("OBJECT_STORAGE_SECRET_ACCESS_KEY"):
            self._config.object_storage.secret_access_key = os.getenv(
                "OBJECT_STORAGE_SECRET_ACCESS_KEY"
            )
        if os.getenv("OBJECT_STORAGE_SESSION_TOKEN"):
            self._config.object_storage.session_token = os.getenv(
                "OBJECT_STORAGE_SESSION_TOKEN"
            )
        if os.getenv("OBJECT_STORAGE_ENDPOINT_URL"):
            self._config.object_storage.endpoint_url = os.getenv(
                "OBJECT_STORAGE_ENDPOINT_URL"
            )
        if os.getenv("OBJECT_STORAGE_TRANSFER_ACCELERATION"):
            self._config.object_storage.use_transfer_acceleration = (
                os.getenv("OBJECT_STORAGE_TRANSFER_ACCELERATION", "").lower() == "true"
            )
        if os.getenv("OBJECT_STORAGE_MAX_CONCURRENCY"):
            self._config.object_storage.max_concurrency = int(
                os.getenv(
                    "OBJECT_STORAGE_MAX_CONCURRENCY",
                    default=self._config.object_storage.max_concurrency,
                )
            )
        if os.getenv("OBJECT_STORAGE_MULTIPART_CHUNKSIZE"):
            self._config.object_storage.multipart_chunksize = int(
                os.getenv(
                    "OBJECT_STORAGE_MULTIPART_CHUNKSIZE",
                    default=self._config.object_storage.multipart_chunksize,
                )
            )
        if os.getenv("OBJECT_STORAGE_USE_RUST_CLIENT"):
            self._config.object_storage.use_rust_client = (
                os.getenv("OBJECT_STORAGE_USE_RUST_CLIENT", "").lower() == "true"
            )
        if os.getenv("CLOUDFRONT_DOMAIN"):
            self._config.object_storage.cloudfront_domain = os.getenv(
                "CLOUDFRONT_DOMAIN"
            )
        if os.getenv("CLOUDFRONT_KEY_PAIR_ID"):
            self._config.object_storage.cloudfront_key_pair_id = os.getenv(
                "CLOUDFRONT_KEY_PAIR_ID"
            )
        if os.getenv("CLOUDFRONT_PRIVATE_KEY"):
            self._config.object_storage.cloudfront_private_key = os.getenv(
                "CLOUDFRONT_PRIVATE_KEY"
            )
        if os.getenv("SIGNED_URL_EXPIRES_IN"):
            self._config.object_storage.signed_url_expires_in = int(
                os.getenv(
                    "SIGNED_URL_EXPIRES_IN",
                    default=self._config.object_storage.signed_url_expires_in,
                )
            )

        logger.debug("Environment variable overrides applied")

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
        return self._config

    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        cfg = self.config.redis
        password_part = f":{cfg.password}@" if cfg.password else ""
        return f"redis://{password_part}{cfg.host}:{cfg.port}/{cfg.db}"

    def setup_logging(self) -> None:
        """Configure logging based on settings"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            format=self.config.logging.format,
        )

        # Add filter to set default execution_id if not present
        class ExecutionIdFilter(logging.Filter):
            """Add default execution_id to records that don't have one"""

            def filter(self, record: LogRecord) -> bool:
                if not hasattr(record, "execution_id"):
                    record.execution_id = ""
                return True

        # Add the filter to all handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(ExecutionIdFilter())

    @property
    def workflow_config(self) -> dict[str, Any]:
        """Get the current configuration"""
        if self._workflow_config is None:
            self._initialize_config()
        return self._workflow_config


# Global configuration instance
_config_manager: ConfigManager | None = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance.

    Returns:
        AppConfig: The application configuration

    Example:
        >>> from api_server.config import get_config
        >>> config = get_config()
        >>> print(config.redis.host)
        'localhost'
    """
    return get_config_manager().config


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.

    Returns:
        ConfigManager: The configuration manager

    Example:
        >>> from api_server.config import get_config_manager
        >>> manager = get_config_manager()
        >>> manager.setup_logging()
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Convenience function to reset config (mainly for testing)
def reset_config() -> None:
    """Reset the global configuration instance (mainly for testing)"""
    global _config_manager
    _config_manager = None
    GlobalHydra.instance().clear()


def get_workflow_config(name: str) -> dict[str, Any]:
    """
    Get the global workflow configuration.

    Returns:
        dict[str, Any]: The workflow configuration

    Example:
        >>> from api_server.config import get_workflow_config
        >>> config = get_workflow_config("deterministic_earth2_workflow")
        >>> print(config["model_type"])
        'fcn'
    """
    return get_config_manager().workflow_config.get(name, {})
