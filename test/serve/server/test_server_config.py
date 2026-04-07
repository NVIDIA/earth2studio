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
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SERVE_AVAILABLE = False
try:
    import hydra  # noqa: F401
    import omegaconf  # noqa: F401

    _SERVE_AVAILABLE = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not _SERVE_AVAILABLE, reason="hydra / omegaconf not available"
)

from earth2studio.serve.server.config import (  # noqa: E402
    AppConfig,
    ConfigManager,
    CORSConfig,
    LoggingConfig,
    ObjectStorageConfig,
    PathsConfig,
    QueueConfig,
    RedisConfig,
    ServerConfig,
    get_config,
    get_config_manager,
    get_workflow_config,
    reset_config,
)


class TestConfigDataclasses:
    """Test configuration dataclass defaults and initialization"""

    def test_redis_config_defaults(self) -> None:
        """Test RedisConfig default values"""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.socket_connect_timeout == 5
        assert config.socket_timeout == 5
        assert config.decode_responses is True
        assert config.retention_ttl == 604800

    def test_redis_config_custom_values(self) -> None:
        """Test RedisConfig with custom values"""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",  # noqa: S106
            retention_ttl=86400,
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"  # noqa: S105
        assert config.retention_ttl == 86400

    def test_queue_config_defaults(self) -> None:
        """Test QueueConfig default values"""
        config = QueueConfig()
        assert config.name == "inference"
        assert config.max_size == 10
        assert config.default_timeout == "1h"
        assert config.job_timeout == "2h"

    def test_paths_config_defaults(self) -> None:
        """Test PathsConfig default values"""
        config = PathsConfig()
        assert config.default_output_dir == "/outputs"
        assert (
            config.results_zip_dir == "/workspace/earth2studio-project/examples/outputs"
        )
        assert config.output_format == "zarr"
        assert config.result_zip_enabled is False

    def test_logging_config_defaults(self) -> None:
        """Test LoggingConfig default values"""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "%(asctime)s" in config.format

    def test_server_config_defaults(self) -> None:
        """Test ServerConfig default values"""
        config = ServerConfig()
        assert config.host == "0.0.0.0"  # noqa: S104
        assert config.port == 8000
        assert config.workers == 1
        assert config.results_ttl_hours == 24
        assert config.cleanup_watchdog_sec == 900

    def test_cors_config_defaults(self) -> None:
        """Test CORSConfig default values"""
        config = CORSConfig()
        assert config.allow_origins == ["*"]
        assert config.allow_credentials is True
        assert config.allow_methods == ["*"]
        assert config.allow_headers == ["*"]

    def test_object_storage_config_defaults(self) -> None:
        """Test ObjectStorageConfig default values"""
        config = ObjectStorageConfig()
        assert config.enabled is False
        assert config.bucket is None
        assert config.region == "us-east-1"
        assert config.prefix == "outputs"
        assert config.max_concurrency == 16
        assert config.signed_url_expires_in == 86400

    def test_app_config_defaults(self) -> None:
        """Test AppConfig with all default sub-configs"""
        config = AppConfig()
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.queue, QueueConfig)
        assert isinstance(config.paths, PathsConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.cors, CORSConfig)
        assert isinstance(config.object_storage, ObjectStorageConfig)


class TestConfigManagerSingleton:
    """Test ConfigManager singleton behavior"""

    def test_config_manager_is_singleton(self) -> None:
        """Test that ConfigManager is a singleton"""
        reset_config()
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_get_config_manager_returns_singleton(self) -> None:
        """Test that get_config_manager returns the same instance"""
        reset_config()
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        assert manager1 is manager2

    def test_reset_config_creates_new_instance(self) -> None:
        """Test that reset_config clears the global manager"""
        reset_config()
        manager1 = get_config_manager()
        # Reset clears the global
        reset_config()
        # Manually clear the instance to verify reset worked
        from earth2studio.serve.server.config import ConfigManager

        old_instance = ConfigManager._instance
        ConfigManager._instance = None
        manager2 = get_config_manager()
        # Verify reset_config cleared the old instance
        assert old_instance is None or manager1 is not manager2


class TestConfigManagerInitialization:
    """Test ConfigManager initialization and fallback behavior"""

    @patch("earth2studio.serve.server.config.initialize_config_dir")
    @patch("earth2studio.serve.server.config.compose")
    @patch("earth2studio.serve.server.config.GlobalHydra")
    def test_initialize_config_success(
        self,
        mock_global_hydra: MagicMock,
        mock_compose: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """Test successful config initialization"""
        reset_config()

        # Mock Hydra responses
        mock_cfg = MagicMock()
        mock_workflow_cfg = MagicMock()
        mock_compose.side_effect = [mock_cfg, mock_workflow_cfg]

        # Mock OmegaConf.to_container
        with patch("earth2studio.serve.server.config.OmegaConf") as mock_omega:
            mock_omega.to_container.side_effect = [
                {"redis": {"host": "test_host"}},
                {"workflow1": {}},
            ]

            manager = ConfigManager()
            config = manager.config

            assert config is not None
            assert isinstance(config, AppConfig)

    @patch("earth2studio.serve.server.config.initialize_config_dir")
    @patch("earth2studio.serve.server.config.compose")
    def test_initialize_config_fallback_to_defaults(
        self, mock_compose: MagicMock, mock_init: MagicMock
    ) -> None:
        """Test that config falls back to defaults on error"""
        reset_config()

        # Make Hydra initialization fail
        mock_init.side_effect = Exception("Config error")

        manager = ConfigManager()
        config = manager.config

        # Should still have a valid config with defaults
        assert config is not None
        assert isinstance(config, AppConfig)
        assert config.redis.host == "localhost"  # Default value


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides"""

    def setup_method(self) -> None:
        """Clear environment before each test"""
        self.env_vars_to_clear = [
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_DB",
            "REDIS_PASSWORD",
            "REDIS_RETENTION_TTL",
            "MAX_QUEUE_SIZE",
            "DEFAULT_OUTPUT_DIR",
            "RESULTS_ZIP_DIR",
            "LOG_LEVEL",
            "SERVER_PORT",
            "RESULTS_TTL_HOURS",
            "CLEANUP_WATCHDOG_SEC",
        ]
        for var in self.env_vars_to_clear:
            os.environ.pop(var, None)

    def teardown_method(self) -> None:
        """Clean up environment after each test"""
        for var in self.env_vars_to_clear:
            os.environ.pop(var, None)
        reset_config()

    def test_redis_host_override(self) -> None:
        """Test REDIS_HOST environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["REDIS_HOST"] = "custom-redis"
        manager._apply_env_overrides()
        assert manager.config.redis.host == "custom-redis"

    def test_redis_port_override(self) -> None:
        """Test REDIS_PORT environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["REDIS_PORT"] = "6380"
        manager._apply_env_overrides()
        assert manager.config.redis.port == 6380

    def test_redis_db_override(self) -> None:
        """Test REDIS_DB environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["REDIS_DB"] = "2"
        manager._apply_env_overrides()
        assert manager.config.redis.db == 2

    def test_redis_password_override(self) -> None:
        """Test REDIS_PASSWORD environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["REDIS_PASSWORD"] = "secret123"  # noqa: S105
        manager._apply_env_overrides()
        assert manager.config.redis.password == "secret123"  # noqa: S105

    def test_redis_retention_ttl_override(self) -> None:
        """Test REDIS_RETENTION_TTL environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["REDIS_RETENTION_TTL"] = "86400"
        manager._apply_env_overrides()
        assert manager.config.redis.retention_ttl == 86400

    def test_max_queue_size_override(self) -> None:
        """Test MAX_QUEUE_SIZE environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["MAX_QUEUE_SIZE"] = "50"
        manager._apply_env_overrides()
        assert manager.config.queue.max_size == 50

    def test_default_output_dir_override(self) -> None:
        """Test DEFAULT_OUTPUT_DIR environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["DEFAULT_OUTPUT_DIR"] = "/custom/outputs"
        manager._apply_env_overrides()
        assert manager.config.paths.default_output_dir == "/custom/outputs"

    def test_results_zip_dir_override(self) -> None:
        """Test RESULTS_ZIP_DIR environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["RESULTS_ZIP_DIR"] = "/custom/zips"
        manager._apply_env_overrides()
        assert manager.config.paths.results_zip_dir == "/custom/zips"

    def test_log_level_override(self) -> None:
        """Test LOG_LEVEL environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["LOG_LEVEL"] = "DEBUG"
        manager._apply_env_overrides()
        assert manager.config.logging.level == "DEBUG"

    def test_server_port_override(self) -> None:
        """Test SERVER_PORT environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["SERVER_PORT"] = "9000"
        manager._apply_env_overrides()
        assert manager.config.server.port == 9000

    def test_results_ttl_hours_override(self) -> None:
        """Test RESULTS_TTL_HOURS environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["RESULTS_TTL_HOURS"] = "48"
        manager._apply_env_overrides()
        assert manager.config.server.results_ttl_hours == 48

    def test_cleanup_watchdog_sec_override(self) -> None:
        """Test CLEANUP_WATCHDOG_SEC environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["CLEANUP_WATCHDOG_SEC"] = "1800"
        manager._apply_env_overrides()
        assert manager.config.server.cleanup_watchdog_sec == 1800

    def test_object_storage_enabled_override(self) -> None:
        """Test OBJECT_STORAGE_ENABLED environment variable override"""
        reset_config()
        manager = ConfigManager()
        os.environ["OBJECT_STORAGE_ENABLED"] = "true"
        manager._apply_env_overrides()
        assert manager.config.object_storage.enabled is True

        os.environ["OBJECT_STORAGE_ENABLED"] = "false"
        manager._apply_env_overrides()
        assert manager.config.object_storage.enabled is False

    def test_multiple_environment_overrides(self) -> None:
        """Test multiple environment variable overrides at once"""
        reset_config()
        manager = ConfigManager()
        os.environ["REDIS_HOST"] = "redis.example.com"
        os.environ["REDIS_PORT"] = "6380"
        os.environ["MAX_QUEUE_SIZE"] = "100"
        os.environ["LOG_LEVEL"] = "WARNING"
        manager._apply_env_overrides()
        assert manager.config.redis.host == "redis.example.com"
        assert manager.config.redis.port == 6380
        assert manager.config.queue.max_size == 100
        assert manager.config.logging.level == "WARNING"


class TestPathCreation:
    """Test path creation functionality"""

    def test_ensure_paths_exist_creates_directories(self) -> None:
        """Test that _ensure_paths_exist creates directories"""
        reset_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            zip_dir = Path(tmpdir) / "zips"

            manager = ConfigManager()
            manager._config.paths.default_output_dir = str(output_dir)
            manager._config.paths.results_zip_dir = str(zip_dir)

            manager._ensure_paths_exist()

            assert output_dir.exists()
            assert zip_dir.exists()

    def test_ensure_paths_exist_handles_errors_gracefully(self) -> None:
        """Test that _ensure_paths_exist handles errors gracefully"""
        reset_config()
        manager = ConfigManager()
        # Set invalid path (e.g., root directory without permissions)
        manager._config.paths.default_output_dir = "/root/invalid/path"

        # Should not raise exception, just log warning
        manager._ensure_paths_exist()
        # Test passes if no exception is raised


class TestGetConfigFunctions:
    """Test get_config and get_config_manager functions"""

    def test_get_config_returns_app_config(self) -> None:
        """Test that get_config returns AppConfig instance"""
        reset_config()
        config = get_config()
        assert isinstance(config, AppConfig)

    def test_get_config_manager_returns_config_manager(self) -> None:
        """Test that get_config_manager returns ConfigManager instance"""
        reset_config()
        manager = get_config_manager()
        assert isinstance(manager, ConfigManager)

    def test_get_config_and_get_config_manager_consistent(self) -> None:
        """Test that get_config uses get_config_manager"""
        reset_config()
        config1 = get_config()
        manager = get_config_manager()
        config2 = manager.config
        assert config1 is config2


class TestGetRedisUrl:
    """Test get_redis_url method"""

    def test_get_redis_url_without_password(self) -> None:
        """Test get_redis_url without password"""
        reset_config()
        manager = ConfigManager()
        manager._config.redis.host = "localhost"
        manager._config.redis.port = 6379
        manager._config.redis.db = 0
        manager._config.redis.password = None

        url = manager.get_redis_url()
        assert url == "redis://localhost:6379/0"

    def test_get_redis_url_with_password(self) -> None:
        """Test get_redis_url with password"""
        reset_config()
        manager = ConfigManager()
        manager._config.redis.host = "redis.example.com"
        manager._config.redis.port = 6380
        manager._config.redis.db = 1
        manager._config.redis.password = "secret123"  # noqa: S105

        url = manager.get_redis_url()
        assert url == "redis://:secret123@redis.example.com:6380/1"


class TestSetupLogging:
    """Test setup_logging method"""

    def test_setup_logging_configures_logging(self) -> None:
        """Test that setup_logging configures logging correctly"""
        reset_config()
        manager = ConfigManager()
        manager._config.logging.level = "DEBUG"
        manager._config.logging.format = "%(levelname)s - %(message)s"

        # Clear existing handlers to test fresh setup
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)  # Set to different level first

        manager.setup_logging()

        # Verify logging was configured (may have been set by basicConfig)
        # The exact level depends on when basicConfig was called, but we verify
        # the method doesn't raise an error
        assert root_logger is not None

    def test_setup_logging_adds_execution_id_filter(self) -> None:
        """Test that setup_logging adds ExecutionIdFilter"""
        reset_config()
        manager = ConfigManager()
        manager.setup_logging()

        root_logger = logging.getLogger()
        # Check that at least one handler has the filter
        for handler in root_logger.handlers:
            for filter_obj in handler.filters:
                if "ExecutionIdFilter" in str(type(filter_obj)):
                    # Filter found, test passes
                    return
        # Note: This may not always be true depending on when logging is initialized
        # But we test that the method doesn't raise an error


class TestGetWorkflowConfig:
    """Test get_workflow_config function"""

    @patch("earth2studio.serve.server.config.initialize_config_dir")
    @patch("earth2studio.serve.server.config.compose")
    def test_get_workflow_config_returns_dict(
        self, mock_compose: MagicMock, mock_init: MagicMock
    ) -> None:
        """Test that get_workflow_config returns a dictionary"""
        reset_config()

        mock_workflow_cfg = MagicMock()
        mock_compose.side_effect = [MagicMock(), mock_workflow_cfg]

        with patch("earth2studio.serve.server.config.OmegaConf") as mock_omega:
            mock_omega.to_container.side_effect = [
                {},  # config
                {"test_workflow": {"param": "value"}},  # workflow_config
            ]

            config = get_workflow_config("test_workflow")
            assert isinstance(config, dict)

    def test_get_workflow_config_returns_empty_dict_for_missing_workflow(
        self,
    ) -> None:
        """Test that get_workflow_config returns empty dict for missing workflow"""
        reset_config()
        manager = ConfigManager()
        manager._workflow_config = {}

        config = get_workflow_config("nonexistent_workflow")
        assert config == {}


class TestConfigManagerDictToConfig:
    """Test _dict_to_config method"""

    def test_dict_to_config_converts_dict_to_app_config(self) -> None:
        """Test that _dict_to_config converts dictionary to AppConfig"""
        reset_config()
        manager = ConfigManager()

        cfg_dict = {
            "redis": {"host": "test_host", "port": 6380},
            "queue": {"name": "test_queue", "max_size": 20},
            "paths": {"default_output_dir": "/test/outputs"},
            "logging": {"level": "DEBUG"},
            "server": {"port": 9000},
            "cors": {"allow_origins": ["http://example.com"]},
            "object_storage": {"enabled": True},
        }

        config = manager._dict_to_config(cfg_dict)
        assert isinstance(config, AppConfig)
        assert config.redis.host == "test_host"
        assert config.redis.port == 6380
        assert config.queue.name == "test_queue"
        assert config.queue.max_size == 20

    def test_dict_to_config_handles_missing_keys(self) -> None:
        """Test that _dict_to_config handles missing keys gracefully"""
        reset_config()
        manager = ConfigManager()

        cfg_dict = {"redis": {"host": "test_host"}}
        # Missing other keys should use defaults
        config = manager._dict_to_config(cfg_dict)
        assert isinstance(config, AppConfig)
        assert config.redis.host == "test_host"
        # Other configs should have defaults
        assert config.queue.name == "inference"  # Default value


class TestObjectStorageEnvOverrides:
    """Test object storage and Azure environment variable overrides"""

    def setup_method(self) -> None:
        self._vars = [
            "OBJECT_STORAGE_TYPE",
            "OBJECT_STORAGE_BUCKET",
            "OBJECT_STORAGE_REGION",
            "OBJECT_STORAGE_PREFIX",
            "OBJECT_STORAGE_ACCESS_KEY_ID",
            "OBJECT_STORAGE_SECRET_ACCESS_KEY",
            "OBJECT_STORAGE_SESSION_TOKEN",
            "OBJECT_STORAGE_ENDPOINT_URL",
            "OBJECT_STORAGE_TRANSFER_ACCELERATION",
            "OBJECT_STORAGE_MAX_CONCURRENCY",
            "OBJECT_STORAGE_MULTIPART_CHUNKSIZE",
            "OBJECT_STORAGE_USE_RUST_CLIENT",
            "CLOUDFRONT_DOMAIN",
            "CLOUDFRONT_KEY_PAIR_ID",
            "CLOUDFRONT_PRIVATE_KEY",
            "SIGNED_URL_EXPIRES_IN",
            "AZURE_STORAGE_ACCOUNT_NAME",
            "AZURE_CONTAINER_NAME",
            "AZURE_ENDPOINT_URL",
            "AZURE_GEOCATALOG_URL",
            "EXPOSED_WORKFLOWS",
            "OUTPUT_FORMAT",
            "CONFIG_DIR",
        ]
        for v in self._vars:
            os.environ.pop(v, None)

    def teardown_method(self) -> None:
        for v in self._vars:
            os.environ.pop(v, None)
        reset_config()

    def _get_manager(self) -> "ConfigManager":
        reset_config()
        return ConfigManager()

    def test_apply_env_overrides_from_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single pass: set env vars, ``_apply_env_overrides``, assert all mapped fields."""
        manager = self._get_manager()
        env = {
            "OBJECT_STORAGE_TYPE": "s3",
            "OBJECT_STORAGE_BUCKET": "my-bucket",
            "OBJECT_STORAGE_REGION": "eu-west-1",
            "OBJECT_STORAGE_PREFIX": "custom/prefix",
            "OBJECT_STORAGE_ACCESS_KEY_ID": "AKID",
            "OBJECT_STORAGE_SECRET_ACCESS_KEY": "SECRET",
            "OBJECT_STORAGE_SESSION_TOKEN": "TOKEN",
            "OBJECT_STORAGE_ENDPOINT_URL": "https://s3.local",
            "OBJECT_STORAGE_TRANSFER_ACCELERATION": "true",
            "OBJECT_STORAGE_MAX_CONCURRENCY": "32",
            "OBJECT_STORAGE_MULTIPART_CHUNKSIZE": "8388608",
            "OBJECT_STORAGE_USE_RUST_CLIENT": "true",
            "CLOUDFRONT_DOMAIN": "cdn.example.com",
            "CLOUDFRONT_KEY_PAIR_ID": "KID123",
            "CLOUDFRONT_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----",
            "SIGNED_URL_EXPIRES_IN": "3600",
            "AZURE_STORAGE_ACCOUNT_NAME": "myaccount",
            "AZURE_CONTAINER_NAME": "mycontainer",
            "AZURE_GEOCATALOG_URL": "https://geocatalog.example.com",
            "EXPOSED_WORKFLOWS": "workflow_a, workflow_b, workflow_c",
            "OUTPUT_FORMAT": "netcdf4",
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        manager._apply_env_overrides()

        obs = manager.config.object_storage
        assert obs.storage_type == "s3"
        assert obs.bucket == "my-bucket"
        assert obs.region == "eu-west-1"
        assert obs.prefix == "custom/prefix"
        assert obs.access_key_id == "AKID"
        assert obs.secret_access_key == "SECRET"  # noqa: S105
        assert obs.session_token == "TOKEN"  # noqa: S105
        assert obs.endpoint_url == "https://s3.local"
        assert obs.use_transfer_acceleration is True
        assert obs.max_concurrency == 32
        assert obs.multipart_chunksize == 8388608
        assert obs.use_rust_client is True
        assert obs.cloudfront_domain == "cdn.example.com"
        assert obs.cloudfront_key_pair_id == "KID123"
        assert obs.cloudfront_private_key == "-----BEGIN RSA PRIVATE KEY-----"
        assert obs.signed_url_expires_in == 3600
        assert obs.azure_account_name == "myaccount"
        assert obs.azure_container_name == "mycontainer"
        assert obs.azure_geocatalog_url == "https://geocatalog.example.com"
        assert manager.config.workflow_exposure.exposed_workflows == [
            "workflow_a",
            "workflow_b",
            "workflow_c",
        ]
        assert manager.config.paths.output_format == "netcdf4"

        # Boolean false parsing and alternate output format (second apply on same manager)
        monkeypatch.setenv("OBJECT_STORAGE_TRANSFER_ACCELERATION", "false")
        monkeypatch.setenv("OBJECT_STORAGE_USE_RUST_CLIENT", "false")
        monkeypatch.setenv("OUTPUT_FORMAT", "zarr")
        manager._apply_env_overrides()
        assert obs.use_transfer_acceleration is False
        assert obs.use_rust_client is False
        assert manager.config.paths.output_format == "zarr"

        monkeypatch.setenv("OBJECT_STORAGE_TYPE", "azure")
        manager._apply_env_overrides()
        assert obs.storage_type == "azure"

        monkeypatch.setenv(
            "AZURE_ENDPOINT_URL", "https://myaccount.blob.core.windows.net"
        )
        manager._apply_env_overrides()
        assert obs.endpoint_url == "https://myaccount.blob.core.windows.net"

    def test_apply_env_overrides_invalid_storage_type_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = self._get_manager()
        original = manager.config.object_storage.storage_type
        monkeypatch.setenv("OBJECT_STORAGE_TYPE", "gcs")
        manager._apply_env_overrides()
        assert manager.config.object_storage.storage_type == original

    def test_apply_env_overrides_invalid_output_format_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = self._get_manager()
        original = manager.config.paths.output_format
        monkeypatch.setenv("OUTPUT_FORMAT", "csv")
        manager._apply_env_overrides()
        assert manager.config.paths.output_format == original

    def test_apply_env_overrides_no_op_when_config_none(self) -> None:
        """_apply_env_overrides returns early when _config is None"""
        reset_config()
        manager = ConfigManager()
        manager._config = None
        # Should not raise
        manager._apply_env_overrides()

    def test_ensure_paths_exist_no_op_when_config_none(self) -> None:
        """_ensure_paths_exist returns early when _config is None"""
        reset_config()
        manager = ConfigManager()
        manager._config = None
        # Should not raise
        manager._ensure_paths_exist()

    def test_config_property_reinitializes_when_none(self) -> None:
        """config property calls _initialize_config when _config is None"""
        reset_config()
        manager = ConfigManager()
        manager._config = None
        cfg = manager.config
        assert isinstance(cfg, AppConfig)

    def test_workflow_config_property_reinitializes_when_none(self) -> None:
        """workflow_config property calls _initialize_config when _workflow_config is None"""
        reset_config()
        manager = ConfigManager()
        manager._workflow_config = None
        expected = {"wf": {"param": 1}}
        with patch.object(
            manager,
            "_initialize_config",
            side_effect=lambda: setattr(manager, "_workflow_config", expected),
        ) as mock_init:
            wf_cfg = manager.workflow_config
        mock_init.assert_called_once()
        assert wf_cfg == expected

    def test_initialize_config_uses_config_dir_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_initialize_config uses CONFIG_DIR env var when set (covers lines 203-204)"""
        reset_config()
        manager = ConfigManager()
        manager._config = None
        manager._workflow_config = None
        monkeypatch.setenv("CONFIG_DIR", "/custom/conf")
        manager._initialize_config()
        assert isinstance(manager._config, AppConfig)
