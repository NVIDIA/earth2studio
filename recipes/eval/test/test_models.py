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

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf
from src.models import load_diagnostics, load_prognostic

from earth2studio.models.dx import Identity
from earth2studio.models.px import Persistence

_RANK0_PATH = "src.models.run_on_rank0_first"

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
VARIABLES = ["t2m", "z500"]


def _passthrough(fn, *a, **kw):
    return fn(*a, **kw)


def _make_fake_prognostic_cls():
    """Build a fake class that mimics the AutoModelMixin protocol."""
    domain = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
    model = Persistence(variable=VARIABLES, domain_coords=domain)

    cls = MagicMock()
    cls.__name__ = "FakePx"
    cls.load_default_package.return_value = MagicMock(name="default_pkg")
    cls.load_model.return_value = model
    return cls, model


class TestLoadPrognostic:
    def test_default_package_path(self):
        fake_cls, expected_model = _make_fake_prognostic_cls()
        cfg = OmegaConf.create({"model": {"architecture": "some.module.FakePx"}})

        with patch("src.models.hydra.utils.get_class", return_value=fake_cls):
            with patch(_RANK0_PATH, side_effect=_passthrough):
                result = load_prognostic(cfg)

        fake_cls.load_default_package.assert_called_once()
        fake_cls.load_model.assert_called_once_with(
            package=fake_cls.load_default_package.return_value
        )
        assert result is expected_model

    def test_custom_package_path(self, tmp_path):
        fake_cls, expected_model = _make_fake_prognostic_cls()
        pkg_dir = str(tmp_path / "custom_pkg")
        cfg = OmegaConf.create(
            {
                "model": {
                    "architecture": "some.module.FakePx",
                    "package_path": pkg_dir,
                }
            }
        )

        with patch("src.models.hydra.utils.get_class", return_value=fake_cls):
            with patch("earth2studio.models.auto.Package") as mock_package:
                with patch(_RANK0_PATH, side_effect=_passthrough):
                    result = load_prognostic(cfg)

        fake_cls.load_default_package.assert_not_called()
        mock_package.assert_called_once_with(pkg_dir)
        assert result is expected_model

    def test_load_args_forwarded(self):
        fake_cls, _ = _make_fake_prognostic_cls()
        cfg = OmegaConf.create(
            {
                "model": {
                    "architecture": "some.module.FakePx",
                    "load_args": {"pretrained": True, "precision": "fp16"},
                }
            }
        )

        with patch("src.models.hydra.utils.get_class", return_value=fake_cls):
            with patch(_RANK0_PATH, side_effect=_passthrough):
                load_prognostic(cfg)

        _, kwargs = fake_cls.load_model.call_args
        assert kwargs["pretrained"] is True
        assert kwargs["precision"] == "fp16"


class TestLoadDiagnostics:
    def test_no_diagnostics_returns_empty(self):
        cfg = OmegaConf.create({"model": {"architecture": "whatever"}})
        result = load_diagnostics(cfg)
        assert result == []

    def test_target_style_instantiation(self):
        cfg = OmegaConf.create(
            {
                "diagnostics": {
                    "identity": {
                        "_target_": "earth2studio.models.dx.Identity",
                    }
                }
            }
        )
        result = load_diagnostics(cfg)
        assert len(result) == 1
        assert isinstance(result[0], Identity)

    def test_architecture_style_loading(self):
        fake_cls = MagicMock()
        fake_cls.__name__ = "FakeDx"
        dx_instance = Identity()
        fake_cls.load_model.return_value = dx_instance
        fake_cls.load_default_package.return_value = MagicMock(name="dx_pkg")

        cfg = OmegaConf.create(
            {"diagnostics": {"my_dx": {"architecture": "some.module.FakeDx"}}}
        )

        with patch("src.models.hydra.utils.get_class", return_value=fake_cls):
            with patch(
                "src.models.hydra.utils.instantiate",
                side_effect=RuntimeError("should not be called"),
            ):
                with patch(_RANK0_PATH, side_effect=_passthrough):
                    result = load_diagnostics(cfg)

        assert len(result) == 1
        assert result[0] is dx_instance
        fake_cls.load_default_package.assert_called_once()

    def test_bad_config_raises(self):
        cfg = OmegaConf.create({"diagnostics": {"bad": {"some_key": "some_value"}}})
        with pytest.raises(ValueError, match="'_target_' or 'architecture'"):
            load_diagnostics(cfg)

    def test_multiple_diagnostics(self):
        cfg = OmegaConf.create(
            {
                "diagnostics": {
                    "dx1": {"_target_": "earth2studio.models.dx.Identity"},
                    "dx2": {"_target_": "earth2studio.models.dx.Identity"},
                }
            }
        )
        result = load_diagnostics(cfg)
        assert len(result) == 2
        assert all(isinstance(d, Identity) for d in result)
