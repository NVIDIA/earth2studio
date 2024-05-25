# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from pathlib import Path

import pytest

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.dx import ClimateNet, CorrDiffTaiwan, PrecipitationAFNO
from earth2studio.models.px import (
    DLWP,
    FCN,
    SFNO,
    FengWu,
    FuXi,
    Pangu3,
    Pangu6,
    Pangu24,
)


@pytest.fixture
def cache_dir():
    return Path("./cache").resolve()


# @pytest.mark.xfail
@pytest.mark.model_download
@pytest.mark.parametrize(
    "model",
    [
        DLWP,
        FCN,
        FengWu,
        FuXi,
        Pangu24,
        Pangu6,
        Pangu3,
        SFNO,
        PrecipitationAFNO,
        ClimateNet,
        CorrDiffTaiwan,
    ],
)
def test_auto_model(model, model_cache_context):
    """Automodel download test from remote stores for all models
    This should not be ran in a CI pipeline, rather reserved to periodic testing /
    manual tests. Can also be used to create a complete model cache.

    Parameters
    ----------
    model : AutoModelMixin
        Model class that is an auto model in Earth2Studio
    model_cache_context : EnvContextManager
        Context manager that changed cache dir for CI, provided via fixture
    """
    assert issubclass(model, AutoModelMixin), "Model class needs to be an AutoModel"
    with model_cache_context(EARTH2STUDIO_CACHE="./cache"):
        package = model.load_default_package()
        model.load_model(package)


@pytest.fixture(scope="session")
def cache_folder(tmp_path_factory):
    fn = tmp_path_factory.mktemp("cache")
    open(fn / "test.txt", "a").close()
    return fn


@pytest.mark.parametrize(
    "url,file",
    [(None, "test.txt"), ("hf://NickGeneva/earth_ai", "README.md")],
)
def test_package(url, file, cache_folder, model_cache_context):
    if url is None:
        url = "file://" / cache_folder
    with model_cache_context(EARTH2STUDIO_CACHE=str(cache_folder.resolve())):
        package = Package(url)
        file_path = package.get(file)
        assert Path(file_path).is_file()


def test_auto_model_mixin():

    with pytest.raises(NotImplementedError):
        AutoModelMixin.load_default_package()

    with pytest.raises(NotImplementedError):
        package = Package("./package")
        AutoModelMixin.load_model(package)

    with pytest.raises(NotImplementedError):
        AutoModelMixin.from_pretrained()

    with pytest.raises(NotImplementedError):
        package = Package("./package")
        AutoModelMixin.from_pretrained(package)
