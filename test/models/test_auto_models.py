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

import fsspec
import pytest

from earth2studio.models.auto import (
    AutoModelMixin,
    Package,
    TqdmCallbackRelative,
    TqdmFormat,
)
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
    [
        (None, "test.txt"),
        ("hf://NickGeneva/earth_ai", "README.md"),
        (
            "ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0",
            "sfno_73ch_small/metadata.json",
        ),
        ("s3://noaa-swpc-pds", "text/3-day-geomag-forecast.txt"),
    ],
)
def test_package(url, file, cache_folder, model_cache_context):
    if url is None:
        url = "file://" / cache_folder
    with model_cache_context(
        EARTH2STUDIO_CACHE=str(cache_folder.resolve()),
        EARTH2STUDIO_PACKAGE_TIMEOUT="30",
    ):
        package = Package(str(url))
        file_path = package.resolve(file)
        assert Path(file_path).is_file()
        # Getting depricated
        file_path2 = package.get(file)
        assert file_path == file_path2


def test_auto_model_mixin():

    with pytest.raises(NotImplementedError):
        AutoModelMixin.load_default_package()

    with pytest.raises(NotImplementedError):
        package = "./package"
        AutoModelMixin.load_model(package)

    with pytest.raises(NotImplementedError):
        AutoModelMixin.from_pretrained()

    with pytest.raises(NotImplementedError):
        package = "./package"
        AutoModelMixin.from_pretrained(package)


@pytest.mark.parametrize("same_names", [True, False])
def test_whole_file_cache(tmp_path, same_names):

    cache_path = tmp_path / "cache"
    put_path = tmp_path / "put"
    put_path.mkdir(parents=True, exist_ok=True)

    file1 = "test.txt"
    file2 = "test2.txt"

    with open(tmp_path / file1, "a") as f:
        f.write("Hello World")

    fs = fsspec.filesystem("file")
    package = Package(
        tmp_path,
        fs=fs,
        cache_options={
            "cache_storage": str(cache_path),
            "same_names": same_names,
            "expiry_time": 30,
        },
    )
    fs = package.fs

    with TqdmCallbackRelative(
        tqdm_kwargs={"desc": "Test"},
        tqdm_cls=TqdmFormat,
    ) as callback:
        fs.get(str(tmp_path / file1), str(tmp_path / file2), callback=callback)
    fs.open(str(tmp_path / file2), mode="rb", compression="infer")
    with fs.open(str(tmp_path / file2), mode="w") as f:
        f.write("Hello World")
    fs.put(str(tmp_path / file2), str(put_path / file2))

    assert (tmp_path / file1).is_file()
    assert (tmp_path / file2).is_file()
    assert (put_path / file2).is_file()
    assert (cache_path / file2).is_file() is same_names
