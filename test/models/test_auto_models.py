# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import http.client
import os
from pathlib import Path

import fsspec
import ngcbase
import pytest

from earth2studio.models.auto import (
    AutoModelMixin,
    Package,
)
from earth2studio.models.auto.ngc import NGCModelFileSystem
from earth2studio.models.auto.package import (
    TqdmCallbackRelative,
    TqdmFormat,
)
from earth2studio.models.dx import (
    ClimateNet,
    CorrDiffTaiwan,
    PrecipitationAFNO,
    PrecipitationAFNOv2,
    WindgustAFNO,
)
from earth2studio.models.px import (
    DLWP,
    FCN,
    SFNO,
    Aurora,
    FengWu,
    FuXi,
    InterpModAFNO,
    Pangu3,
    Pangu6,
    Pangu24,
)


# @pytest.mark.xfail
@pytest.mark.model_download
@pytest.mark.parametrize(
    "model",
    [
        Aurora,
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
        WindgustAFNO,
        InterpModAFNO,
        PrecipitationAFNOv2,
    ],
)
def test_auto_model_download(model, model_cache_context):
    """Automodel download test from remote stores for all models
    This should not be ran in a CI pipeline, rather reserved to periodic testing /
    manual tests. Can also be used to create a complete model cache.

    The cache variable `EARTH2STUDIO_CACHE` should be set before invoking

    Parameters
    ----------
    model : AutoModelMixin
        Model class that is an auto model in Earth2Studio
    model_cache_context : EnvContextManager
        Context manager that changed cache dir for CI, provided via fixture
    """
    assert issubclass(model, AutoModelMixin), "Model class needs to be an AutoModel"
    with model_cache_context():
        package = model.load_default_package()
        model.load_model(package)


@pytest.fixture(scope="session")
def cache_folder(tmp_path_factory):
    fn = tmp_path_factory.mktemp("cache")
    open(fn / "temp.txt", "w").close()
    return fn


@pytest.mark.parametrize(
    "url,file",
    [
        (None, "temp.txt"),
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


@pytest.mark.parametrize(
    "url,file,api_key",
    [
        (
            "ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0",  # Public
            "sfno_73ch_small/metadata.json",
            False,
        ),
        (
            "ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0",  # Public
            "sfno_73ch_small/metadata.json",
            True,
        ),
        (
            "ngc://models/nvstaging/simnet/physicsnemo_ci@0.1",  # Private
            "test.txt",
            True,
        ),
    ],
)
def test_ngc_package(url, file, api_key, cache_folder, model_cache_context):
    # Clear instance cache to make sure we always create a new fsspec file system
    # every test. Fsspec caches fs instances by default
    # https://github.com/fsspec/filesystem_spec/blob/master/fsspec/spec.py#L47
    NGCModelFileSystem.clear_instance_cache()
    # No API key is tested above in test_package
    current_key = ngcbase.environ.NGC_CLI_API_KEY
    with model_cache_context(
        EARTH2STUDIO_CACHE=str(cache_folder.resolve()),
        EARTH2STUDIO_PACKAGE_TIMEOUT="30",
    ):
        if api_key:
            ngcbase.environ.NGC_CLI_API_KEY = os.environ.get("NGC_CLI_API_KEY")
        else:
            ngcbase.environ.NGC_CLI_API_KEY = None
        package = Package(str(url))
        file_path = package.resolve(file)
        assert Path(file_path).is_file()
    ngcbase.environ.NGC_CLI_API_KEY = current_key


def test_ngc_filesystem():
    fs = NGCModelFileSystem()
    # fs._parse_ngc_uri("ngc://models/<org_id/team_id/model_id>@<version>")
    name, version, org, team, filepath = fs._parse_ngc_uri(
        "ngc://models/org/model_name@1.0"
    )
    assert name == "model_name"
    assert version == "1.0"
    assert org == "org"
    assert team is None
    assert filepath is None

    name, version, org, team, filepath = fs._parse_ngc_uri(
        "ngc://models/org/team/model_name@1.0/"
    )
    assert name == "model_name"
    assert version == "1.0"
    assert org == "org"
    assert team == "team"
    assert filepath is None

    name, version, org, team, filepath = fs._parse_ngc_uri(
        "ngc://models/model_name@1.0/file"
    )
    assert name == "model_name"
    assert version == "1.0"
    assert org is None
    assert team is None
    assert filepath == "file"

    with pytest.raises(ValueError):
        fs._parse_ngc_uri("ngc://models/a/b/c/d@1.0/file")
    with pytest.raises(ValueError):
        fs._parse_ngc_uri("ngc://models/a/b/c/d")
    with pytest.raises(ValueError):
        fs._parse_ngc_uri("models/a/b/c@1.0/file")

    url = fs._get_ngc_model_url("name", "1.0", authenticated_api=False)
    assert url == "https://api.ngc.nvidia.com/v2/models/name/1.0/files"

    url = fs._get_ngc_model_url(
        "name", "1.0", "org", "team", "file.txt", authenticated_api=False
    )
    assert (
        url
        == "https://api.ngc.nvidia.com/v2/models/org/org/team/team/name/1.0/files?path=file.txt"
    )

    url = fs._get_ngc_model_url(
        "name", "1.0", "org", None, "file.txt", authenticated_api=False
    )
    assert (
        url
        == "https://api.ngc.nvidia.com/v2/models/org/org/name/1.0/files?path=file.txt"
    )

    url = fs._get_ngc_model_url("name", "1.0", authenticated_api=True)
    assert url == "https://api.ngc.nvidia.com/v2/models/name/1.0/files"

    url = fs._get_ngc_model_url(
        "name", "1.0", "orgname", "teamname", "file.txt", authenticated_api=True
    )
    assert (
        url
        == "https://api.ngc.nvidia.com/v2/org/orgname/team/teamname/models/name/1.0/files?path=file.txt"
    )

    url = fs._get_ngc_model_url(
        "name", "1.0", "orgname", None, "file.txt", authenticated_api=True
    )
    assert (
        url
        == "https://api.ngc.nvidia.com/v2/org/orgname/models/name/1.0/files?path=file.txt"
    )


# Very hard to test this since the ngcbcp package does module level api variable init
@pytest.mark.parametrize(
    "url,file",
    [
        (
            "ngc://models/nvidia/modulus/sfno_73ch_small@0.1.0",  # Public
            "sfno_73ch_small/wrong-metadata.json",
        ),
        (
            "ngc://models/nvidia/earth2/test@v0.1",  # Private with no access
            "test.txt",
        ),
    ],
)
def test_ngc_package_errors(url, file, cache_folder, model_cache_context):
    # Clear instance cache to make sure we always create a new fsspec file system
    # every test. Fsspec caches fs instances by default
    # https://github.com/fsspec/filesystem_spec/blob/master/fsspec/spec.py#L47
    NGCModelFileSystem.clear_instance_cache()

    with model_cache_context(
        EARTH2STUDIO_CACHE=str(cache_folder.resolve()),
        EARTH2STUDIO_PACKAGE_TIMEOUT="30",
    ):
        with pytest.raises(http.client.HTTPException):
            package = Package(str(url))
            package.open(file)


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


async def test_ngc_unsupported_operations():
    fs = NGCModelFileSystem()
    with pytest.raises(
        NotImplementedError, match="Glob / recursive patterns not supported"
    ):
        await fs.expand_path("some/path")

    with pytest.raises(
        NotImplementedError, match="Glob / recursive patterns not supported"
    ):
        await fs.glob("some/path")

    with pytest.raises(
        NotImplementedError, match="Glob / recursive patterns not supported"
    ):
        await fs.find("some/path", maxdepth=1)
