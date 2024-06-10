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

import hashlib
import io
import os
import re
from typing import Any

import fsspec
import s3fs
from fsspec.callbacks import Callback, TqdmCallback
from fsspec.compression import compr
from fsspec.core import BaseCache
from fsspec.implementations.cached import LocalTempFile, WholeFileCacheFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import infer_compression
from huggingface_hub import HfFileSystem
from loguru import logger
from modulus.utils.filesystem import _download_cached
from tqdm import tqdm

# TODO: Make this package wide? Same as in run.py
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class CallbackWholeFileCacheFileSystem(WholeFileCacheFileSystem):
    """Extension of Fsspec WholeFileCacheFileSystem to include callback function when
    downloading files to cache (progress bar).

    See: https://github.com/fsspec/filesystem_spec/blob/8be9763e5f895073a9f46c8147aebbc64933e013/fsspec/implementations/cached.py#L651
    """

    def _open(self, path, mode="rb", **kwargs):  # type: ignore
        path = self._strip_protocol(path)
        if "r" not in mode:
            hash = self._mapper(path)
            fn = os.path.join(self.storage[-1], hash)
            user_specified_kwargs = {
                k: v
                for k, v in kwargs.items()
                # those kwargs were added by open(), we don't want them
                if k not in ["autocommit", "block_size", "cache_options"]
            }
            return LocalTempFile(self, path, mode=mode, fn=fn, **user_specified_kwargs)
        detail = self._check_file(path)
        if detail:
            detail, fn = detail
            _, blocks = detail["fn"], detail["blocks"]
            if blocks is True:
                f = open(fn, mode)
                f.original = detail.get("original")
                return f
            else:
                raise ValueError(
                    f"Attempt to open partially cached file {path}"
                    f" as a wholly cached file"
                )
        else:
            fn = self._make_local_details(path)
        kwargs["mode"] = mode

        # call target filesystems open
        self._mkcache()
        if self.compression:
            with self.fs._open(path, **kwargs) as f, open(fn, "wb") as f2:
                if isinstance(f, AbstractBufferedFile):
                    # want no type of caching if just downloading whole thing
                    f.cache = BaseCache(0, f.cache.fetcher, f.size)
                comp = (
                    infer_compression(path)
                    if self.compression == "infer"
                    else self.compression
                )
                f = compr[comp](f, mode="rb")
                data = True
                while data:
                    block = getattr(f, "blocksize", 5 * 2**20)
                    data = f.read(block)
                    f2.write(data)  # type: ignore
        else:
            if "callback" in kwargs:  # Patch here
                self.fs.get_file(path, fn, callback=kwargs["callback"])
            else:
                self.fs.get_file(path, fn)
        self.save_cache()
        return self._open(path, mode)  # type: ignore


class TqdmFormat(tqdm):
    """Provides a `total_time` format parameter. Not used.
    See: https://filesystem-spec.readthedocs.io/en/stable/api.html#fsspec.callbacks.TqdmCallback
    """

    @property
    def format_dict(self) -> dict:
        d = super().format_dict
        # total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        # d.update(total_time=self.format_interval(total_time) + " in total")
        # d.update(total=round(d["total"] / 10**9), n=round(d["n"] / 10**9))
        # d.update(n_fmt=f"{round(d['n'] / 10**9)}{d['unit']}")
        return d


class TqdmCallbackRelative(TqdmCallback):
    """Simple extention of Tqdm callback to support progress bar on recurrive gets"""

    def branched(self, path_1, path_2, **kwargs) -> Callback:  # type: ignore
        tqdm_kwargs = self._tqdm_kwargs
        tqdm_kwargs["unit"] = "B"
        tqdm_kwargs["unit_scale"] = True
        tqdm_kwargs["unit_divisor"] = 1024
        callback = TqdmCallback(
            tqdm_kwargs=tqdm_kwargs,
            tqdm_cls=self._tqdm_cls,
        )
        return callback


class PackageV2:
    """A generic file system abstraction with local caching, uses FSSpec
    WholeFileCacheFileSystem to manage files. Designed to be used for accessing remote
    resources, particularly checkpoint files for pre-trained models. Presently supports
    public folders on NGC, huggingface repos, s3 and any other built in file system
    Fsspec supports.

    Parameters
    ----------
    root : str
        Root directory for file system
    fs : AbstractFileSystem | None, optional
        The target filesystem to run under neath. If none is provided one will get
        initialized based on root url, by default None
    cache_options : dict, optional
        Caching options provided to Fsspec. See CachingFileSystem in fsspec to see
        valid options https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.cached.CachingFileSystem,
        by default {}
    """

    def __init__(
        self, root: str, fs: AbstractFileSystem | None = None, cache_options: dict = {}
    ):

        self.cache_options = cache_options.copy()
        if "cache_storage" not in self.cache_options:
            self.cache_options["cache_storage"] = os.environ.get(
                "EARTH2STUDIO_CACHE", self.default_cache()
            )

        self.root = root

        if fs is not None:
            pass
        elif root.startswith("ngc://models/"):
            # Taken from Modulus file utils
            # Strip ngc model url prefix
            suffix = "ngc://models/"
            # The regex check
            pattern = re.compile(rf"{suffix}[\w-]+(/[\w-]+)?/[\w-]+@[A-Za-z0-9.]+")
            if not pattern.match(root):
                raise ValueError(
                    "Invalid URL, should be of form ngc://models/<org_id/team_id/model_id>@<version>\n"
                    + f" got {root}"
                )
            root = root.replace(suffix, "")
            if len(root.split("@")[0].split("/")) == 3:
                (org, team, model_version) = root.split("/", 2)
                (model, version) = model_version.split("@", 1)
            else:
                (org, model_version) = root.split("/", 1)
                (model, version) = model_version.split("@", 1)
                team = None
            if team:
                self.root = f"https://api.ngc.nvidia.com/v2/models/{org}/{team}/{model}/versions/{version}/files/"
            else:
                self.root = f"https://api.ngc.nvidia.com/v2/models/{org}/{model}/versions/{version}/files/"
            fs = HTTPFileSystem(block_size=2**10)
        elif root.startswith("hf://"):
            fs = HfFileSystem(target_options={"default_block_size": 2**20})
        elif root.startswith("s3://"):
            fs = s3fs.S3FileSystem(
                anon=True,
                client_kwargs={},
                target_options={"default_block_size": 2**20},
            )
        else:
            fs = fsspec.filesystem(root)

        self.fs = CallbackWholeFileCacheFileSystem(fs=fs, **self.cache_options)

    @classmethod
    def default_cache(cls, path: str = "") -> str:
        """Default cache location for packages located in `~/.cache/earth2studio`

        Parameters
        ----------
        path : str, optional
            Sub-path in cache direction, by default ""

        Returns
        -------
        str
            Local cache path
        """
        default_cache = os.path.join(os.path.expanduser("~"), ".cache", "earth2studio")
        return os.path.join(default_cache, path)

    @property
    def cache(self) -> str:
        """Cache path"""
        return self.cache_options["cache_storage"]

    def open(self, file_path: str) -> io.BufferedReader:
        """Open file inside package, caching it to local cache store in the process

        Parameters
        ----------
        file_path : str
            Local file path in package directory

        Returns
        -------
        io.BufferedReader
            Opened file, can get file path with BufferedReader.name
        """
        full_path = os.path.join(self.root, file_path)
        filename = os.path.basename(full_path)

        with TqdmCallbackRelative(
            tqdm_kwargs={
                "desc": "Earth2Studio Package Download",
                "bar_format": f"Downloading {filename}: "
                + "{percentage:.0f}%|{bar}{r_bar}",
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,
            },
            tqdm_cls=TqdmFormat,
        ) as callback:
            return self.fs.open(full_path, callback=callback)

    def resolve(self, file_path: str) -> str:
        """Resolves current relative file path to absolute path inside Package cache

        Parameters
        ----------
        path : str
            local path of file in package directory

        Returns
        -------
        str
            File path inside cache
        """
        local_file_path = ""
        with self.open(file_path) as file:
            local_file_path = file.name
        return local_file_path


class Package:
    """A generic file system abstraction

    Note
    ----
    This is a slight variation of the Modulus Package... figured upstream was too much
    of a pain right now.

    Parameters
    ----------
    root : str
        Root directory for file system
    cache_dir : str, optional
        Cache directory, if remote path files will be downloaded here. If none is
        provided then the path set in LOCAL_CACHE enviroment variable will be used.
        If that is empty the path "~/.cache/earth2studio" is used, by default None
    """

    def __init__(self, root: str, cache_dir: str | None = None):
        self.root = root
        self.cache_dir = cache_dir
        if not self.cache_dir:
            # TODO: Update to earth2 studio after fix to
            # https://github.com/NVIDIA/modulus/blob/main/modulus/utils/filesystem.py#L101
            # Is upstreamed
            default_cache = os.path.join(os.path.expanduser("~"), ".cache", "modulus")
            self.cache_dir = os.environ.get("EARTH2STUDIO_CACHE", default_cache)
        # Create cache dir if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def get(self, path: str, recursive: bool = False, same_names: bool = False) -> str:
        """Get a local path to the item at ``path``

        Parameters
        ----------
        path : str
            local path of file in package directory
        recursive : bool, optional
            recursively fetch all assets under local directory. Only relevant for remote
            packages, by default False
        same_names : bool, optional
            If true, file names will not be hashed to avoid potential conflicts. By
            default False

        Returns
        -------
        str
            path to asset
        """
        # TODO upstream to modulus, change to returning the file object (let fspsec deal
        # with the cache checks) not a string? Not sure if will work with ONNX
        path = self._fullpath(path)
        if path.startswith("hf://"):
            # TODO: Temp fix, needs better support
            if same_names:
                filename = os.path.basename(path)
            else:
                sha = hashlib.sha256(path.encode())
                filename = sha.hexdigest()
            cache_path = os.path.join(str(self.cache_dir), filename)
            fs = HfFileSystem(target_options={"default_block_size": 2**20})
            if not os.path.isfile(cache_path):
                fs.get(path, cache_path, recursive=recursive)
            return cache_path
        else:
            return _download_cached(
                path, recursive=recursive, local_cache_path=self.cache_dir
            )

    def _fullpath(self, path: str) -> str:
        return os.path.join(self.root, path)


class AutoModelMixin:
    """Abstract class that defines the utils needed auto loading / instantiating models"""

    @classmethod
    def load_default_package(cls) -> Package:
        """Loads the default model package

        Returns
        -------
        Package
            Model package, file system, object
        """
        raise NotImplementedError("No default package supported")

    @classmethod
    def load_model(
        cls,
        package: Package,
    ) -> Any:  # TODO: Fix types here
        """Instantiates and loads default model object from provided model package

        Parameters
        ----------
        package: Package
            Model package, file system, to fetch assets
        """
        raise NotImplementedError("Load model function not implemented")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | None = None) -> Any:
        """Loads and instantiates a pre-trained Earth2Studio model

        Parameters
        ----------
        pretrained_model_name_or_path : str, optional
            Path to model package (file system). If none is provided, the built in
            package will be used if provide, by default None. Valid inputs include:
            - A path to a directory containing model weights saved e.g.,
                ./my_model_directory/.
            - A path or url/uri to a remote file system supported by Fsspec
            - A s3 uri supported by s3fs
            - A NGC model registry uri

        Returns
        -------
        Union[PrognosticModel, Diagnostic]
            Instantiated model with loaded checkpoint from loaded model package
        """
        if pretrained_model_name_or_path is None:
            package = cls.load_default_package()
        else:
            package = Package(pretrained_model_name_or_path)

        return cls.load_model(package)
