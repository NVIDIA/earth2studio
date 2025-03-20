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

import io
import os
import re
import warnings

import aiohttp
import fsspec
import s3fs
from fsspec.callbacks import Callback, TqdmCallback
from fsspec.compression import compr
from fsspec.core import BaseCache, split_protocol
from fsspec.implementations.cached import LocalTempFile, WholeFileCacheFileSystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import infer_compression
from huggingface_hub import HfFileSystem
from loguru import logger
from tqdm import tqdm

from earth2studio.models.auto.ngc import NGCModelFileSystem

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
        return d


class TqdmCallbackRelative(TqdmCallback):
    """Simple extention of Tqdm callback to support progress bar on recurrive gets"""

    def branched(self, path_1, path_2, **kwargs) -> Callback:  # type: ignore
        """Child callback for recursive get"""
        tqdm_kwargs = self._tqdm_kwargs
        tqdm_kwargs["unit"] = "B"
        tqdm_kwargs["unit_scale"] = True
        tqdm_kwargs["unit_divisor"] = 1024
        callback = TqdmCallback(
            tqdm_kwargs=tqdm_kwargs,
            tqdm_cls=self._tqdm_cls,
        )
        return callback


class Package:
    """A generic file system abstraction with local caching, uses Fsspec
    WholeFileCacheFileSystem to manage files. Designed to be used for accessing remote
    resources, particularly checkpoint files for pre-trained models. Presently supports
    public folders on NGC, huggingface repos, s3 and any other built in file system
    Fsspec supports.

    Parameters
    ----------
    root : str
        Root directory for file system
    fs : AbstractFileSystem | None, optional
        The target filesystem to run underneath. If none is provided a fsspec filesystem
        will get initialized based on the protocal of the root url, by default None
    cache : bool, optional
        Toggle local caching, typically you want this to be true unless the package is
        a local file system, by default True
    cache_options : dict, optional
        Caching options provided to Fsspec. See CachingFileSystem in fsspec for
        valid options https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.cached.CachingFileSystem,
        by default {}
    """

    def __init__(
        self,
        root: str,
        fs: AbstractFileSystem | None = None,
        cache: bool = True,
        cache_options: dict = {},
    ):

        self.cache_options = cache_options.copy()
        if "cache_storage" not in self.cache_options:
            self.cache_options["cache_storage"] = self.default_cache()
        if "expiry_time" not in self.cache_options:
            self.cache_options["expiry_time"] = 31622400  # 1 year

        self.root = root

        if fs is not None:
            self.fs = fs
        elif root.startswith("ngc://models/"):
            # Taken from PhysicsNeMo file utils
            # Strip ngc model url prefix
            suffix = "ngc://models/"
            # The regex check
            pattern = re.compile(rf"{suffix}[\w-]+(/[\w-]+)?/[\w-]+@[A-Za-z0-9.]+")
            if not pattern.match(root):
                raise ValueError(
                    "Invalid URL, should be of form ngc://models/<org_id/team_id/model_id>@<version>\n"
                    + f" got {root}"
                )
            self.root = root
            self.fs = NGCModelFileSystem(  # type: ignore
                block_size=Package.default_blocksize(),
                client_kwargs={
                    "timeout": aiohttp.ClientTimeout(total=Package.default_timeout())
                },
            )
        elif root.startswith("hf://"):
            # https://github.com/huggingface/huggingface_hub/blob/v0.23.4/src/huggingface_hub/hf_file_system.py#L816
            if "HF_HUB_DOWNLOAD_TIMEOUT" not in os.environ:
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(Package.default_timeout())
            self.fs = HfFileSystem(
                target_options={"default_block_size": Package.default_blocksize()}
            )
        elif root.startswith("s3://"):
            self.fs = s3fs.S3FileSystem(
                anon=True,
                client_kwargs={},
                default_block_size=Package.default_blocksize(),
            )
            self.fs.read_timeout = Package.default_timeout()
        else:
            protocol = split_protocol(root)[0]
            self.fs = fsspec.filesystem(protocol)

        if cache:
            self.fs = CallbackWholeFileCacheFileSystem(fs=self.fs, **self.cache_options)

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
        default_cache = os.environ.get("EARTH2STUDIO_CACHE", default_cache)
        return os.path.join(default_cache, path)

    @classmethod
    def default_timeout(cls) -> int:
        """Default remote store timeout in seconds

        Returns
        -------
        int
            Time out in seconds
        """
        default_timeout = 300
        try:
            timeout = os.environ.get("EARTH2STUDIO_PACKAGE_TIMEOUT", default_timeout)
            default_timeout = int(timeout)
        except ValueError:
            pass
        return default_timeout

    @classmethod
    def default_blocksize(cls) -> int:
        """Default remote store block size

        Returns
        -------
        int
            Download block size in bytes
        """
        return 2**20

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
            try:
                return self.fs.open(full_path, callback=callback)
            except fsspec.exceptions.FSTimeoutError as e:
                logger.error(
                    f"Package fetch timeout. Consider increasing timeout through environment variable 'EARTH2STUDIO_PACKAGE_TIMEOUT'. Currently {self.default_timeout()}s."
                )
                raise e

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
        # WARNING: THIS CAN FAIL IF FILE DOES NOT HAVE NAME ATTRIB. BUFFERED FILE TYPES
        # ARE NOT SUPPORTED HERE. NEED TO LOOK INTO THIS MORE.
        local_file_path = ""
        with self.open(file_path) as file:
            local_file_path = file.name
        return local_file_path

    def get(self, file_path: str) -> str:
        """PhysicsNeMo / backwards compatibility

        Parameters
        ----------
        path : str
            local path of file in package directory

        Returns
        -------
        str
            File path inside cache
        """
        warnings.warn(
            "Package.get(path) deprecated. Use Package.resolve(path) instead."
        )
        return self.resolve(file_path)
