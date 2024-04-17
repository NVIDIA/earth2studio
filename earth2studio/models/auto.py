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
import os
from typing import Any, Optional, Union

from huggingface_hub import HfFileSystem
from modulus.utils.filesystem import _download_cached


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
    cache_dir : Optional[str], optional
        Cache directory, if remote path files will be downloaded here. If none is
        provided then the path set in LOCAL_CACHE enviroment variable will be used.
        If that is empty the path "~/.cache/earth2studio" is used, by default None
    """

    def __init__(self, root: str, cache_dir: Optional[str] = None):
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

    def get(self, path: str, recursive: bool = False) -> str:
        """Get a local path to the item at ``path``

        Parameters
        ----------
        path : str
            local path of file in package directory
        recursive : bool, optional
            recursively fetch all assets under local directory. Only relevant for remote
            packages, by default False

        Returns
        -------
        str
            path to asset
        """
        # TODO upstream to modulus, change to returning the file object (let fspsec deal
        # with the cache checks) not a string? Not sure if will work with ONNX
        path = self._fullpath(path)
        if path.startswith("hf://"):
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
    ) -> Union[Any]:  # TODO: Fix types here
        """Instantiates and loads default model object from provided model package

        Parameters
        ----------
        package: Package
            Model package, file system, to fetch assets
        """
        raise NotImplementedError("Load model function not implemented")

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[str] = None
    ) -> Any:
        """Loads and instantiates a pre-trained Earth2Studio model

        Parameters
        ----------
        pretrained_model_name_or_path : Optional[str], optional
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
