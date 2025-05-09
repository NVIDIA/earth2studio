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

from typing import Any

from earth2studio.models.auto.package import Package


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
        **kwargs
            Addition keyword arguments are allowed, must have defaults
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
