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

"""Load GDAL before the ECMWF libraries.

On Linux, importing ``eccodes`` (and so ``cfgrib``) before ``rasterio`` makes
every GDAL ``/vsicurl`` read, which :mod:`earth2studio.data.stac` relies on,
segfault: the wheels bundle conflicting copies of the HTTP stack and GDAL ends
up bound to the wrong one. Importing rasterio first pins GDAL's own libraries.
This module exists only to be imported first by :mod:`earth2studio.data`.
"""

try:
    import rasterio  # type: ignore[import-untyped]  # noqa: F401
except ImportError:  # rasterio belongs to the optional "data" extra
    pass
