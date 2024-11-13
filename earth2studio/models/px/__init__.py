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

from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.dlwp import DLWP
from earth2studio.models.px.fcn import FCN
from earth2studio.models.px.fengwu import FengWu
from earth2studio.models.px.fuxi import FuXi
from earth2studio.models.px.pangu import Pangu3, Pangu6, Pangu24
from earth2studio.models.px.persistence import Persistence
from earth2studio.models.px.sfno import SFNO
from earth2studio.models.px.stormcast import StormCast
