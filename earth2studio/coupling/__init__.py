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

"""Graph authoring for coupled and impact workflows: wiring components together.

Components are engine types in :mod:`earth2studio.run.component`; this package
exports only what connects them. Runtime internals -- ``ExecutionGraph``,
``BoundProvider``, ``ActivationContext``, ``GraphSnapshot``,
``ConnectorSnapshot`` -- remain in :mod:`earth2studio.coupling.contracts` and
carry no stability promise. Running a model needs none of this.
"""

from earth2studio.coupling.contracts import Binding, ProviderRegistry, TransformSpec

__all__ = ["Binding", "ProviderRegistry", "TransformSpec"]
