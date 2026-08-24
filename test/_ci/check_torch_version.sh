#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

set -Eeuo pipefail

if [[ -z "${TORCH_VERSION:-}" ]]; then
    warning="TORCH_VERSION is not set by the test container; skipping the Torch version check."
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::warning title=Torch version check skipped::${warning}"
    else
        echo "WARNING: ${warning}"
    fi
    exit 0
fi

python_version="${UV_PYTHON:-3.13.13}"
locked_torch_version="$(
    uv tree \
        --frozen \
        --no-cache \
        --quiet \
        --python-version "${python_version}" \
        --package torch \
        --depth 0 \
        | sed -n 's/^torch v//p'
)"

if [[ -z "${locked_torch_version}" ]]; then
    echo "Unable to determine the locked Torch version from uv.lock."
    exit 1
fi

if [[ "${locked_torch_version}" != "${TORCH_VERSION}" ]]; then
    echo "Torch version mismatch: container has ${TORCH_VERSION}, but uv.lock requires ${locked_torch_version}."
    exit 1
fi

echo "Test container Torch ${TORCH_VERSION} matches uv.lock."
