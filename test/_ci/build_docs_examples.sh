#!/bin/bash
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

set -euo pipefail

docs_jobs="${DOCS_JOBS:-4}"
uv_python="${UV_PYTHON:-3.13.13}"

uv sync --python "${uv_python}" --locked --extra all --group docs

# Start from a clean docs tree once, then preserve generated outputs while each
# example is built with its own Sphinx Gallery filename pattern.
rm -rf docs/examples
rm -rf docs/modules/generated
rm -rf docs/modules/backreferences
rm -rf examples/outputs

uv run make -C docs clean
uv run make -C docs html

mapfile -t sections < <(
    find examples -mindepth 1 -maxdepth 1 -type d -name "[0-9]*" | sort
)

for section in "${sections[@]}"; do
    mapfile -t examples < <(
        find "${section}" -maxdepth 1 -type f -name "[0-9]*.py" | sort
    )

    if [ "${#examples[@]}" -eq 0 ]; then
        continue
    fi

    echo "::group::Build docs examples: ${section}"
    for example in "${examples[@]}"; do
        relative_example="${example#examples/}"
        filename_pattern="/${relative_example//./\\.}$"

        echo "Building docs example: ${example}"
        PLOT_GALLERY=True \
            RUN_STALE_EXAMPLES=True \
            FILENAME_PATTERN="${filename_pattern}" \
            uv run make -j "${docs_jobs}" -C docs html
    done
    echo "::endgroup::"
done

# Refresh the final HTML without cleaning or re-running examples.
uv run make -C docs html
