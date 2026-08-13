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

set -Eeuo pipefail

uv_python="${UV_PYTHON:-3.13.13}"
docs_jobs="${DOCS_JOBS:-1}"
log_dir="${DOCS_EXAMPLE_LOG_DIR:-docs/_build/example-logs}"
main_log="${log_dir}/docs-full.log"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/opt/uv-cache}"

# Keep the warmed uv package cache, but drop isolated gallery harness envs.
prune_gallery_harness_envs() {
    local environments_dir
    environments_dir="$(realpath -m "${UV_CACHE_DIR}/environments-v2")"
    if [[ -d "${environments_dir}" && "${environments_dir}" == */environments-v2 ]]; then
        find "${environments_dir}" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name 'harness-*' \
            -print \
            -exec rm -rf -- {} +
    fi
}

mkdir -p "${log_dir}"
exec > >(tee -a "${main_log}") 2>&1

echo "Full docs-full log: ${main_log}"

# Prepare the docs environment and generated metadata pages.
uv sync --python "${uv_python}" --locked --extra all --group docs
uv run python docs/generate_api.py
uv run python docs/generate_catalog.py
uv run python docs/generate_install_options.py

# Rebuild examples from source, section by section, so stale examples are refreshed.
rm -rf docs/examples examples/outputs

mapfile -t sections < <(find examples -mindepth 1 -maxdepth 1 -type d -name "[0-9]*" | sort)
for section in "${sections[@]}"; do
    selector="${section#examples/}"
    echo "::group::Build docs examples: ${selector}"
    if ! uv run e2s-gallery build "${selector}" --execute stale --jobs "${docs_jobs}"; then
        echo "::endgroup::"
        exit 1
    fi
    echo "::endgroup::"
    prune_gallery_harness_envs
done

# Render the final gallery index and build the MkDocs/Zensical site.
uv run e2s-gallery render

rm -rf docs/_build/html
E2S_GALLERY_EXECUTE=never uv run zensical build --clean
mkdir -p docs/_build
rsync -a --delete site/ docs/_build/html/
