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

docs_jobs="${DOCS_JOBS:-1}"
uv_docs=(uv run --locked --extra all --group docs)
log_dir="${DOCS_EXAMPLE_LOG_DIR:-docs/_build/example-logs}"
main_log="${log_dir}/docs-full.log"

mkdir -p "${log_dir}"
exec > >(tee -a "${main_log}") 2>&1

echo "Full docs-full log: ${main_log}"

# Generate metadata pages used by the docs build.
"${uv_docs[@]}" python docs/generate_api.py
"${uv_docs[@]}" python docs/generate_catalog.py
"${uv_docs[@]}" python docs/generate_install_options.py
"${uv_docs[@]}" python docs/generate_blog_rss.py
"${uv_docs[@]}" python docs/generate_blog_index.py

# Rebuild examples from source, section by section, so stale examples are refreshed.
rm -rf docs/examples examples/outputs

mapfile -t sections < <(find examples -mindepth 1 -maxdepth 1 -type d -name "[0-9]*" | sort)
for section in "${sections[@]}"; do
    selector="${section#examples/}"
    echo "::group::Build docs examples: ${selector}"
    if ! "${uv_docs[@]}" e2s-gallery build "${selector}" --execute stale --jobs "${docs_jobs}"; then
        echo "::endgroup::"
        exit 1
    fi
    echo "::endgroup::"
done

# Render the final gallery index and build the MkDocs/Zensical site.
"${uv_docs[@]}" e2s-gallery render

E2S_GALLERY_EXECUTE=never "${uv_docs[@]}" zensical build --clean
rm -rf site/__pycache__ site/_build/html
find site -maxdepth 1 -type f -name "*.py" -delete
