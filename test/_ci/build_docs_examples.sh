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

uv_python="${UV_PYTHON:-3.13.13}"
docs_jobs="${DOCS_JOBS:-1}"

uv sync --python "${uv_python}" --locked --extra all --group docs
uv run python docs/generate_api.py

rm -rf docs/examples examples/outputs

mapfile -t sections < <(
    find examples -mindepth 1 -maxdepth 1 -type d -name "[0-9]*" | sort
)

for section in "${sections[@]}"; do
    selector="${section#examples/}"
    echo "::group::Build MkDocs examples: ${selector}"
    uv run e2s-gallery build "${selector}" --execute stale --jobs "${docs_jobs}"
    echo "::endgroup::"
done

uv run e2s-gallery render
rm -rf docs/_build/html
E2S_GALLERY_EXECUTE=never uv run mkdocs build --clean --site-dir site
mkdir -p docs/_build
rsync -a --delete site/ docs/_build/html/
