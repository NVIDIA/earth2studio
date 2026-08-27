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
uv_docs=(uv run --locked --group docs)
log_dir="${DOCS_EXAMPLE_LOG_DIR:-docs/_build/example-logs}"
main_log="${log_dir}/docs-examples.log"
requested_section="${1:-}"

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [SECTION]" >&2
    exit 2
fi
if [ -n "${requested_section}" ]; then
    if [[ ! "${requested_section}" =~ ^[0-9][0-9]_[A-Za-z0-9_-]+$ ]]; then
        echo "Invalid example section: ${requested_section}" >&2
        exit 2
    fi
    if [ ! -d "examples/${requested_section}" ]; then
        echo "Example section does not exist: ${requested_section}" >&2
        exit 2
    fi
fi

mkdir -p "${log_dir}"
exec > >(tee -a "${main_log}") 2>&1

echo "Example execution log: ${main_log}"

# Rebuild examples from source, section by section, so stale examples are refreshed.
rm -rf docs/examples examples/outputs

if [ -n "${requested_section}" ]; then
    sections=("examples/${requested_section}")
else
    mapfile -t sections < <(
        find examples -mindepth 1 -maxdepth 1 -type d -name "[0-9]*" | sort
    )
fi

for section in "${sections[@]}"; do
    selector="${section#examples/}"
    echo "::group::Build docs examples: ${selector}"
    if ! "${uv_docs[@]}" e2s-gallery build "${selector}" --execute stale --jobs "${docs_jobs}"; then
        echo "::endgroup::"
        exit 1
    fi
    echo "::endgroup::"
done
