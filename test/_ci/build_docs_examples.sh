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
current_step="initializing"
current_selector=""
current_section_log=""

print_gallery_failures() {
    local selector="$1"
    if [ -z "${selector}" ]; then
        echo "No active gallery selector was recorded."
        return
    fi
    uv run python - "${selector}" <<'PY'
import json
import sys
from pathlib import Path

selector = sys.argv[1].replace("/", "-")
runs = Path(".e2sgallery/runs")
if not runs.exists():
    print("No .e2sgallery run manifests were found.")
    raise SystemExit

failures = []
for manifest in runs.glob(f"{selector}*/manifest.json"):
    try:
        result = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if int(result.get("returncode", 0)) != 0:
        failures.append((manifest.parent.name, result))

if not failures:
    print(f"No failed retained runs found for selector {sys.argv[1]!r}.")
    raise SystemExit

for name, result in sorted(failures):
    print(f"\n{name}: exited {result.get('returncode')}")
    error = str(result.get("error") or "No error text was captured.").strip()
    if error:
        print("\nProcess error tail:")
        print(error[-8000:])
    for index, event in enumerate(result.get("events", [])):
        stdout = str(event.get("stdout") or "").strip()
        stderr = str(event.get("stderr") or "").strip()
        if not stdout and not stderr:
            continue
        print(f"\nCell/event {index} output tail:")
        if stdout:
            print("\nstdout:")
            print(stdout[-8000:])
        if stderr:
            print("\nstderr:")
            print(stderr[-8000:])
PY
}

on_error() {
    local status=$?
    echo
    echo "::error::Earth2Studio docs-full failed during '${current_step}' at line ${BASH_LINENO[0]} with exit code ${status}"
    echo "Full docs-full log: ${main_log}"
    if [ -n "${current_section_log}" ] && [ -f "${current_section_log}" ]; then
        echo
        echo "Last 200 lines from ${current_section_log}:"
        tail -n 200 "${current_section_log}" || true
    fi
    echo
    echo "Retained gallery failure details:"
    print_gallery_failures "${current_selector}" || true
    exit "${status}"
}

mkdir -p "${log_dir}"
exec > >(tee -a "${main_log}") 2>&1
trap on_error ERR

echo "Full docs-full log: ${main_log}"

current_step="sync docs environment"
uv sync --python "${uv_python}" --locked --extra all --group docs

current_step="generate API markdown"
uv run python docs/generate_api.py

current_step="generate catalog metadata"
uv run python docs/generate_catalog.py

current_step="generate install options"
uv run python docs/generate_install_options.py

current_step="clean generated example outputs"
rm -rf docs/examples examples/outputs

current_step="discover example sections"
mapfile -t sections < <(
    find examples -mindepth 1 -maxdepth 1 -type d -name "[0-9]*" | sort
)

for section in "${sections[@]}"; do
    selector="${section#examples/}"
    log_file="${log_dir}/${selector//\//-}.log"
    current_step="build docs examples: ${selector}"
    current_selector="${selector}"
    current_section_log="${log_file}"
    echo "::group::Build docs examples: ${selector}"
    set +e
    uv run e2s-gallery build "${selector}" --execute stale --jobs "${docs_jobs}" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e
    echo "::endgroup::"
    if [ "${status}" -ne 0 ]; then
        echo "::error::Earth2Studio gallery build failed for ${selector}"
        echo "Full section log: ${log_file}"
        echo
        echo "Last 200 lines from ${log_file}:"
        tail -n 200 "${log_file}" || true
        echo
        echo "Retained gallery failure details:"
        print_gallery_failures "${selector}" || true
        exit "${status}"
    fi
done

current_step="render complete gallery"
current_selector=""
current_section_log=""
uv run e2s-gallery render

current_step="clean docs HTML"
rm -rf docs/_build/html

current_step="build Zensical site"
E2S_GALLERY_EXECUTE=never uv run zensical build --clean

current_step="repair API cross-references"

current_step="copy built site to docs/_build/html"
mkdir -p docs/_build
rsync -a --delete site/ docs/_build/html/
