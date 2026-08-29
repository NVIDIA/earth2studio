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

readonly shard_timeout="${E2S_LIVE_SHARD_TIMEOUT:-15m}"
readonly kill_after="${E2S_LIVE_KILL_AFTER:-15s}"
readonly diagnostic_timeout="${E2S_LIVE_DIAGNOSTIC_TIMEOUT:-600}"
readonly results_dir="${E2S_LIVE_RESULTS_DIR:-${RUNNER_TEMP:-/tmp}/earth2studio-live-results}"
readonly test_file="test/data/test_live_contracts.py"
readonly shards=(object_store multiprocess)

mkdir -p "${results_dir}"

selector_for() {
    case "$1" in
        object_store)
            echo "gfs or ghcn_daily"
            ;;
        multiprocess)
            echo "ufs_obs_conv"
            ;;
        *)
            echo "Unknown live data shard: $1" >&2
            return 2
            ;;
    esac
}

sweep_process_group() {
    local group_id="$1"

    if ! kill -0 -- "-${group_id}" 2>/dev/null; then
        return
    fi

    echo "[live-data] cleaning descendant processes in group ${group_id}"
    kill -TERM -- "-${group_id}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${group_id}" 2>/dev/null || true
}

run_shard() {
    local shard="$1"
    local selector
    local log_file="${results_dir}/${shard}.log"
    local junit_file="${results_dir}/${shard}.xml"
    local status
    local timeout_pid

    selector="$(selector_for "${shard}")"
    echo "[live-data] shard=${shard} selector=${selector} timeout=${shard_timeout}"

    set +e
    EARTH2STUDIO_LIVE_CONTRACTS=1 timeout \
        --signal=TERM \
        --kill-after="${kill_after}" \
        "${shard_timeout}" \
        python -m pytest \
        -p no:timeout \
        -s \
        -o "faulthandler_timeout=${diagnostic_timeout}" \
        -m live_network \
        -k "${selector}" \
        --tb=short \
        --junitxml="${junit_file}" \
        "${test_file}" >"${log_file}" 2>&1 &
    timeout_pid=$!
    wait "${timeout_pid}"
    status=$?
    set -e

    # GNU timeout stops monitoring after its direct child exits. Sweep the
    # process group as well so a child cannot survive an early pytest exit.
    sweep_process_group "${timeout_pid}"

    echo "::group::Live data shard: ${shard}"
    cat "${log_file}"
    echo "::endgroup::"

    if [[ "${status}" -ne 0 ]]; then
        echo "[live-data] shard=${shard} failed with status ${status}" >&2
    fi
    return "${status}"
}

if [[ "$#" -gt 1 ]]; then
    echo "Usage: $0 [object_store|multiprocess]" >&2
    exit 2
fi

if [[ "$#" -eq 1 ]]; then
    run_shard "$1"
    exit $?
fi

pids=()
for shard in "${shards[@]}"; do
    run_shard "${shard}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
