#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run from the repository root; the CI action supplies TOX_ENV and baseline inputs.
set -eo pipefail

# Reuse the uv binary that populated the container's package/build cache.
TOX_UV_PATH="$(command -v uv)"
export TOX_UV_PATH
"$TOX_UV_PATH" --version

if [[ "$TOX_ENV" == "test-data" && "${DATA_TEST_LIVE:-none}" == "changed" ]]; then
    # Copied PR branches need their merge base, not just the latest commit.
    if [[ "$(git rev-parse --is-shallow-repository)" == "true" ]]; then
        git fetch --unshallow --no-tags origin
    fi
    # Use the previous push tip only if its commit is available locally; otherwise
    # fall back to the baseline branch's merge base (e.g. after a force-push).
    if [[ "$DATA_TEST_BASELINE_RUN" == "true" && -n "$DATA_TEST_PREVIOUS_SHA" && "$DATA_TEST_PREVIOUS_SHA" != "0000000000000000000000000000000000000000" ]] &&
        git cat-file -e "${DATA_TEST_PREVIOUS_SHA}^{commit}" 2>/dev/null; then
        DATA_TEST_BASE="$DATA_TEST_PREVIOUS_SHA"
    else
        git fetch --no-tags origin "$DATA_TEST_BASE_BRANCH"
        DATA_TEST_BASE="$(git merge-base HEAD FETCH_HEAD)"
    fi
    export DATA_TEST_BASE
fi

mkdir -p ".cache/test-results/${TOX_ENV}"
make pytest-ci 2>&1 | tee ".cache/test-results/${TOX_ENV}/output.log"
