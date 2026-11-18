#!/usr/bin/env bash
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

# torchrun --no-python entry used when SCORECARD_PIN_GPU=1 (run_scorecard.py).
# Makes only this rank's GPU visible before CUDA starts, so every library and
# worker thread of the rank lands on it.  cudaSetDevice is per host thread:
# in an 8-rank HealDA run without this, each other rank held about 2.6 GiB on
# the first GPU, which runs local rank 0 out of memory in long jobs.  The GPU
# is the rank's entry of an inherited CUDA_VISIBLE_DEVICES list (indices or
# UUIDs), else the local rank itself.  The rank then sees one device, so
# LOCAL_RANK becomes 0 for the recipe's cuda:{LOCAL_RANK}.
set -euo pipefail
rank="${LOCAL_RANK:-0}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible <<< "${CUDA_VISIBLE_DEVICES}"
    if (( rank >= ${#visible[@]} )); then
        echo "rank_entry.sh: LOCAL_RANK=${rank}, but CUDA_VISIBLE_DEVICES lists" \
            "${#visible[@]} device(s)" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="${visible[rank]}"
else
    export CUDA_VISIBLE_DEVICES="${rank}"
fi
export LOCAL_RANK=0
# The interpreter that runs run_scorecard.py, which a bare "python" from PATH
# need not be.
exec "${SCORECARD_PYTHON:-python}" "$@"
