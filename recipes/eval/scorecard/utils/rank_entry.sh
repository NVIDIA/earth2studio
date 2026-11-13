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

#!/usr/bin/env bash
# torchrun --no-python entry used when SCORECARD_PIN_GPU=1 (run_scorecard.py).
# Pins each rank to its own GPU before CUDA starts, so every library, worker
# thread and default "cuda" device of the rank lands on that GPU; without it,
# cudaSetDevice is per host thread and stray allocations pile up on GPU 0,
# which runs local rank 0 out of memory in large jobs.  PhysicsNeMo derives
# cuda:{LOCAL_RANK}, so with one visible device the local rank is 0.
export CUDA_VISIBLE_DEVICES="${LOCAL_RANK:-0}"
export LOCAL_RANK=0
exec python "$@"
