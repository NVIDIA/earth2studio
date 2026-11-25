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

# Check the retries of unstable batches on one GPU. With thresholds no forecast
# can meet, every attempt of every batch is unstable, so each batch has to be
# attempted max_retries + 1 times and then end as failed: recorded as such,
# with NaN fields, seed -1 and no tracks.
#
#   ./test_retries.sh          # FCN3
#   ./test_retries.sh aifs     # AIFS-ENS v1
set -e

model=${1:-fcn3}
case "$model" in
    fcn3 | aifs) ;;
    *) echo "unknown model $model, choose from fcn3, aifs" >&2; exit 1 ;;
esac
# one member per batch, which also keeps AIFS-ENS within 48 GB of GPU memory
batches=3
max_retries=1
out_dir="outputs_retries"
rm -rf "$out_dir"

banner() {
    echo ""
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo " >>> $1"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo ""
}

fail() {
    banner "oh no!!! ERROR: $1"
    exit 1
}

banner "running $batches batches with $model, every attempt unstable..."
python ../main.py --config-path="$(pwd)/cfg" --config-name=baseline_helene.yaml \
    project=retries model="$model" ensemble_size="$batches" batch_size=1 \
    n_steps=4 store_type=netcdf "+out_vars=[msl]" \
    "stability_check.thresholds=[1e-6,1e-6,1e-6]" stability_check.max_retries="$max_retries"

# every attempt is recorded, so the last attempt of each batch shows it was retried
attempts=$((max_retries + 1))
exhausted=$(cat "$out_dir"/retries_progress/failed-*.log | grep -c "\"attempts\": $attempts" || true)
if [ "$exhausted" -ne "$batches" ]; then
    fail "$exhausted of $batches batches were recorded as failed after $attempts attempts"
fi

if ls "$out_dir"/cyclone_tracks_te/tracks_*.csv > /dev/null 2>&1; then
    fail "failed batches have track files"
fi

python - "$out_dir" "$batches" << 'EOF' || fail "fields or seeds of failed batches are wrong, see above"
import glob
import sys

import xarray as xr

members = 0
for file in sorted(glob.glob(f"{sys.argv[1]}/retries_*.nc")):
    with xr.open_dataset(file) as ds:
        assert (ds.random_seed == -1).all(), f"{file}: seeds {ds.random_seed.values}"
        assert ds.msl.isnull().all(), f"{file}: holds fields of an unstable attempt"
        members += ds.sizes["ensemble"]
assert members == int(sys.argv[2]), f"{members} members in the output"
EOF

banner "all good, yay (: $batches batches were retried and ended as failed"
