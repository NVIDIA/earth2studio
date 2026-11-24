#!/bin/bash
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
    fcn3) batch_size=2 ;;
    # one member per batch keeps AIFS-ENS within 48 GB of GPU memory
    aifs) batch_size=1 ;;
    *) echo "unknown model $model, choose from fcn3, aifs" >&2; exit 1 ;;
esac
members=3
max_retries=1
batches=$(( (members + batch_size - 1) / batch_size ))
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
    project=retries model="$model" ensemble_size="$members" batch_size="$batch_size" \
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

python - "$out_dir" "$members" << 'EOF' || fail "fields or seeds of failed batches are wrong, see above"
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
