#!/bin/bash
# Generate a small Helene ensemble with cyclone tracking, reproduce two of its
# members and check that their tracks are identical.
#
#   ./test_tc_hunt.sh          # FCN3
#   ./test_tc_hunt.sh aifs     # AIFS-ENS v1 (aifs-v2 for v2)
set -e

model=${1:-fcn3}
case "$model" in
    # FCN3 is reproducible as is and batches members efficiently
    fcn3) overrides="model=fcn3 batch_size=2" ;;
    # AIFS-ENS needs deterministic GPU kernels to be reproducible, and one
    # member per batch keeps it within 48 GB of GPU memory
    aifs|aifs-v2) overrides="model=$model batch_size=1 deterministic=True" ;;
    *) echo "unknown model $model, choose from fcn3, aifs, aifs-v2" >&2; exit 1 ;;
esac

baseline_dir="outputs_baseline_helene/cyclone_tracks_te"
reproduce_dir="outputs_reproduce_helene/cyclone_tracks_te"
rm -rf outputs_baseline_helene outputs_reproduce_helene

banner() {
    echo ""
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo " >>> $1"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo ""
}

banner "running baseline with $model. might take a couple of minutes..."
python ../main.py --config-path="$(pwd)/cfg" --config-name=baseline_helene.yaml $overrides

track_count=$(ls "$baseline_dir"/tracks_*.csv 2>/dev/null | wc -l)
if [ "$track_count" -eq 0 ]; then
    banner "ERROR: no track files produced in $baseline_dir"
    exit 1
fi
banner "baseline produced $track_count track files"

# seeds of members 3 and 4 from the track file names
seed_mem3=$(ls "$baseline_dir" | grep 'mem_0003' | grep -oP 'seed_\K\d+')
seed_mem4=$(ls "$baseline_dir" | grep 'mem_0004' | grep -oP 'seed_\K\d+')

# seeds injected via Hydra overrides, so the config stays untouched
banner "reproducing members 3 and 4 (and the members sharing their batches)..."
python ../main.py --config-path="$(pwd)/cfg" --config-name=reproduce_helene.yaml $overrides \
    "reproduce_members.0.2=${seed_mem3}" \
    "reproduce_members.1.2=${seed_mem4}"

# every reproduced member has to match the baseline
for f in "$reproduce_dir"/tracks_*.csv; do
    name=$(basename "$f")
    if ! diff -q "$baseline_dir/$name" "$f" > /dev/null 2>&1; then
        banner "oh no!!! ERROR: $name differs between baseline and reproduce"
        diff "$baseline_dir/$name" "$f"
        exit 1
    fi
done

banner "all good, yay (: $(ls "$reproduce_dir"/tracks_*.csv | wc -l) track files were reproduced successfully"
