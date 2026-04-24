#!/bin/bash
set -e

baseline_dir="outputs_baseline_helene/cyclone_tracks_te"
reproduce_dir="outputs_reproduce_helene/cyclone_tracks_te"

# Run baseline
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo " >>> running baseline. might take a couple of minutes..."
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""
python ../tc_hunt.py --config-path=$(pwd)/cfg --config-name=baseline_helene.yaml

# Verify that track files were produced
track_count=$(ls "$baseline_dir"/tracks_*.csv 2>/dev/null | wc -l)
if [ "$track_count" -eq 0 ]; then
    echo ""
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo " >>> ERROR: no track files produced in $baseline_dir"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo ""
    exit 1
fi
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo " >>> baseline produced $track_count track files"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""

# Get track files for members 2, 3, 4
mapfile -t track_files < <(ls "$baseline_dir"/tracks_*.csv | sort | grep -E 'mem_000[234]' | xargs -n1 basename)

# Extract seeds for members 3 and 4 from filenames
seed_mem3=$(printf '%s\n' "${track_files[@]}" | grep 'mem_0003' | grep -oP 'seed_\K\d+')
seed_mem4=$(printf '%s\n' "${track_files[@]}" | grep 'mem_0004' | grep -oP 'seed_\K\d+')

# Run reproduce (seeds injected via Hydra overrides so the config stays untouched)
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo " >>> reproducing members 2, 3 and 4. might take a couple of minutes..."
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""
python ../tc_hunt.py --config-path=$(pwd)/cfg --config-name=reproduce_helene.yaml \
    "reproduce_members.0.2=${seed_mem3}" \
    "reproduce_members.1.2=${seed_mem4}"

# Compare track files for members 3 and 4, error if any differ
for f in "${track_files[@]}"; do
    [[ $f =~ mem_000[34] ]] || continue
    if ! diff -q "$baseline_dir/$f" "$reproduce_dir/$f" > /dev/null 2>&1; then
        echo ""
        echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo " >>> oh no!!! ERROR: $f differs between baseline and reproduce"
        echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo ""
        diff "$baseline_dir/$f" "$reproduce_dir/$f"
        exit 1
    fi
done
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo " >>> all good, yay (: track files were reproduced successfully"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""
