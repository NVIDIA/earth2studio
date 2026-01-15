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

# Get track files for members 2, 3, 4
mapfile -t track_files < <(ls "$baseline_dir"/tracks_*.csv | sort | grep -E 'mem_000[234]' | xargs -n1 basename)

# Extract seeds for members 3 and 4 from filenames
seed_mem3=$(printf '%s\n' "${track_files[@]}" | grep 'mem_0003' | grep -oP 'seed_\K\d+')
seed_mem4=$(printf '%s\n' "${track_files[@]}" | grep 'mem_0004' | grep -oP 'seed_\K\d+')

# Check if seeds in reproduce_helene.yaml match, update if not
yaml_seed3=$(grep -oP "\['2024-09-24 00:00:00', 3, \K\d+" reproduce_helene.yaml)
yaml_seed4=$(grep -oP "\['2024-09-24 00:00:00', 4, \K\d+" reproduce_helene.yaml)

if [ "$seed_mem3" != "$yaml_seed3" ] || [ "$seed_mem4" != "$yaml_seed4" ]; then
    echo "WARNING: Updating seeds in reproduce_helene.yaml (mem3: $yaml_seed3->$seed_mem3, mem4: $yaml_seed4->$seed_mem4)"
    sed -i "s/\['2024-09-24 00:00:00', 3, [0-9]*\]/['2024-09-24 00:00:00', 3, $seed_mem3]/" reproduce_helene.yaml
    sed -i "s/\['2024-09-24 00:00:00', 4, [0-9]*\]/['2024-09-24 00:00:00', 4, $seed_mem4]/" reproduce_helene.yaml
fi

# Run reproduce
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo " >>> reproducing members 2, 3 and 4. might take a couple of minutes..."
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""
python ../tc_hunt.py --config-path=$(pwd)/cfg --config-name=reproduce_helene.yaml

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
echo " >>> all good, yay (: track files have been reproduced successfully"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""