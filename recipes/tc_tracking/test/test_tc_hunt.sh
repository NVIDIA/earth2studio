#!/bin/bash
set -e

baseline_dir="outputs_baseline_helene/cyclone_tracks_te"

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
echo " >>> all good, yay (: $track_count track files produced successfully"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""
