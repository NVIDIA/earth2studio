#!/bin/bash
set -e


# Run extraction
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo " >>> running baseline. might take a couple of minutes to download data..."
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""
python ../tc_hunt.py --config-path=$(pwd)/cfg --config-name=extract_era5.yaml


# Compare reference track files between ground truth and produced output
reference_tracks=(
    "reference_track_hato_2017_west_pacific.csv"
    "reference_track_helene_2024_north_atlantic.csv"
)
for f in "${reference_tracks[@]}"; do
    if ! diff -q "../aux_data/$f" "outputs_reference_tracks/$f" > /dev/null 2>&1; then
        echo ""
        echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo " >>> oh no!!! ERROR: $f differs between baseline and reproduce"
        echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        echo ""
        diff "../aux_data/$f" "outputs_reference_tracks/$f"
        exit 1
    fi
done


echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo " >>> all good, yay (: reference tracks have been reproduced successfully"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo ""
