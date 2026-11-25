#!/bin/bash
# WeatherNext 2 Cyclones (Hurricane Melissa, IFS analysis): generate two members
# with both trackers, the one built into the model and TempestExtremes,
# reproduce the members and check that the tracks of both trackers stay close.
# Needs an 80 GB GPU.
#
#   ./test_wn2.sh
set -e

ic="2025-10-24 00:00:00"
baseline="outputs_baseline_melissa_wn2"
reproduce="outputs_reproduce_melissa_wn2"
trackers=(cyclone_tracks_wn2 cyclone_tracks_te)
rm -rf "$baseline" "$reproduce"

banner() {
    echo ""
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo " >>> $1"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo ""
}

banner "running baseline with wn2. might take a couple of minutes..."
python ../main.py --config-path="$(pwd)/cfg" --config-name=baseline_melissa_wn2.yaml

# both trackers write one file per member, under the same name
for tracker in "${trackers[@]}"; do
    track_count=$(ls "$baseline/$tracker"/tracks_*.csv 2>/dev/null | wc -l)
    if [ "$track_count" -ne 2 ]; then
        banner "ERROR: expected 2 track files in $baseline/$tracker, found $track_count"
        exit 1
    fi
done
if ! diff <(ls "$baseline/cyclone_tracks_wn2") <(ls "$baseline/cyclone_tracks_te"); then
    banner "ERROR: the trackers named their track files differently"
    exit 1
fi

# the built-in tracker cannot be used with another model
if python ../main.py --config-path="$(pwd)/cfg" --config-name=baseline_melissa_wn2.yaml \
    model=fcn3 > /dev/null 2>&1; then
    banner "ERROR: wn2_tracking was accepted for model fcn3"
    exit 1
fi

# seeds of the members from the track file names
seed_mem0=$(ls "$baseline/cyclone_tracks_wn2" | grep 'mem_0000' | grep -oP 'seed_\K\d+')
seed_mem1=$(ls "$baseline/cyclone_tracks_wn2" | grep 'mem_0001' | grep -oP 'seed_\K\d+')

banner "reproducing members 0 and 1..."
python ../main.py --config-path="$(pwd)/cfg" --config-name=reproduce_melissa_wn2.yaml \
    "reproduce_members=[['${ic}',0,${seed_mem0}],['${ic}',1,${seed_mem1}]]"

# WN2 is not bitwise reproducible on GPU: the reproduced tracks have to stay
# close to the original ones, see compare_tracks.py for the tolerances
for tracker in "${trackers[@]}"; do
    if ! python compare_tracks.py "$baseline/$tracker" "$reproduce/$tracker"; then
        banner "oh no!!! ERROR: $tracker tracks of the reproduced members differ"
        exit 1
    fi
done

banner "all good, yay (: the tracks of both trackers were reproduced successfully"
