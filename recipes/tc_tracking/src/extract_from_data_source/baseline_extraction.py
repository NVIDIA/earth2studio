import copy
import os

import numpy as np
import pandas as pd
import tropycal.tracks as tropytracks
from physicsnemo.distributed import DistributedManager

from earth2studio.data import fetch_data
from earth2studio.io import KVBackend
from earth2studio.utils.coords import split_coords
from src.tempest_extremes import AsyncTempestExtremes, TempestExtremes
from src.data.utils import DataSourceManager, load_heights
from src.utils import great_circle_distance


def extract_from_historic_data(cfg,
                               ic,
                               n_steps,
                               time_step: np.timedelta64 = np.timedelta64(6, 'h'),
                               vars: list[str] = ['msl', 'z300', 'z500', 'u10m', 'v10m'],
                               data_source_mngr: DataSourceManager = None):

    times = np.arange(np.timedelta64(0, 'h'), n_steps*time_step, time_step)
    data_source = data_source_mngr.select_data_source(ic+times)

    # fetch data on which TC tracking is executed and add ensemble dim, so that it works with the diagnostic
    xx, coords = fetch_data(data_source,
                            time=[ic],
                            lead_time=times,
                            variable=vars,
                            device='cpu')
    xx = xx.unsqueeze(0)
    coords["ensemble"] = np.array([0])
    coords.move_to_end("ensemble", last=False)

    heights, height_coords = load_heights(cfg.orography_path)

    # set up TC tracking, add downloaded data and execute tracking algorithm
    cyclone_tracking = TempestExtremes(detect_cmd=cfg.cyclone_tracking.detect_cmd,
                                       stitch_cmd=cfg.cyclone_tracking.stitch_cmd,
                                       input_vars=cfg.cyclone_tracking.vars,
                                       batch_size=1,
                                       n_steps=n_steps-1, # assumes prediction steps
                                       time_step=time_step,
                                       lats=coords['lat'],
                                       lons=coords['lon'],
                                       static_vars=heights,
                                       static_coords=height_coords,
                                       store_dir=cfg.store_dir,
                                       keep_raw_data=cfg.cyclone_tracking.keep_raw_data,
                                       print_te_output=cfg.cyclone_tracking.print_te_output,
                                       use_ram=cfg.cyclone_tracking.use_ram)
    cyclone_tracking.record_state(xx, coords)
    cyclone_tracking()

    # load track_file into dataFrame
    hist_tracks = pd.read_csv(cyclone_tracking.track_files[0])
    hist_tracks.columns = hist_tracks.columns.str.strip()
    os.remove(cyclone_tracking.track_files[0])

    return hist_tracks


def extract_from_ibtracs(cfg, ibtracs, case, time_step):
    print(f'extracting storm {case} ({cfg.cases[case].year}, {cfg.cases[case].basin}) from ibtracs')
    storm = ibtracs.get_storm((case, cfg.cases[case].year))
    storm.time = storm.time.astype(np.datetime64)

    # convert to pd.DataFrame, add day year month hour for easier merging later on
    ib_storm = pd.DataFrame({
        'time': storm.time,
        'year': storm.time.astype('datetime64[Y]').astype(int) + 1970,
        'month': storm.time.astype('datetime64[M]').astype(int) % 12 + 1,
        'day': np.array([int(pd.Timestamp(tt).day) for tt in storm.time]),
        'hour': storm.time.astype('datetime64[h]').astype(int) % 24,
        'lat': storm.lat,
        'lon': storm.lon,
        'wind_speed': storm.vmax,
        'mslp': storm.mslp,
        'type': storm.type,
    })

    # get time window to extract from historic data
    ic = storm.time[0]

    # Ensure ic is at 00h, 06h, 12h, or 18h. if not, set to previous 6-hourly time step
    ic_datetime = pd.to_datetime(ic)
    hour = ic_datetime.hour
    if hour % 6 != 0:
        # Round down to the previous 6-hourly time step
        hours_to_subtract = hour % 6
        ic = ic - np.timedelta64(hours_to_subtract, 'h')
        ic = ic.astype('datetime64[h]')

    # calculate number of steps to extract from historic data
    n_steps = (storm.time[-1] - ic) // time_step
    if (storm.time[-1] - ic) % time_step != 0:
        n_steps += 1

    # select ib_storm times that are in query_times
    ib_storm = ib_storm[ib_storm['time'].isin(np.arange(ic, ic + n_steps*time_step, time_step))]

    return ib_storm, ic, n_steps


def match_tracks(ib_storm, hist_tracks, case):
    # add time as np.datetime64
    times = np.array([pd.to_datetime(f'{hist_tracks["year"].iloc[jj]}-{int(hist_tracks.iloc[jj]["month"]):02d}-{int(hist_tracks.iloc[jj]["day"]):02d} {int(hist_tracks.iloc[jj]["hour"]):02d}:00:00') for jj in range(len(hist_tracks))])#.astype('datetime64[h]')
    # hist_tracks.drop(columns=['year', 'month', 'day', 'hour', 'i', 'j', 'track_id'], inplace=True)
    hist_tracks.insert(0, 'time', times)

    n_tracks = hist_tracks['track_id'].nunique()

    # go over tracks and see if they match ibtracks
    matched_track = None
    for time in ib_storm['time']:
        lat_ib = ib_storm.loc[ib_storm['time'] == time]['lat'].item()
        lon_ib = ib_storm.loc[ib_storm['time'] == time]['lon'].item()

        for track_id in range(n_tracks):
            track = hist_tracks[hist_tracks['track_id'] == track_id]
            # check if time is in track
            # if time in track['time']:
            if (track['time'] == time).any():
                lat_track = track.loc[track['time'] == time]['lat'].item()
                lon_track = track.loc[track['time'] == time]['lon'].item()
                dist = great_circle_distance(lat_ib, lon_ib, lat_track, lon_track)
                if dist < 300000:
                    matched_track = track_id
                    print(f'matched track {matched_track} for strom {case} with distance {dist/1000:.2f}km')
                    break

        if matched_track is not None:
            break

    return hist_tracks[hist_tracks['track_id'] == matched_track].reset_index(drop=True)


def add_ibtracs_data(matched_track, ib_storm):
    # add ib_storm data to matched_track, only at times in matched_track
    matched_track = pd.merge(matched_track, ib_storm, on='time', how='right', suffixes=('', '_ib')).reset_index(drop=True)
    matched_track.rename(columns={'mslp': 'msl_ib'}, inplace=True)

    # Preserve integer data types for datetime components
    # Convert back to integers, handling any NaN values appropriately
    datetime_cols = ['year', 'month', 'day', 'hour']
    for col in datetime_cols:
        if col in matched_track.columns:
            # Use 'Int64' (nullable integer) to handle NaN values if they exist
            matched_track[col] = matched_track[col].astype('Int64')

    # set track_id to 0, keep as dummy value to have data format consistent with simulation output
    matched_track['track_id'] = 0

    # if values at lon_ib are negative, set to 360 + lon_ib
    matched_track.loc[matched_track['lon_ib'] < 0, 'lon_ib'] = 360 + matched_track.loc[matched_track['lon_ib'] < 0, 'lon_ib']

    return matched_track


def write_track_to_csv(matched_track, case, store_dir, basin):
    # remove columns i, j, time
    matched_track.drop(columns=['i', 'j', 'time'], inplace=True)

    # re-order columns
    matched_track = matched_track[['track_id', 'year_ib', 'month_ib', 'day_ib', 'hour_ib', 'lon', 'lat', 'msl', 'wind_speed', 'lon_ib', 'lat_ib', 'msl_ib', 'wind_speed_ib', 'type']]

    # rename columns
    matched_track.rename(columns={'year_ib': 'year', 'month_ib': 'month', 'day_ib': 'day', 'hour_ib': 'hour'}, inplace=True)

    # multiply msl_ib by 100 to get Pa
    matched_track['msl_ib'] = matched_track['msl_ib'] * 100

    # write to csv
    csv_name = f'reference_track_{case}_{matched_track['year'].iloc[0]}_{basin}.csv'
    matched_track.to_csv(os.path.join(store_dir, csv_name), index=False)

    return


def extract_baseline(cfg,
                     time_step: np.timedelta64 = np.timedelta64(6, 'h'),
                     vars:list[str] = ['msl', 'z300', 'z500', 'u10m', 'v10m'],):

    DistributedManager.initialize()

    ibtracs = tropytracks.TrackDataset(basin='all',
                                       source='ibtracs',
                                       ibtracs_mode='jtwc_neumann',
                                       ibtracs_url=cfg.ibtracs_source_data)

    data_source_mngr = DataSourceManager(cfg)

    for case in cfg.cases:

        ib_storm, ic, n_steps = extract_from_ibtracs(cfg, ibtracs, case, time_step)

        hist_tracks = extract_from_historic_data(cfg=cfg, ic=ic, n_steps=n_steps, vars=vars, data_source_mngr=data_source_mngr)

        matched_track = match_tracks(ib_storm, hist_tracks, case)

        matched_track = add_ibtracs_data(matched_track, ib_storm)

        write_track_to_csv(matched_track, case, cfg.store_dir, cfg.cases[case].basin)

    return
