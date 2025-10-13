import glob
import numpy as np
import pandas as pd


def great_circle_distance(lat1, lon1, lat2, lon2, radius=6371000):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    aa = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    cc = 2 * np.arctan2(np.sqrt(aa), np.sqrt(1-aa))

    return radius * cc


def merge_tracks_by_time(track, tru_track):
    merged_track  = pd.merge(track, tru_track, on='time', how='left', suffixes=('', '_tru'))

    # remove columns which are later than last time of tru_track
    merged_track = merged_track[merged_track['time'] <= tru_track['time'].max()]

    return merged_track


def add_track_distance(track, tru_track):

    merged_track = merge_tracks_by_time(track, tru_track)[['time', 'lat', 'lon', 'lat_tru', 'lon_tru']]

    # compute great circle distance between each point on the track and the true track
    dist = great_circle_distance(merged_track['lat'], merged_track['lon'], merged_track['lat_tru'], merged_track['lon_tru'])

    # add to merged_track
    merged_track['dist'] = dist

    # add dist to track
    track = pd.merge(track, merged_track[['time', 'dist']], on='time', how='left', suffixes=('', ''))

    return track


def match_tracks(pred_tracks, true_track, case):
    # iterate over tracks and see if track exists in simulation
    matched_tracks = []
    min_dist, max_dist = 25371000, -1.

    for _pred_track_dict in pred_tracks:
        _pred_tracks = _pred_track_dict['tracks']

        if len(_pred_tracks) == 0:
            continue

        n_tracks = _pred_tracks['track_id'].iloc[-1] + 1

        for ii in range(n_tracks):
            track = _pred_tracks.loc[_pred_tracks['track_id'] == ii].copy()

            lat_pred = track['lat'].iloc[0]
            lon_pred = track['lon'].iloc[0]

            if not (true_track['time'] == track['time'].iloc[0]).any():
                continue

            lat_true = true_track.loc[true_track['time'] == track['time'].iloc[0]]['lat_ib'].item()
            lon_true = true_track.loc[true_track['time'] == track['time'].iloc[0]]['lon_ib'].item()
            dist = great_circle_distance(lat_pred, lon_pred, lat_true, lon_true)

            if dist <= 300000:
                min_dist, max_dist = min(min_dist, dist), max(max_dist, dist)

                # add dist to track
                track = add_track_distance(track, true_track)

                matched_tracks.append({'ic': _pred_track_dict['ic'],
                                       'member': _pred_track_dict['member'],
                                       'first_match': track['time'].iloc[0],
                                       'initial_dist': dist,
                                       'tracks': track})
                break

    line = case.split("_")[:2]
    line = f'{line[0].upper()} {line[1]}: matched {len(matched_tracks)} out of {len(pred_tracks)} tracks, '
    line += f'with distances ranging from {min_dist/1000:.1f} to {max_dist/1000:.1f} km'
    print(line)

    return matched_tracks


def extract_tracks_from_file(csv_file):
    tracks = pd.read_csv(csv_file, sep=',')
    tracks.columns = tracks.columns.str.strip()

    times = [pd.to_datetime(f'{tracks["year"].iloc[jj]}-{int(tracks.iloc[jj]["month"]):02d}-{int(tracks.iloc[jj]["day"]):02d} {int(tracks.iloc[jj]["hour"]):02d}:00:00') for jj in range(len(tracks))]

    tracks.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)
    # if i j in tracks, drop them
    if 'i' in tracks.columns:
        tracks.drop(columns=['i', 'j'], inplace=True)

    tracks.insert(0, 'time', times)

    return tracks


def extract_tracks(in_dir):
    tracks = []
    files = glob.glob(f'{in_dir}/*.csv')
    files.sort()

    for csv_file in files:
        # read csv, strip whitespace from column names
        _tracks = extract_tracks_from_file(csv_file)

        mem = int(csv_file.split('_mems')[-1].split('.csv')[0])
        ic = pd.to_datetime(csv_file.split('_mems')[0][-19:])

        tracks.append({'ic': ic, 'member': mem, 'tracks': _tracks})

    return tracks


def merge_tracks_by_lead_time(track_dict, tru_track):

    merged_track = merge_tracks_by_time(track_dict['tracks'], tru_track)

    # clip leading and trailing rows in which only predicted or only true track are present
    t_max = max(track_dict['tracks']['time'].max(), tru_track['time'].max())
    t_min = min(track_dict['tracks']['time'].min(), tru_track['time'].min())
    merged_track = merged_track[merged_track['time'] >= t_min]
    merged_track = merged_track[merged_track['time'] <= t_max]

    return merged_track


# function computing MAE
def compute_mae(tru_vars, pred_vars):
    return np.nanmean(np.abs(tru_vars - pred_vars), axis=0)


# function computing mse
def compute_mse(tru_vars, pred_vars):
    return np.nanmean((tru_vars - pred_vars)**2, axis=0)


# function computing RMSE
def compute_rmse(tru_vars, pred_vars):
    return np.sqrt(compute_mse(tru_vars, pred_vars))


# function computing variance
def compute_variance(arr):
    return np.nanvar(arr, axis=0)


def compute_std(arr):
    return np.sqrt(compute_variance(arr))


def remove_trailing_nans(merged_track, var):
    either_nans = np.logical_or(merged_track[var+'_tru'].isna(), merged_track[var].isna())
    cut_off = np.where(~either_nans)[0][-1]

    return merged_track.iloc[:cut_off+1]


def rebase_by_lead_time(pred_tracks, tru_track, vars):
    err_dict = {}
    for var in vars:
        err_dict[var] = {'pred': [], 'tru': []}

    max_len = 0
    ii = 0
    for track in pred_tracks:
        ii += 1
        merged_track = merge_tracks_by_lead_time(track, tru_track)
        if merged_track is None:
            continue

        # remove trailing rows in which both true and predicted track are nan
        merged_track = remove_trailing_nans(merged_track, 'msl')

        max_len = max(max_len, len(merged_track))

        for var in err_dict.keys():
            err_dict[var]['pred'].append(merged_track[var])
            err_dict[var]['tru'].append(merged_track[var+'_tru'])

    return err_dict, max_len


def compute_error_metrics(err_dict, max_len):
    for var in err_dict.keys():
        pred_vars = err_dict[var]['pred']
        tru_vars = err_dict[var]['tru']

        # pad with nans to make all arrays the same length
        counts = np.zeros(max_len, dtype=int)
        for ii in range(len(pred_vars)):
            counts[:len(pred_vars[ii])] += 1

            pred_vars[ii] = np.pad(pred_vars[ii], (0, max_len - len(pred_vars[ii])), mode='constant', constant_values=np.nan)
            tru_vars[ii] = np.pad(tru_vars[ii], (0, max_len - len(tru_vars[ii])), mode='constant', constant_values=np.nan)


        # merge to single array
        pred_vars, tru_vars = np.array(pred_vars), np.array(tru_vars)

        err_dict[var] = {'mae': compute_mae(tru_vars, pred_vars),
                         'mse': compute_mse(tru_vars, pred_vars),
                         'variance': compute_variance(pred_vars),
                         'max': np.nanmax(pred_vars, axis=-1),
                         'min': np.nanmin(pred_vars, axis=-1),
                         'n_members': counts}

    return err_dict


def compute_averages_of_errors_over_lead_time(pred_tracks, tru_track, vars):

    err_dict, max_len = rebase_by_lead_time(pred_tracks, tru_track, vars)

    err_dict = compute_error_metrics(err_dict, max_len)

    return err_dict, max_len


def lat_lon_to_xyz(lat, lon, radius=6371000):
    """
    Converts longitude and latitude coordinates to 3D Cartesian (XYZ) coordinates using numpy.
    This function can handle single values or numpy arrays as input.

    Args:
        lon (float or np.ndarray): Longitude(s) in degrees (range [0, 360)).
        lat (float or np.ndarray): Latitude(s) in degrees (range [-90, 90]).
        radius (float, optional): The radius of the sphere (e.g., Earth's radius).
                                  Defaults to 1.0 for a unit sphere.

    Returns:
        np.ndarray: A numpy array containing the X, Y, and Z coordinates.
                    - If inputs are scalars, returns a (3,) array.
                    - If inputs are arrays of length N, returns an (N, 3) array.
    """
    # Convert degrees to radians using numpy's vectorized function
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)

    # Apply the conversion formulas using numpy's trigonometric functions
    xx = radius * np.cos(lat_rad) * np.cos(lon_rad)
    yy = radius * np.cos(lat_rad) * np.sin(lon_rad)
    zz = radius * np.sin(lat_rad)

    # Stack the coordinates into a single numpy array
    return xx, yy, zz


def xyz_to_lat_lon(xx, yy, zz):
    """
    Converts 3D Cartesian (XYZ) coordinates back to longitude and latitude using numpy.
    This function can handle single values or numpy arrays as input.

    Args:
        x (float or np.ndarray): X coordinate(s).
        y (float or np.ndarray): Y coordinate(s).
        z (float or np.ndarray): Z coordinate(s).

    Returns:
        tuple: A tuple containing longitude and latitude (in degrees).
               - Longitude is in the range [0, 360).
               - Latitude is in the range [-90, 90].
    """
    # Calculate the radius. This is needed to normalize the vector for latitude calculation.
    radius = np.sqrt(xx**2 + yy**2 + zz**2)

    # Latitude can be calculated using arcsin.
    # We add a small epsilon to the denominator to avoid division by zero if the input is (0,0,0).
    lat_rad = np.arcsin(zz / (radius + 1e-9))

    # Longitude is calculated using arctan2(y, x) for quadrant-correct results.
    lon_rad = np.arctan2(yy, xx)

    # Convert radians to degrees and adjust longitude to [0, 360)
    lat = np.degrees(lat_rad)
    lon = (np.degrees(lon_rad) + 360) % 360

    return lat, lon


def cartesian_to_spherical_track(stats, tru_track, frame_of_reference):

    mean_lat, mean_lon = xyz_to_lat_lon(stats['mean']['x'], stats['mean']['y'], stats['mean']['z'])

    # delete cartesian mean/variance
    for var in ['x', 'y', 'z']:
        for metric in ['mean', 'variance']:
            del stats[metric][var]

    # add spherical mean
    stats['mean']['lat'] = mean_lat
    stats['mean']['lon'] = mean_lon

    # compute distance
    tru_cont = pd.merge(frame_of_reference, tru_track, on='time', how='left')

    # compute great circle distance between each point on the track and the true track
    dist = great_circle_distance(tru_cont['lat'], tru_cont['lon'], stats['mean']['lat'], stats['mean']['lon'])

    # add to merged_track
    stats['mean']['dist'] = dist.to_numpy(dtype=float)

    for var in ['msl', 'wind_speed']:
        stats['mean'][var+'_err_of_mean'] = stats['mean'][var] - tru_cont[var]

    return stats


def get_ensemble_averages(pred_tracks, tru_track, vars: list[str]=['msl', 'wind_speed', 'x', 'y', 'z']):

    stats = {'time': None,
             'n_members': None,
             'mean': {var: [] for var in vars},
             'variance': {var: [] for var in vars}}

    # loop over all tracks, match with
    this_is_the_last_time = pred_tracks[0]['ic']

    # iterate over all tracks to get set of all times present
    for track in pred_tracks:
        this_is_the_last_time = max(this_is_the_last_time, track['tracks']['time'].values[-1])

    all_times = np.arange(pred_tracks[0]['ic'], this_is_the_last_time, np.timedelta64(6, 'h'))
    stats["time"] = all_times

    # start data frame single column called time which holds all_times
    frame_of_reference = pd.DataFrame(data=all_times, index=np.arange(len(all_times)), columns=['time'])

    for track in pred_tracks:
        # convert lat/lon to xyz
        xx, yy, zz = lat_lon_to_xyz(track['tracks']['lat'], track['tracks']['lon'])
        track['tracks']['x'] = xx
        track['tracks']['y'] = yy
        track['tracks']['z'] = zz

        contextualised = pd.merge(frame_of_reference, track['tracks'], on='time', how='left')

        for var in vars:
            stats['mean'][var].append(contextualised[var])

    for var in vars:
        # get number of ensemble members contributing to each time step
        if stats["n_members"] is None:
            stats["n_members"] = np.count_nonzero(~np.isnan(stats['mean'][var]), axis=0)
        else:
            assert np.all(stats["n_members"] == np.count_nonzero(~np.isnan(stats['mean'][var]), axis=0)), \
                f'n_members is not the same for all variables but should be....'

        help = np.stack(stats['mean'][var])
        stats['variance'][var] = np.nanvar(help, axis=0)
        stats['mean'][var] = np.nanmean(help, axis=0)

    # convert xyz back to lat/lon
    stats = cartesian_to_spherical_track(stats, tru_track, frame_of_reference)

    return stats
