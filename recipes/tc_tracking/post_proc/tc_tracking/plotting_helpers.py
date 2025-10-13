import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from data_handling import merge_tracks_by_time

def add_some_gap(lat_min, lat_max, lon_min, lon_max):
    gap_fac = .1
    lat_gap = (lat_max - lat_min) * gap_fac
    lon_gap = (lon_max - lon_min) * gap_fac

    lat_min, lat_max = lat_min - lat_gap, lat_max + lat_gap
    lon_min, lon_max = lon_min - lon_gap, lon_max + lon_gap

    if lat_gap/lon_gap > 2:
        d_lon = .5*(lat_max - lat_min)
        med_lon = .5*(lon_min + lon_max)
        lon_min, lon_max = med_lon - d_lon/2, med_lon + d_lon/2

    elif lon_gap/lat_gap > 2:
        d_lat = .5*(lon_max - lon_min)
        med_lat = .5*(lat_min + lat_max)
        lat_min, lat_max = med_lat - d_lat/2, med_lat + d_lat/2

    return lat_min, lat_max, lon_min, lon_max


def get_central_coords(track):
    lat_cen = track['lat'].median()
    lon_cen = track['lon'].median()

    return lat_cen, lon_cen


def plot_spaghetti(true_track,
                   pred_tracks,
                   ensemble_mean,
                   case,
                   n_members,
                   out_dir: str | None = None,
                   alpha: float=.2,
                   line_width: float=2,
                   ic = None):
    plt.close('all')

    lat_cen, lon_cen = get_central_coords(true_track)

    # Create figure and axis
    fig = plt.figure(figsize=(22,10))
    sup_title = f'{(case.split("_")[0]).upper()} '
    sup_title += f'\n initialised on {pred_tracks[0]["ic"]}'
    sup_title += f'\n {len(pred_tracks)} tracks in {n_members} ensemble members'
    fig.suptitle(sup_title, fontsize=16)

    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=lon_cen, central_latitude=lat_cen)
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.add_feature(cfeature.COASTLINE,lw=.5)
    ax.add_feature(cfeature.RIVERS,lw=.5)
    if case != 'debbie_2017_southern_pacific': # cartopy issues with small islands
        ax.add_feature(cfeature.OCEAN, facecolor='#b0c4de')
        ax.add_feature(cfeature.LAND, facecolor='#C4B9A3')

    lat_min, lat_max, lon_min, lon_max = 90, -90, 360, -.1

    # Plot the line in white
    for _track in pred_tracks:
        track = _track['tracks']
        if ic is not None and _track['ic'] not in ic:
            continue

        lat_min, lat_max = min(lat_min, track['lat'].min()), max(lat_max, track['lat'].max())
        lon_min, lon_max = min(lon_min, track['lon'].min()), max(lon_max, track['lon'].max())

        ax.plot(track['lon'], track['lat'], transform=ccrs.PlateCarree(),
                color='black', linewidth=line_width, alpha=alpha)

    ax.plot(true_track['lon'], true_track['lat'], transform=ccrs.PlateCarree(),
            color='red', linewidth=line_width, alpha=1.)

    ax.plot(ensemble_mean['lon'], ensemble_mean['lat'], transform=ccrs.PlateCarree(),
            color='lime', linewidth=line_width, alpha=1.)

    lat_min, lat_max = min(lat_min, true_track['lat'].min()), max(lat_max, true_track['lat'].max())
    lon_min, lon_max = min(lon_min, true_track['lon'].min()), max(lon_max, true_track['lon'].max())

    lat_min, lat_max, lon_min, lon_max = add_some_gap(lat_min, lat_max, lon_min, lon_max)

    # Remove extra space around the plot
    plt.tight_layout()

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)


    if out_dir:
        fig.savefig(os.path.join(out_dir, f'{case}_tracks.png'))

    return


def normalised_intensities(track, tru_track, var):

    merged_track = merge_tracks_by_time(track, tru_track)

    if var == 'msl':
        merged_track[var] = (merged_track[var] - merged_track[var+'_tru']) / (101325 - merged_track[var+'_tru'])
    else:
        merged_track[var] = (merged_track[var] - merged_track[var+'_tru']) / merged_track[var+'_tru']

    return merged_track


def plot_relative_over_time(pred_tracks, tru_track, ensemble_mean, case, n_members, ics=None, out_dir: str | None = None):

    fig, _ax = plt.subplots(2,1, figsize=(11, 11), sharex=True)
    sup_title = f'{(case.split("_")[0]).upper()} '
    sup_title += f'\n initialised on {pred_tracks[0]["ic"]}'
    sup_title += f'\n {len(pred_tracks)} tracks in {n_members} ensemble members'
    fig.suptitle(sup_title, fontsize=16)

    vars = ['msl', 'wind_speed']
    labels = ['(msl - msl_ref)/(101325Pa - msl_ref)', 'max_wind/max_wind_ref - 1']

    ic, end = pred_tracks[0]['ic'], tru_track['time'].max()
    rel_steps = int(((end - ic) / np.timedelta64(6, 'h') + 1)*.75)


    for ii in range(_ax.shape[0]):
        vmin, vmax = 1000, -1000
        for _track in pred_tracks:

            track = _track['tracks']
            if ics is not None and _track['ic'] not in ics:
                continue

            track = normalised_intensities(track, tru_track, vars[ii])

            vmin, vmax = min(vmin, track[vars[ii]][:rel_steps].min()), max(vmax, track[vars[ii]][:rel_steps].max())

            ax = _ax[ii]
            ax.plot(track['time'], track[vars[ii]], color='black', alpha=.1)

        _ax[ii].set_ylabel(labels[ii])
        _ax[ii].grid(True)
        _ax[ii].set_ylim(vmin, vmax)

        ax.plot(tru_track['time'], [0 for _ in range(len(tru_track))], color='orangered', linewidth=2.5, label='era5 comparison')

        mean = pd.DataFrame({'time': ensemble_mean['time'],
                             vars[ii]: ensemble_mean['mean'][vars[ii]]})
        _track = normalised_intensities(mean, tru_track, vars[ii])
        ax.plot(_track['time'], _track[vars[ii]], color='lime', linewidth=2.5, label='ensemble mean', linestyle='--')


        ax.legend()

    # Set x-axis label to indicate hours
    _ax[-1].set_xlabel('time [UTC]')

    plt.xlim(track['time'].min()-np.timedelta64(6, 'h'), tru_track['time'].max()+np.timedelta64(6, 'h'))

    if out_dir:
        plt.savefig(os.path.join(out_dir, f'{case}_rel_intensities.png'))

    return


def plot_over_time(pred_tracks,
                   tru_track,
                   ensemble_mean,
                   case,
                   n_members,
                   vars=['msl', 'wind_speed', 'dist'],
                   labels=['msl [hPa]', 'maximum instantaneous wind speed [m/s]', 'distance [km]'],
                   ics=None,
                   out_dir: str | None = None):

    fig, _ax = plt.subplots(len(vars),1, figsize=(11, 15), sharex=True)
    sup_title = f'{(case.split("_")[0]).upper()} '
    sup_title += f'\n initialised on {pred_tracks[0]["ic"]}'
    sup_title += f'\n {len(pred_tracks)} tracks in {n_members} ensemble members'
    fig.suptitle(sup_title, fontsize=16)

    t_min, t_max = np.datetime64('2120-05-16 12:00:00'), np.datetime64('1820-05-16 12:00:00')

    for ii in range(_ax.shape[0]):
        for _track in pred_tracks:
            track = _track['tracks']
            if ics is not None and _track['ic'] not in ics:
                continue

            if vars[ii] == 'dist':
                yy = track[vars[ii]]/1000
            elif vars[ii] == 'msl':
                yy = track[vars[ii]]/100
            else:
                yy = track[vars[ii]]

            _ax[ii].plot(track['time'], yy, color='black', alpha=.1)

            t_min, t_max = min(t_min, track['time'].min()), max(t_max, track['time'].max())

        _ax[ii].set_xlim(t_min-np.timedelta64(6, 'h'), t_max+np.timedelta64(6, 'h'))
        _ax[ii].set_ylabel(labels[ii])
        _ax[ii].grid(True)

        # if vars[ii] != 'dist':
        yy = tru_track[vars[ii]]/100 if vars[ii] == 'msl' else tru_track[vars[ii]]
        _ax[ii].plot(tru_track['time'], yy, color='orangered', linewidth=2.5, label='era5 comparison')


        if vars[ii] == 'dist':
            yy = ensemble_mean['mean'][vars[ii]]/1000
        elif vars[ii] == 'msl':
            yy = ensemble_mean['mean'][vars[ii]]/100
        else:
            yy = ensemble_mean['mean'][vars[ii]]
        _ax[ii].plot(ensemble_mean['time'], yy, color='lime', linewidth=2.5, label='ensemble mean', linestyle='--')
        _ax[ii].legend()


    # Set x-axis label to indicate hours
    _ax[-1].set_xlabel('time [UTC]')

    if out_dir:
        plt.savefig(os.path.join(out_dir, f'{case}_abs_intensities.png'))

    return


def plot_ib_era5(tru_track, case, vars=['msl', 'wind_speed'], out_dir: str | None = None):
    plt.close('all')

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    sup_title = f'{(case.split("_")[0]).upper()} '
    fig.suptitle(sup_title, fontsize=16)

    ax2 = ax1.twinx()

    if 'msl' in vars:
        p_norm = 101325
        ax1.plot(tru_track['time'], (p_norm-tru_track['msl'])/(p_norm-tru_track['msl_ib']), 'black')
        ax1.set_ylabel('(1013hPa-msl_era5)/(1013hPa-msl_ib)', color='black')

    if 'wind_speed' in vars:
        ax2.plot(tru_track['time'], tru_track['wind_speed']/tru_track['wind_speed_ib'], 'orangered')
        ax2.set_ylabel('wind_speed_era5/wind_speed_ib', color='orangered')

    fig.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f'{case}_ib_era5_wind_speed.png'))

    return


def root_metrics(err_dict):
    for var in err_dict.keys():
        mse = err_dict[var].pop('mse')
        err_dict[var]['rmse'] = np.sqrt(mse)
        variance = err_dict[var].pop('variance')
        err_dict[var]['standard_deviation'] = np.sqrt(variance)
        n_mems_lt = err_dict[var].pop('n_members')

    return err_dict, n_mems_lt


def plot_errors_over_lead_time(err_dict,
                               case,
                               ic,
                               n_members,
                               n_tracks,
                               norm_dict = {'msl': 100, 'wind_speed': 1, 'dist': 1000},
                               unit_dict = {'msl': 'hPa', 'wind_speed': 'm/s', 'dist': 'km'},
                               out_dir: str | None = None):

    if 'mse' in err_dict[list(err_dict.keys())[0]].keys():
        err_dict, n_mems_lt = root_metrics(err_dict)

    vars = list(err_dict.keys())
    metrics = list(err_dict[vars[0]].keys())

    # remove min, max as not over lead time but per member
    for extreme in ['min', 'max']:
        if extreme in metrics:
            metrics.remove(extreme)

    print(metrics)

    lead_time = np.arange(err_dict[vars[0]][metrics[0]].shape[0])*np.timedelta64(6, 'h')

    fig, ax = plt.subplots(len(vars), len(metrics), figsize=((len(metrics)+1)*2, (len(vars)+1)*2), sharex=True)

    for ivar, var in enumerate(err_dict.keys()):
        for imet, metric in enumerate(metrics):

            ax[ivar, imet].plot(lead_time, err_dict[var][metric]/norm_dict[var])

            if ivar == 0:
                ax[ivar, imet].set_title(metric, fontsize=12, weight='bold')

            if imet == 0:
                ax[ivar, imet].set_ylabel(f'{var} [{unit_dict[var]}]', fontsize=12, weight='bold')

            if ivar == len(vars) - 1:
                ax[ivar, imet].set_xlabel('lead time [h]', fontsize=12)


    sup_title = f'{(case.split("_")[0]).upper()} '
    sup_title += f'\n initialised on {ic}'
    sup_title += f'\n {n_tracks} tracks in {n_members} ensemble members'
    fig.suptitle(sup_title, fontsize=16)

    fig.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, f'{case}_error_metrics_over_lead_time.png'))

    return


def extract_reference_extremes(tru_track, pred_tracks, ens_mean, vars):

    extreme_dict = {}
    for var in vars:
        extreme_dict[var] = {'pred': np.zeros(len(pred_tracks))}
        for ii, track in enumerate(pred_tracks):
            if var in ['wind_speed']:
                extreme_dict[var]['pred'][ii] = np.nanmax(track['tracks'][var])
                extreme_dict[var]['tru'] = np.nanmax(tru_track[var])
                extreme_dict[var]['ens_mean'] = np.nanmax(ens_mean['mean'][var])
            elif var in ['msl']:
                extreme_dict[var]['pred'][ii] = np.nanmin(track['tracks'][var])
                extreme_dict[var]['tru'] = np.nanmin(tru_track[var])
                extreme_dict[var]['ens_mean'] = np.nanmin(ens_mean['mean'][var])

    return extreme_dict


def add_stats_box(ax, pred_var, tru_var, var, reduction, unit):

    # add text box below plot with number of members exceeding the threshold
    n_exceed_spd = len(pred_var[pred_var > tru_var])
    n_total = len(pred_var)

    # Create table-like format with aligned colons for wind speed
    comp = 'exceeding' if var in ['wind_speed'] else 'below'
    stats = [
        ('era5 reference:', f"{tru_var:.1f} {unit}"),
        (f'members {comp} ref:', f'{n_exceed_spd} of {n_total} ({(n_exceed_spd/n_total)*100:.1f}%)'),
        (f'max {reduction} {var}:', f'{pred_var.max():.1f} {unit}'),
        (f'min {reduction} {var}:', f'{pred_var.min():.1f} {unit}'),
        (f'avg {reduction} {var}:', f'{pred_var.mean():.1f} {unit}'),
        (f'std {reduction} {var}:', f'{pred_var.std():.1f} {unit}')
    ]

    # Format as table with aligned colons
    max_label_width = max(len(label) for label, _ in stats)
    text = '\n'.join([f'{label:<{max_label_width}}  {value}' for label, value in stats])

    ax.text(.01, -0.25, text, transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            fontfamily='monospace')

    return


def plot_extreme_extremes_histograms(pred_tracks, tru_track, ensemble_mean, case, vars=['wind_speed', 'msl'], out_dir: str | None = None, nbins: int=12):

    extreme_dict = extract_reference_extremes(tru_track, pred_tracks, ensemble_mean, vars)

    # plot histogram of spd and msl
    fig, ax = plt.subplots(1, len(vars), figsize=(3*(len(vars)+1), 6), sharey=True)
    fig.suptitle(f'{(case.split("_")[0]).upper()} initialised on {pred_tracks[0]["ic"]}', fontsize=16)
    ax[0].set_ylabel('count')

    for ii, var in enumerate(vars):

        reduction = 'max' if var in ['wind_speed'] else 'min'
        unit = 'm/s' if var in ['wind_speed'] else 'hPa'
        scale = 100 if var in ['msl'] else 1

        pred_var = extreme_dict[var]['pred']/scale
        tru_var = extreme_dict[var]['tru']/scale
        mean_var = extreme_dict[var]['ens_mean']/scale

        # plot histo and vertical line at tru_var
        ax[ii].hist(pred_var, bins=nbins)
        ax[ii].axvline(tru_var, color='orangered', linestyle='--', label='era5 reference')
        ax[ii].axvline(mean_var, color='lime', linestyle='--', label='ensemble mean')

        ax[ii].set_title(f'{reduction} {var} (x, t)')
        ax[ii].set_xlabel(f'{var} [{unit}]')
        ax[ii].legend()

        add_stats_box(ax[ii], pred_var, tru_var, var, reduction, unit)

    fig.tight_layout()
    if out_dir:
        fig.savefig(os.path.join(out_dir, f'{case}_histograms.png'))

    plt.show()



def plot_ensemble_mean_metrics(ens_means, out_path):

    errs = []
    for _, means in ens_means.items():
        errs.append(means)

    errs = np.stack(errs)
    print(errs.shape)
    exit()

    return
