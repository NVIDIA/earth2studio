import os
from collections import OrderedDict

import numpy as np
from data_handling import (
    compute_averages_of_errors_over_lead_time,
    extract_tracks,
    extract_tracks_from_file,
    get_ensemble_averages,
    match_tracks,
)
from plotting_helpers import (
    plot_ensemble_mean_metrics,
    plot_errors_over_lead_time,
    plot_extreme_extremes_histograms,
    plot_ib_era5,
    plot_over_time,
    plot_relative_over_time,
    plot_spaghetti,
)
from tqdm import tqdm


def load_tracks(case, pred_track_dir, tru_track_dir, out_dir):
    tru_track = extract_tracks_from_file(
        os.path.join(tru_track_dir, f"reference_track_{case}.csv")
    )
    tru_track["dist"] = np.zeros(len(tru_track))

    pred_tracks = extract_tracks(in_dir=os.path.join(pred_track_dir))
    n_members = len(pred_tracks)

    pred_tracks = match_tracks(pred_tracks, tru_track, case)

    if out_dir:
        out_dir = os.path.join(out_dir, case)
        os.makedirs(out_dir, exist_ok=True)

    ens_mean = get_ensemble_averages(pred_tracks=pred_tracks, tru_track=tru_track)

    return tru_track, pred_tracks, ens_mean, n_members, out_dir


def analyse_individual_storms(cases, pred_track_dir, tru_track_dir, out_path):
    if isinstance(cases, str):
        cases = [cases]

    for case in tqdm(cases):
        tru_track, pred_tracks, ens_mean, n_members, out_dir = load_tracks(
            case=case,
            pred_track_dir=pred_track_dir,
            tru_track_dir=tru_track_dir,
            out_dir=out_path,
        )

        # spaghetti plot
        plot_spaghetti(
            true_track=tru_track,
            pred_tracks=pred_tracks,
            ensemble_mean=ens_mean["mean"],
            case=case,
            n_members=n_members,
            out_dir=out_dir,
        )

        # plot intensities over time
        plot_over_time(
            pred_tracks=pred_tracks,
            tru_track=tru_track,
            ensemble_mean=ens_mean,
            case=case,
            n_members=n_members,
            out_dir=out_dir,
        )

        # plot relative intensities over time
        plot_relative_over_time(
            pred_tracks=pred_tracks,
            tru_track=tru_track,
            ensemble_mean=ens_mean,
            case=case,
            n_members=n_members,
            out_dir=out_dir,
        )

        # plot comparison of intensities between ibtracs and era5 over time
        plot_ib_era5(
            tru_track=tru_track, case=case, vars=["msl", "wind_speed"], out_dir=out_dir
        )

        # plot histogram of extreme values of intensities over full track
        plot_extreme_extremes_histograms(
            pred_tracks=pred_tracks,
            tru_track=tru_track,
            ensemble_mean=ens_mean,
            case=case,
            out_dir=out_dir,
        )

        # plot error metrics over lead time
        err_dict, _ = compute_averages_of_errors_over_lead_time(
            pred_tracks=pred_tracks,
            tru_track=tru_track,
            vars=["wind_speed", "msl", "dist"],
        )

        plot_errors_over_lead_time(
            err_dict=err_dict,
            case=case,
            ic=pred_tracks[0]["ic"],
            n_members=n_members,
            n_tracks=len(pred_tracks),
            out_dir=out_dir,
        )

    return


def stack_metrics(err_dict):
    var_errs = []
    for var in err_dict.keys():
        metrics = []
        for metric in err_dict[var].keys():
            metrics.append(err_dict[var][metric])

        metrics = np.stack(metrics, axis=0)
        var_errs.append(metrics)

    return np.stack(var_errs, axis=0)


def stack_cases(storm_metrics, max_len):
    for ii in range(len(storm_metrics["case"])):
        # pad storm_metrics['data'][ii].shape with nan to max_len
        storm_metrics["data"][ii] = np.pad(
            storm_metrics["data"][ii],
            pad_width=(
                (0, 0),
                (0, 0),
                (0, max_len - storm_metrics["data"][ii].shape[-1]),
            ),
            mode="constant",
            constant_values=np.nan,
        )

    storm_metrics["data"] = np.stack(storm_metrics["data"], axis=0)

    # check if shapes ok
    should_shape = (
        len(storm_metrics["case"]),
        len(storm_metrics["var"]),
        len(storm_metrics["metric"]),
        max_len,
    )
    assert (
        storm_metrics["data"].shape == should_shape
    ), f'shapes not matching when stacking cases: {storm_metrics["data"].shape=} {should_shape=}'

    return storm_metrics


def extract_weights(storm_metrics, max_len):

    # extract weights
    ens_idx = storm_metrics["metric"].index("n_members")
    weights = storm_metrics["data"][:, 0, ens_idx, :]
    weights = np.nan_to_num(weights, nan=0).astype(int)

    # remove 'n_members' from metrics
    storm_metrics["metric"].remove("n_members")
    storm_metrics["data"] = np.delete(storm_metrics["data"], ens_idx, axis=-2)

    # add weights to storm_metrics OrderedDict
    storm_metrics["weights"] = weights

    # check if shapes ok
    should_shape = (
        len(storm_metrics["case"]),
        len(storm_metrics["var"]),
        len(storm_metrics["metric"]),
        max_len,
    )
    assert (
        storm_metrics["data"].shape == should_shape
    ), f'shapes not matching when stacking cases: {storm_metrics["data"].shape=} {should_shape=}'

    return storm_metrics


def get_individual_storm_metrics(
    cases, pred_track_dir, tru_track_dir, out_path, vars=["wind_speed", "msl", "dist"]
):
    storm_metrics = OrderedDict(
        {"case": [], "var": None, "metric": None, "lead time": None, "data": []}
    )
    max_len, ensemble_averages, extremes = 0, {}, {}
    for case in tqdm(cases, desc="loading storm data"):
        tru_track, pred_tracks, ens_mean, n_members, out_dir = load_tracks(
            case=case,
            pred_track_dir=pred_track_dir,
            tru_track_dir=tru_track_dir,
            out_dir=out_path,
        )
        # record ensemble mean
        ensemble_averages[case] = ens_mean

        # compute averages of error metrics over lead time for storm
        err_dict, _max_len = compute_averages_of_errors_over_lead_time(
            pred_tracks=pred_tracks, tru_track=tru_track, vars=vars
        )

        extremes[case] = {}
        for var in vars:
            extremes[case][var] = {}
            for ext, npfun in zip(["min", "max"], [np.nanmin, np.nanmax]):
                extremes[case][var][ext + "_pred"] = err_dict[var].pop(ext)
                extremes[case][var][ext + "_tru"] = npfun(tru_track[var])

        max_len = max(max_len, _max_len)
        storm_metrics["case"].append(case)
        storm_metrics["data"].append(stack_metrics(err_dict))

    storm_metrics["var"] = list(err_dict.keys())
    storm_metrics["metric"] = list(
        err_dict[list(err_dict.keys())[0]].keys()
    )  # TODO remove n_members
    storm_metrics["lead time"] = np.arange(max_len) * np.timedelta64(6, "h")

    return storm_metrics, max_len, ensemble_averages, extremes


def reduce_over_all_storms(storm_metrics):

    ensemble_metrics = {}
    for var in storm_metrics["var"]:
        ensemble_metrics[var] = {}
        var_idx = storm_metrics["var"].index(var)
        for metric in storm_metrics["metric"]:
            met_idx = storm_metrics["metric"].index(metric)
            ensemble_metrics[var][metric] = np.nanmean(
                storm_metrics["data"][:, var_idx, met_idx, :], axis=0
            )

    ensemble_metrics["n_members"] = np.sum(storm_metrics["weights"], axis=0)

    return ensemble_metrics


def analyse_ensemble_of_storms(cases, pred_track_dir, tru_track_dir, out_path):

    storm_metrics, max_len, ens_means = get_individual_storm_metrics(
        cases, pred_track_dir, tru_track_dir, out_path
    )

    storm_metrics = stack_cases(storm_metrics, max_len)

    storm_metrics = extract_weights(storm_metrics, max_len)

    ensemble_metrics = reduce_over_all_storms(storm_metrics)

    plot_ensemble_mean_metrics(ens_means, out_path)
    # plot average dist in space and intensity over lead time (reduced over all storms)

    # print()
    # print(f'{storm_metrics.keys()=}')
    # print(f'{storm_metrics['case']=}')
    # print(f'{storm_metrics['var']=}')
    # print(f'{storm_metrics['metric']=}')
    # print(f'{storm_metrics['data'].shape=}')
    # print(f'{storm_metrics['weights'].shape=}')

    # print()
    # print(f'{ensemble_metrics.keys()=}')
    # print(f'{ensemble_metrics['msl']['mae'].shape=}')
    # print(f'{ensemble_metrics['n_members'].shape=}')

    # print()
    # print(f'{ens_means['helene_2024_north_atlantic'].keys()=}')
    # print(f'{ens_means['helene_2024_north_atlantic']['mean'].keys()=}')
    # print(f'{ens_means['helene_2024_north_atlantic']['time']=}')
    # print(f'{ens_means['helene_2024_north_atlantic']['n_members'].shape=}')
    # print(f'{ens_means['helene_2024_north_atlantic']['mean']['dist'].shape=}')

    return ensemble_metrics


def analyse_n_plot_tracks():
    cases = [
        "amphan_2020_north_indian",  # 00
        "beryl_2024_north_atlantic",  # 01
        "debbie_2017_southern_pacific",  # 02
        "dorian_2019_north_atlantic",  # 03
        "harvey_2017_north_atlantic",  # 04
        "hato_2017_west_pacific",  # 05
        "helene_2024_north_atlantic",  # 06
        "ian_2022_north_atlantic",  # 07
        "iota_2020_north_atlantic",  # 08
        "irma_2017_north_atlantic",  # 09
        "lan_2017_west_pacific",  # 10
        "lee_2023_north_atlantic",  # 11
        "lorenzo_2019_north_atlantic",  # 12
        "maria_2017_north_atlantic",  # 13
        "mawar_2023_west_pacific",  # 14
        "michael_2018_north_atlantic",  # 15
        "milton_2024_north_atlantic",  # 16
        "ophelia_2017_north_atlantic",  # 17
        "yagi_2024_west_pacific",  # 18
    ]

    # case_selection = list(range(len(cases)))
    case_selection = [6, 13]
    individual_storms = False
    ensemble_of_storms = True

    pred_track_dir = f"/home/mkoch/Documents/projects/fcn3_tcs/track_data/predictions/case_studies_256_mem_random_seed_issue/cyclone_tracks_te"
    tru_track_dir = (
        f"/home/mkoch/Documents/projects/fcn3_tcs/track_data/reference_tracks"
    )
    out_dir = f"/home/mkoch/Documents/projects/fcn3_tcs/plots/case_studies_256_random_seed_issue"

    if individual_storms:
        analyse_individual_storms(
            cases=[cases[ii] for ii in case_selection],
            pred_track_dir=pred_track_dir,
            tru_track_dir=tru_track_dir,
            out_path=out_dir,
        )

    if ensemble_of_storms:
        analyse_ensemble_of_storms(
            cases=[cases[ii] for ii in case_selection],
            pred_track_dir=pred_track_dir,
            tru_track_dir=tru_track_dir,
            out_path=out_dir,
        )

    return


if __name__ == "__main__":
    analyse_n_plot_tracks()
