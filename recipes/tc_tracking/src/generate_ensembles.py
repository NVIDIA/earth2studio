import os
import random
from math import ceil

import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm
from zarr import consolidate_metadata

from earth2studio.data import fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.dx import AsyncTempestExtremes, TempestExtremes
from earth2studio.perturbation import SphericalGaussian
from src.data.file_output import initialise_netcdf_output, setup_output, write_to_store
from src.data.utils import DataSourceManager, load_heights
from src.utils import (
    get_set_of_random_seeds,
    remove_duplicates,
    run_with_rank_ordered_execution,
    set_initial_times,
)


def initialise(cfg):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    DistributedManager.initialize()

    # set seed
    if "random_seed" in cfg:
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)
        torch.cuda.manual_seed(cfg.random_seed)

    return


def load_model(cfg):
    model = "fcn3"
    if "model" in cfg:
        model = cfg.model

    if model == "fcn3":
        from earth2studio.models.px import FCN3

        package = Package(cfg.model_package)
        model = FCN3.load_model(package)
    elif model == "sfno":
        from earth2studio.models.px import SFNO

        package = Package(
            "/lustre/fsw/coreai_climate_earth2/mkoch/hens/hens_checkpoints/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed72"
        )
        model = SFNO.load_model(package)
    elif model == "aifs-ens":
        from earth2studio.models.px import AIFSENS

        model = AIFSENS.load_model(AIFSENS.load_default_package())
    else:
        raise ValueError(f"model {model} not supported")

    return model.to(DistributedManager().device)


def run_inference(model, cfg, store, out_coords, ic_mems):
    dist = DistributedManager()

    # data_source = hydra.utils.instantiate(cfg.data_source)
    data_source_mngr = DataSourceManager(cfg)

    # iterate over initial conditions
    ic_prev = None

    cyclone_tracking = None
    if "cyclone_tracking" in cfg:
        oco = model.output_coords(model.input_coords())

        heights, height_coords = (
            load_heights(cfg.orography_path)
            if "orography_path" in cfg
            else (None, None)
        )

        tracker = (
            AsyncTempestExtremes
            if cfg.cyclone_tracking.asynchronous
            else TempestExtremes
        )
        cyclone_tracking = tracker(
            detect_cmd=cfg.cyclone_tracking.detect_cmd,
            stitch_cmd=cfg.cyclone_tracking.stitch_cmd,
            input_vars=cfg.cyclone_tracking.vars,
            batch_size=cfg.batch_size,
            n_steps=cfg.n_steps,
            time_step=oco["lead_time"][0],
            lats=oco["lat"],
            lons=oco["lon"],
            static_vars=heights,
            static_coords=height_coords,
            store_dir=cfg.store_dir,
            keep_raw_data=cfg.cyclone_tracking.keep_raw_data,
            print_te_output=cfg.cyclone_tracking.print_te_output,
            use_ram=cfg.cyclone_tracking.use_ram,
            timeout=cfg.cyclone_tracking.task_timeout_seconds,
        )

    for ic, mems, seed in ic_mems:
        mini_batch_size = len(mems)

        data_source = data_source_mngr.select_data_source(ic)

        # if new IC, fetch data, create iterator
        if ic != ic_prev:
            print(model.variables)
            # exit()
            if cfg.store_type == "netcdf":
                store = initialise_netcdf_output(cfg, out_coords, ic, ic_mems)
            x0, coords0 = fetch_data(
                data_source,
                time=[np.datetime64(ic)],
                lead_time=model.input_coords()["lead_time"],
                variable=model.variables,
                device=dist.device,
            )
            ic_prev = ic
            exit()

        coords = {"ensemble": np.array(mems)} | coords0.copy()
        xx = x0.unsqueeze(0).repeat(mini_batch_size, *([1] * x0.ndim))

        # set random state or apply perturbation
        if (not "model" in cfg) or (cfg.model == "fcn3"):
            model.set_rng(seed=seed)
        else:
            sg = SphericalGaussian(noise_amplitude=0.0005)
            xx, coords = sg(xx, coords)

        iterator = model.create_iterator(xx, coords)

        # roll out the model and record data as desired
        for _, (xx, coords) in tqdm(
            zip(range(cfg.n_steps + 1), iterator), total=cfg.n_steps + 1
        ):
            write_to_store(store, xx, coords, out_coords)
            if cyclone_tracking:
                cyclone_tracking.record_state(xx, coords)

        if cyclone_tracking:
            cyclone_tracking(
                out_file_names=[
                    f"tracks_{np.datetime_as_string(ic, unit='s')}_mem_{mem:04d}_seed_{seed}_bs_{cfg.batch_size}.csv"
                    for mem in mems
                ]
            )

    # Consolidate metadata in zarr files
    if dist.rank == 0 and cfg.store_type == "zarr":
        # TODO add barrier such that rank 0 finsihes last
        consolidate_metadata(store.store)

    return store


def distribute_runs(ic_mems):
    dist = DistributedManager()

    # get the number of initial conditions
    ic_mems_per_rank = len(ic_mems) // dist.world_size
    if len(ic_mems) % dist.world_size != 0:
        ic_mems_per_rank += 1

    # get the initial conditions for this rank
    ic_mems = ic_mems[dist.rank * ic_mems_per_rank : (dist.rank + 1) * ic_mems_per_rank]

    if len(ic_mems) == 0:
        print(f"nothing to do for rank {dist.rank}, exiting")
        ic_mems = None

    return ic_mems


def configure_runs(cfg):
    ic_mems = []

    ics = set_initial_times(cfg)

    seeds = get_set_of_random_seeds(
        n_ics=len(ics),
        ensemble_size=cfg.ensemble_size,
        batch_size=cfg.batch_size,
        seed=cfg.random_seed if "random_seed" in cfg else None,
    )

    ii = 0
    for ic in ics:
        for mem in range(0, cfg.ensemble_size, cfg.batch_size):
            mems = np.arange(mem, min(mem + cfg.batch_size, cfg.ensemble_size))
            ic_mems.append((ic, mems, int(seeds[ii])))
            ii += 1

    if not DistributedManager().distributed:
        return ic_mems, ics

    ic_mems = distribute_runs(ic_mems)

    return ic_mems, ics


def set_reproduction_configs(cfg):
    ic_mems = OmegaConf.to_container(cfg.reproduce_members)

    ics = []
    for ii in range(len(ic_mems)):
        # time to numpy object
        ic_mems[ii][0] = np.datetime64(ic_mems[ii][0])
        ics.append(ic_mems[ii][0])

        # get full batches that include members which shall be reproduced
        batch_id = ic_mems[ii][1] // cfg.batch_size
        ic_mems[ii][1] = np.arange(
            batch_id * cfg.batch_size,
            min((batch_id + 1) * cfg.batch_size, cfg.ensemble_size),
        )

    # remove duplicates
    ic_mems = remove_duplicates(ic_mems)
    ics = list(set(ics))

    if not DistributedManager().distributed:
        return ic_mems, ics

    ic_mems = distribute_runs(ic_mems)

    return ic_mems, ics


def generate_ensemble(cfg):

    initialise(cfg)

    ic_mems, ics = configure_runs(cfg)

    model = load_model(cfg)

    # store, out_coords = setup_output(cfg, model, add_arrays=DistributedManager().rank == 0)
    store, out_coords = (
        run_with_rank_ordered_execution(  # TODO: wrap only zarr store in that loop
            setup_output,
            cfg=cfg,
            model=model,
            ics=ics,
            add_arrays=DistributedManager().rank == 0,
        )
    )

    if ic_mems is None:
        DistributedManager().cleanup()
        exit()

    store = run_inference(model, cfg, store, out_coords, ic_mems)

    return


def reproduce_members(cfg):
    if cfg.store_type == "zarr":
        raise ValueError("Zarr output not suported for reproducing ensemble members")

    initialise(cfg)

    ic_mems, ics = set_reproduction_configs(cfg)

    model = load_model(cfg)

    # store, out_coords = setup_output(cfg, model, add_arrays=DistributedManager().rank == 0)
    store, out_coords = (
        run_with_rank_ordered_execution(  # TODO: wrap only zarr store in that loop
            setup_output,
            cfg=cfg,
            model=model,
            ics=ics,
            add_arrays=DistributedManager().rank == 0,
        )
    )

    if ic_mems is None:
        DistributedManager().cleanup()
        exit()

    store = run_inference(model, cfg, store, out_coords, ic_mems)

    return
