from collections import OrderedDict
import copy
import os
import shutil
from torch import from_numpy
import numpy as np

from earth2studio.io import NetCDF4Backend, ZarrBackend
from earth2studio.utils.coords import map_coords, split_coords


def initialise_output_coords(
    cfg,
    model,
    ics,
    out_vars,
) -> dict:
    """Initialize output coordinates

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    lon_coords: np.ndarray[np.float64]
        a 1d array containing the longitude values of the grid. ordered ascending
    lat_coords: np.ndarray[np.float64]
        a 1d array containing the latitude values of the grid. ordered descending

    Returns
    -------
    output_coords: dict
        output coordinates
    """
    out_coords = OrderedDict(
    {
        k: v for k, v in model.output_coords(model.input_coords()).items() if
        (k != "batch") and (v.shape != 0)
    }
    )

    out_coords["time"] = ics
    out_coords["lead_time"] = np.asarray(
        [out_coords["lead_time"] * i for i in range(cfg.n_steps + 1)]
    ).flatten()
    out_coords["variable"] = np.array(out_vars)

    out_coords.move_to_end("lead_time", last=False)
    out_coords.move_to_end("time", last=False)

    return out_coords


def add_arrays_to_store(store, out_coords, mems, add_arrays, ic=None, array_kwargs={}):
    oco = copy.deepcopy(out_coords)
    oco["ensemble"] = mems
    oco.move_to_end("ensemble", last=False)
    out_vars = oco.pop("variable")

    if ic is not None:
        oco["time"] = np.array([ic])

    if add_arrays:
        store.add_array(coords=oco,
                        array_name=out_vars,
                        **array_kwargs)

    return


def setup_output(cfg, model, ics, add_arrays):

    out_vars = list(dict.fromkeys(cfg.out_vars))
    out_coords = initialise_output_coords(cfg, model, ics, out_vars)
    chunks = {
        "ensemble": 1,
        "time": 1,
        "lead_time": 1,
        "variable": 1,
    }
    array_kwargs = {}

    # TODO: let proc zero create dir and add barrier
    os.makedirs(cfg.store_dir, exist_ok=True)
    file_name = os.path.join(cfg.store_dir, cfg.project)

    if cfg.store_type == "netcdf" or cfg.store_type == "none":
        store = None

    elif cfg.store_type == "zarr":
        # TODO: wrap in ordered_execution function
        if not file_name.endswith(".zarr"):
            file_name += ".zarr"

        if os.path.exists(file_name) and add_arrays:
            shutil.rmtree(file_name)

        store = ZarrBackend(
            file_name=file_name,
            chunks=chunks,
            backend_kwargs={"overwrite": False},
        )

        add_arrays_to_store(store=store,
                            out_coords=out_coords,
                            mems=np.asarray(list(range(cfg.ensemble_size))).flatten(),
                            add_arrays=add_arrays,
                            array_kwargs=array_kwargs)

    else:
        raise ValueError(f"Invalid store type: {cfg.store_type}")

    return store, out_coords


def initialise_netcdf_output(cfg, out_coords, ic, ic_mems):
    mems = np.concatenate([mem for iic, mem, _ in ic_mems if iic == ic]).flatten()
    seeds = np.concatenate([np.array([seed]*len(mem), dtype=int) for iic, mem, seed in ic_mems if iic == ic]).flatten()

    # setup filename
    file_name = os.path.join(cfg.store_dir, cfg.project)
    if file_name.endswith(".nc"):
        file_name = file_name.replace(".nc", "")
    file_name = file_name + f"_{np.datetime_as_string(ic, unit='s')}_mems{mems[0]:04d}-{mems[-1]:04d}.nc"
    file_name = file_name.replace(":", ".")

    # create store
    chunks = {
        "ensemble": 1,
        "time": 1,
        "lead_time": 1,
        "variable": 1,
    }
    store = NetCDF4Backend(
        file_name=file_name,
        backend_kwargs={"mode": "w", "diskless": False, "chunks": chunks},
    )

    # add random seed to store
    store.add_array(coords={"ensemble": mems}, array_name='random_seed')
    store.write(from_numpy(seeds), {"ensemble": mems}, 'random_seed')

    # add arrays to the store
    add_arrays_to_store(store=store,
                        out_coords=out_coords,
                        mems=mems,
                        ic=ic,
                        add_arrays=True)

    return store

def write_to_store(store, xx, coords, out_coords):
    if store is not None:
        xx_sub, coords_sub = map_coords(xx, coords, out_coords)
        store.write(*split_coords(xx_sub, coords_sub, dim="variable"))
