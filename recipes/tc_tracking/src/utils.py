from math import ceil
from typing import Callable, Any
from omegaconf import DictConfig
import numpy as np
import torch

from physicsnemo.distributed import DistributedManager
from earth2studio.utils.time import to_time_array


def set_initial_times(cfg: DictConfig) -> list[np.datetime64]:
    """Build list of IC times.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    list[np.datetime64]
        Dictionary containing model, model class and model package.
    """
    # list of ICs
    if "ics" in cfg:
        if "ic_block_start" in cfg:
            raise ValueError(
                "either provide a list of start times or define a block, not both"
            )
        ics = to_time_array(sorted(cfg.ics))

    # block of ICs
    else:
        ics = to_time_array([cfg.ic_block_start, cfg.ic_block_end])
        ics = np.arange(
            ics[0],
            ics[1] + np.timedelta64(cfg.ic_block_step, "h"),
            np.timedelta64(cfg.ic_block_step, "h"),
        )

    return ics


def remove_duplicates(data_list):
    """
    Remove duplicates while preserving numpy dtype distinctions.
    Arrays with same values but different dtypes are considered different.
    """
    def to_hashable(item):
        if isinstance(item, np.ndarray):
            # Include dtype and shape information
            return (tuple(item.tolist()), str(item.dtype), item.shape)
        elif isinstance(item, np.datetime64):
            return str(item)
        return item

    seen = set()
    result = []

    for sublist in data_list:
        hashable_key = tuple(to_hashable(item) for item in sublist)
        if hashable_key not in seen:
            seen.add(hashable_key)
            result.append(sublist)

    return result


def get_set_of_random_seeds(n_ics, ensemble_size, batch_size, seed):
    n_batches = ceil(ensemble_size/batch_size)
    rng = np.random.default_rng(seed=seed)

    seeds = np.array([])
    ii = 0
    while len(np.unique(seeds)) != n_batches*n_ics:
        ii += 1
        if ii > 1000:
            raise RecursionError(f'failed to generate unique set of {n_batches*n_ics} random seeds after 1000 iterations. giving up :(')
        seeds = rng.integers(low=0, high=2**32, size=n_batches*n_ics, dtype=np.uint32)

    return seeds


def run_with_rank_ordered_execution(
    func: Callable, *args: Any, first_rank: int = 0, **kwargs: Any
) -> Any:
    """Executes `func(*args, **kwargs)` safely in a distributed setting:
    - First on the specified `rank`
    - Then, after synchronization, on the other ranks

    Args:
        func (Callable): Function to execute
        args (tuple, optional): Positional arguments for the function. Defaults to ().
        first_rank (int, optional): Rank to run the function first. Defaults to 0.
        kwargs (dict, optional): Keyword arguments for the function. Defaults to None.

    Returns:
        The return value of func(*args, **kwargs)
    """
    if kwargs is None:
        kwargs = {}

    dist = DistributedManager()
    current_rank = dist.rank

    if current_rank == first_rank:
        result = func(*args, **kwargs)
    else:
        result = None

    # Synchronize all processes after the first rank runs the function
    # Skip the barrier if single-process (no distributed process group)
    if dist.distributed:
        torch.distributed.barrier()

    if current_rank != first_rank:
        result = func(*args, **kwargs)

    if dist.distributed:
        torch.distributed.barrier()

    return result


def squeeze_coords(xx, coords, squeeze_dim):
    if isinstance(squeeze_dim, str):
        squeeze_dim = [squeeze_dim]

    for dim in squeeze_dim:
        idx = list(coords.keys()).index(dim)

        assert coords[dim].shape == (1,), f"can only squeeze dims of length 1, coords[{dim}] has shape {coords[dim].shape}"
        assert xx.shape[idx] == 1, f"can only squeeze dims of length 1, xx[{dim}] has shape {xx.shape[idx]}"

        xx = xx.squeeze(idx)
        coords.pop(dim)

    return xx, coords


def great_circle_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    aa = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    cc = 2 * np.arctan2(np.sqrt(aa), np.sqrt(1-aa))

    return 6371000 * cc
