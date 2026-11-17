# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import secrets
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, open_dict

from earth2studio.data import DataSource
from earth2studio.io import IOBackend, KVBackend, XarrayBackend
from earth2studio.models.auto import Package
from earth2studio.models.dx import CorrDiff
from earth2studio.utils.time import to_time_array

from .hens_distributed import worker_rank
from .hens_io import BufferedIOBackend


def get_reproducibility_settings(
    cfg: DictConfig, num_packages: int
) -> tuple[str | int, list[int], bool]:
    """Resolve the seed and selected batches using the discovered packages."""
    if "random_seed" not in cfg:
        with open_dict(cfg):
            cfg.random_seed = secrets.randbelow(1_000_000)
    num_batches = -(-cfg.nensemble // cfg.batch_size) * num_packages
    return (
        cfg.random_seed,
        list(cfg.get("batch_ids_reproduce", range(num_batches))),
        cfg.get("torch_use_deterministic_algorithms", False),
    )


def calculate_torch_seed(s: str) -> int:
    """Preserve HENS' SHA256-based batch seed derivation."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**64) - 1


def create_base_seed_string(pkg: str, ic: np.datetime64, base_random_seed: str) -> str:
    """Combine seed, sanitized package path and second-resolution IC time."""
    package = "".join(char for char in pkg if char.isalnum())
    return f"{base_random_seed}_{package}_{ic.astype('datetime64[s]')}"


def ensure_all_torch_seeds_are_unique(
    ensemble_configs: list[tuple], base_random_seed: str
) -> None:
    """Reject repeated batch seeds across this worker's checkpoint/IC pairs."""
    seen: set[int] = set()
    for pkg, ic, _, batches in ensemble_configs:
        base = create_base_seed_string(pkg, ic, base_random_seed)
        for batch in batches:
            seed = calculate_torch_seed(f"{base}_{batch}")
            if seed in seen:
                raise ValueError("Calculated torch seeds for every run must be unique")
            seen.add(seed)


def build_package_list(cfg: DictConfig) -> list[str]:
    """Discover checkpoints in stable order, or use the model's default package."""
    registry = cfg.forecast_model.get("registry", "default")
    if registry == "default":
        return [registry]
    root = Path(registry)
    if (root / "config.json").is_file():
        return [registry]
    packages = sorted(
        str(path.absolute())
        for path in root.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    )
    if not packages:
        raise ValueError(f"Found no valid model packages under {root}")
    return packages[: cfg.forecast_model.get("max_num_checkpoints", 29)]


def set_initial_times(cfg: DictConfig) -> np.ndarray:
    """Normalize explicit IC times or an inclusive, regularly spaced IC block."""
    if "start_times" in cfg:
        if "ic_block_start" in cfg:
            raise ValueError("Provide start_times or an IC block, not both")
        return to_time_array(sorted(cfg.start_times))
    start, end = to_time_array([cfg.ic_block_start, cfg.ic_block_end])
    step = np.timedelta64(cfg.ic_block_step, "h")
    return np.arange(start, end + step, step)


def _output_path(
    cfg: DictConfig, ic: str | np.datetime64, pkg: str, region: str
) -> Path:
    """Keep the existing file naming convention for disk and in-memory outputs."""
    suffix = pkg if pkg == "vanilla" else "_pkg_" + pkg.split("_")[-1]
    return (
        Path(cfg.file_output.get("path", "outputs"))
        / region
        / f"{cfg.project}_{str(ic)[:13]}{suffix}"
    )


def initialise_output(
    cfg: DictConfig, time: np.datetime64, model_dict: dict, output_coords_dict: dict
) -> dict[str, IOBackend]:
    """Create one optionally buffered backend per region; unwind partial setup."""
    if "file_output" not in cfg:
        return {}
    stores: dict[str, IOBackend] = {}
    with ExitStack() as cleanup:
        for region in output_coords_dict:
            path = _output_path(cfg, time, model_dict["package"], region)
            path.parent.mkdir(parents=True, exist_ok=True)

            def create_backend(path: Path = path) -> IOBackend:
                backend = hydra.utils.instantiate(cfg.file_output.format)
                if isinstance(backend, partial):
                    name = cfg.file_output.format._target_.split(".")[-1]
                    extension = next(
                        (
                            ext
                            for prefix, ext in (("NetCDF", ".nc"), ("Zarr", ".zarr"))
                            if name.startswith(prefix)
                        ),
                        None,
                    )
                    if extension is None:
                        raise ValueError(f"No file extension implemented for {name}")
                    backend = backend(file_name=str(path) + extension)
                return backend

            stores[region] = (
                BufferedIOBackend(create_backend, cfg.file_output.get("buffer_size", 2))
                if cfg.file_output.get("thread_io", False)
                else create_backend()
            )
            cleanup.callback(stores[region].close)
        cleanup.pop_all()
    return stores


def pair_packages_ics(
    ics: np.ndarray,
    model_packages: list[str],
    ensemble_size: int,
    batch_ids_produce: list[int],
    batch_size: int,
) -> list[tuple]:
    """Assign complete checkpoint/IC pairs to launcher ranks, preserving batch IDs."""
    num_batches = -(-ensemble_size // batch_size)
    selected = set(batch_ids_produce)
    configs: list[tuple] = []
    for index, pkg in enumerate(model_packages):
        batches = sorted(
            selected.intersection(range(index * num_batches, (index + 1) * num_batches))
        )
        if batches:
            configs.extend((pkg, ic, index * ensemble_size, batches) for ic in ics)
    rank, world_size = worker_rank()
    per_rank = -(-len(configs) // world_size)
    configs = configs[rank * per_rank : (rank + 1) * per_rank]
    logger.info(
        f"rank {rank}: predicting from following models/initial times: {configs}"
    )
    return configs


class TCTracking:
    """Tracker and its per-batch output directory."""

    def __init__(self, tracking_cfg: DictConfig) -> None:
        self.tracker = hydra.utils.instantiate(tracking_cfg.tracker)
        self.out_path = str((Path(tracking_cfg.path) / "cyclones").resolve())


class Downscaling:
    """CorrDiff model and its per-batch output directory."""

    def __init__(self, corrdiff: CorrDiff, path: str) -> None:
        self.corrdiff = corrdiff
        self.out_path = str(Path(path).resolve())


def _load_diagnostic(cfg: DictConfig) -> Any:
    """Load packaged diagnostics or instantiate configured callable models."""
    if "architecture" in cfg:
        cls = hydra.utils.get_class(cfg.architecture)
        package = cfg.get("package", "default")
        return cls.load_model(
            package=(
                cls.load_default_package() if package == "default" else Package(package)
            )
        )
    if "_target_" in cfg:
        return hydra.utils.instantiate(cfg)
    raise ValueError("Diagnostic configuration requires architecture or _target_")


def initialise(
    cfg: DictConfig,
) -> tuple[list[Any], dict, dict, dict, TCTracking | None, DataSource, dict, str | int]:
    """Resolve recipe configuration, models, source, region selections and seeds.

    Region values are selectors, not coordinate signatures. HENSRunner applies
    them to each checkpoint's native output signature and to its emitted fields.
    """
    if "project" not in cfg or not ("file_output" in cfg or "cyclone_tracking" in cfg):
        raise ValueError("Specify project and file_output or cyclone_tracking")
    if cfg.nensemble < 1 or cfg.batch_size < 1 or cfg.nsteps < 0:
        raise ValueError(
            "HENS requires positive ensemble/batch sizes and nonnegative nsteps"
        )
    packages = build_package_list(cfg)
    model_dict = {
        "model": None,
        "class": hydra.utils.get_class(cfg.forecast_model.architecture),
        "package": None,
    }
    output = cfg.get("file_output", {})
    regions = {
        name: dict(bounds)
        for name, bounds in output.get("cropboxes", {"global": {}}).items()
    }
    if "output_vars" in output:
        for region in regions.values():
            region["variable"] = list(dict.fromkeys(output["output_vars"]))
    seed, batches, deterministic = get_reproducibility_settings(cfg, len(packages))
    if deterministic:
        torch.use_deterministic_algorithms(True)
    configs = pair_packages_ics(
        set_initial_times(cfg), packages, cfg.nensemble, batches, cfg.batch_size
    )
    ensure_all_torch_seeds_are_unique(configs, str(seed))
    if configs:
        update_model_dict(model_dict, configs[0][0])
    diagnostics = {
        name: _load_diagnostic(value)
        for name, value in cfg.get("diagnostic_models", {}).items()
    }
    downscaling = {
        name: Downscaling(_load_diagnostic(value), value.path)
        for name, value in cfg.get("corrdiff_models", {}).items()
    }
    tracker = TCTracking(cfg.cyclone_tracking) if "cyclone_tracking" in cfg else None
    return (
        configs,
        model_dict,
        diagnostics,
        downscaling,
        tracker,
        hydra.utils.instantiate(cfg.data_source),
        regions,
        seed,
    )


def update_model_dict(model_dict: dict, root: str) -> dict:
    """Reuse the current checkpoint; offload it before loading a different one."""
    if root != model_dict["package"]:
        if model_dict["model"] is not None:
            model_dict["model"].to("cpu")
        cls = model_dict["class"]
        package = cls.load_default_package() if root == "default" else Package(root)
        model_dict["model"] = cls.load_model(package=package)
        model_dict["package"] = root
    return model_dict


def write_to_disk(
    cfg: DictConfig, ic: str, model_dict: dict, io_dict: dict[str, IOBackend]
) -> None:
    """Export explicitly configured in-memory stores after Pipeline has closed them."""
    for region, backend in io_dict.items():
        if isinstance(backend, BufferedIOBackend):
            backend = backend.backend
        if isinstance(backend, XarrayBackend):
            dataset = backend.root
        elif isinstance(backend, KVBackend):
            dataset = backend.to_xarray()
        else:
            continue
        dataset = dataset.assign_attrs(
            batch_ids=[
                get_batchid_from_ensid(cfg.nensemble, cfg.batch_size, member)
                for member in dataset.ensemble.values
            ],
            torch_version=str(torch.__version__),
            model_package=model_dict["package"],
            batch_size=cfg.batch_size,
            nensemble=cfg.nensemble,
            random_seed=cfg.random_seed,
        )
        dataset.to_netcdf(
            str(_output_path(cfg, ic, model_dict["package"], region)) + ".nc",
            format="NETCDF4",
        )


def get_batchid_from_ensid(
    nensemble_per_package: int, batch_size: int, ensid: int
) -> int:
    """Map a global ensemble member to its original checkpoint-local batch."""
    if batch_size < 1 or nensemble_per_package < 1:
        raise ValueError("Ensemble and batch sizes must be positive")
    num_batches = -(-nensemble_per_package // batch_size)
    return (ensid // nensemble_per_package) * num_batches + (
        ensid % nensemble_per_package
    ) // batch_size
