# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.io import XarrayBackend
from earth2studio.utils.coords import coord_array, coord_array_like
from recipes.hens.src.hens_ensemble import EnsembleBase


class Forecast:
    def __init__(self):
        self.steps = 0

    def input_coords(self):
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {"lead_time": np.array([0], dtype="timedelta64[h]"), "variable": ["t2m"]},
            dynamic=("batch", "time"),
            grid=LatLonGrid(np.array([10.0, 0.0]), np.array([0.0, 10.0])),
        )

    def output_coords(self, x):
        return coord_array_like(
            x, {"lead_time": x.lead_time.values + np.timedelta64(6, "h")}
        )

    def to(self, device):
        return self

    def create_iterator(self, x):
        while True:
            self.steps += 1
            x = (x + 1).assign_coords(lead_time=x.lead_time + np.timedelta64(6, "h"))
            yield x


class Source:
    def __call__(self, time, variable):
        return xr.DataArray(
            np.zeros((len(time), len(variable), 2, 2), dtype=np.float32),
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                "lat": [10.0, 0.0],
                "lon": [0.0, 10.0],
            },
        )


class Diagnostic:
    def __init__(self, source, target):
        self.source, self.target = source, target

    def input_coords(self):
        return coord_array_like(Forecast().input_coords(), {"variable": [self.source]})

    def output_coords(self, x):
        return coord_array_like(x, {"variable": [self.target]})

    def __call__(self, x):
        return (x * 2).assign_coords(variable=[self.target])

    def to(self, device):
        return self


def perturb(x):
    return x + torch.rand(x.shape).numpy()


def run(batches, *, nsteps=2):
    model = Forecast()
    regions = {
        "north": {"lat": np.array([10.0]), "variable": np.array(["d2"])},
        "south": {"lat": np.array([0.0]), "variable": np.array(["t2m"])},
    }
    backends = {name: XarrayBackend() for name in regions}
    runner = EnsembleBase(
        time=[np.datetime64("2024-09-25")],
        nsteps=nsteps,
        nensemble=5,
        prognostic=model,
        data=Source(),
        io_dict=backends,
        perturbation=perturb,
        output_coords_dict=regions,
        batch_size=2,
        ensemble_idx_base=5,
        batch_ids_produce=batches,
        base_seed_string="test",
        device=torch.device("cpu"),
        dx_model_dict=OrderedDict(
            first=Diagnostic("t2m", "d1"), second=Diagnostic("d1", "d2")
        ),
    )
    runner()
    return backends, model, regions


def test_initial_condition_two_advances_and_independent_regions():
    backends, model, regions = run([3, 4, 5])
    assert model.steps == 6
    north, south = backends["north"].root, backends["south"].root
    np.testing.assert_array_equal(
        north.lead_time.values, np.array([0, 6, 12], dtype="timedelta64[h]")
    )
    np.testing.assert_array_equal(north.ensemble, [5, 6, 7, 8, 9])
    np.testing.assert_allclose(north.d2.diff("lead_time"), 4, atol=1e-6)
    np.testing.assert_allclose(south.t2m.diff("lead_time"), 1, atol=1e-6)
    assert north.lat.item() == 10.0
    assert south.lat.item() == 0.0
    assert set(regions["north"]) == {"lat", "variable"}


def test_selected_batch_and_short_final_batch_reproduce():
    complete, _, _ = run([3, 4, 5])
    for batch, members in [(4, [7, 8]), (5, [9])]:
        subset, _, _ = run([batch])
        for region in subset:
            xr.testing.assert_equal(
                subset[region].root, complete[region].root.sel(ensemble=members)
            )


def test_zero_steps_publishes_only_initial_condition():
    backends, model, _ = run([3], nsteps=0)
    assert model.steps == 0
    assert backends["north"].root.sizes["lead_time"] == 1


def test_negative_steps_rejected():
    with pytest.raises(ValueError):
        run([3], nsteps=-1)


def test_regional_longitudes_wrap():
    from recipes.hens.src.hens_ensemble import _regional_field

    x = xr.DataArray(
        [1.0, 2.0, 3.0], dims="lon", coords={"lon": [0.0, 180.0, 350.0]}
    ).expand_dims(lat=[0.0])
    result = _regional_field(x, {"lon": np.array([-180.0, -10.0, 0.0])})
    np.testing.assert_array_equal(result.values, [[2.0, 3.0, 1.0]])


def test_skill_scaling_uses_variable_labels(tmp_path):
    from recipes.hens.src.hens_perturbation import HENSPerturbation

    skill = tmp_path / "skill.nc"
    xr.Dataset(
        {"value": (("channel", "lead_time"), [[3.0]])},
        coords={"channel": ["t2m"], "lead_time": [48]},
    ).to_netcdf(skill)
    noise = HENSPerturbation.get_noise_vector(None, Forecast(), str(skill), 2.0, "t2m")
    assert noise.shape == (1, 1, 1, 1, 1, 1)
    assert noise.item() == 6.0


def test_utility_output_and_idle_worker(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from recipes.hens.src.hens_utilities import (
        initialise_output,
        initialize_output_coords,
        pair_packages_ics,
    )

    cfg = OmegaConf.create(
        {
            "project": "test",
            "file_output": {
                "path": str(tmp_path),
                "format": {
                    "_target_": "earth2studio.io.NetCDF4Backend",
                    "_partial_": True,
                },
            },
        }
    )
    stores = initialise_output(
        cfg,
        np.datetime64("2024-09-25"),
        {"package": "default"},
        {"north": {}, "south": {}},
    )
    for store in stores.values():
        store.close()
    assert (tmp_path / "north" / "test_2024-09-25_pkg_default.nc").exists()
    assert (tmp_path / "south" / "test_2024-09-25_pkg_default.nc").exists()
    assert set(
        initialize_output_coords(OmegaConf.create({}), np.array([0.0]), np.array([0.0]))
    ) == {"lat", "lon"}
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert (
        pair_packages_ics([np.datetime64("2024-09-25")], ["default"], 2, [0], 2) == []
    )


def test_auxiliary_files_without_regional_output(tmp_path):
    from types import SimpleNamespace

    class Tracker:
        def __init__(self):
            self.resets = 0
            self.frames = []

        def to(self, device):
            return self

        def input_coords(self):
            return Forecast().input_coords().isel(lead_time=0, drop=True)

        def reset_path_buffer(self):
            self.resets += 1
            self.frames = []

        def __call__(self, x):
            assert "lead_time" not in x.dims
            self.frames.append(x.mean(("variable", "lat", "lon")))
            return xr.concat(self.frames, dim="step")

    tracker = Tracker()
    runner = EnsembleBase(
        time=[np.datetime64("2024-09-25")],
        nsteps=2,
        nensemble=3,
        prognostic=Forecast(),
        data=Source(),
        io_dict={},
        perturbation=perturb,
        output_coords_dict={},
        batch_size=2,
        device=torch.device("cpu"),
        cd_model_dict={
            "test": SimpleNamespace(
                corrdiff=Diagnostic("t2m", "downscaled"), out_path=tmp_path
            )
        },
        cyclone_tracking=SimpleNamespace(tracker=tracker, out_path=tmp_path),
        pkg="seed1",
    )
    runner()
    assert tracker.resets == 2
    for batch, members in [(0, [0, 1]), (1, [2])]:
        with xr.open_dataset(
            tmp_path / f"corrdiff_test_pkg_1_2024-09-25T00:00:00_batch_{batch}.nc"
        ) as ds:
            np.testing.assert_array_equal(ds.ensemble, members)
            assert ds.sizes["lead_time"] == 3
            np.testing.assert_allclose(ds.downscaled.diff("lead_time"), 2, atol=1e-6)
        with xr.open_dataset(
            tmp_path / f"tracks_pkg_1_2024-09-25T00:00:00_batch_{batch}.nc"
        ) as ds:
            assert ds.sizes["step"] == 3


def test_pipeline_closes_outputs_after_forecast_failure():
    class BrokenForecast(Forecast):
        def create_iterator(self, x):
            raise RuntimeError("forecast failed")
            yield

    backend = XarrayBackend()
    runner = EnsembleBase(
        time=[np.datetime64("2024-09-25")],
        nsteps=1,
        nensemble=2,
        prognostic=BrokenForecast(),
        data=Source(),
        io_dict={"global": backend},
        perturbation=perturb,
        output_coords_dict={"global": {}},
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="forecast failed"):
        runner()
    assert backend._closed


def test_memory_output_has_reproduction_metadata(tmp_path):
    from omegaconf import OmegaConf

    from recipes.hens.src.hens_utilities import write_to_disk

    backends, _, _ = run([4])
    for region in backends:
        (tmp_path / region).mkdir()
    cfg = OmegaConf.create(
        {
            "project": "test",
            "nensemble": 5,
            "batch_size": 2,
            "random_seed": "42",
            "file_output": {"path": str(tmp_path)},
        }
    )
    write_to_disk(cfg, "2024-09-25", {"package": "default"}, backends)
    with xr.open_dataset(tmp_path / "north" / "test_2024-09-25_pkg_default.nc") as ds:
        np.testing.assert_array_equal(ds.attrs["batch_ids"], [4, 4])
        assert ds.attrs["random_seed"] == "42"


def test_idle_seed_validation_and_generated_seed():
    from omegaconf import OmegaConf

    from recipes.hens.src.hens_utilities_reproduce import (
        ensure_all_torch_seeds_are_unique,
        get_reproducibility_settings,
    )

    ensure_all_torch_seeds_are_unique([], "42")
    cfg = OmegaConf.create(
        {"nensemble": 2, "batch_size": 2, "forecast_model": {"max_num_checkpoints": 1}}
    )
    seed, _, _ = get_reproducibility_settings(cfg)
    assert cfg.random_seed == seed
    assert get_reproducibility_settings(cfg)[0] == seed


@pytest.mark.parametrize("batch_size", [1, 3])
def test_odd_bred_vector_batches_reproduce(batch_size):
    from earth2studio.perturbation import HemisphericCentredBredVector

    class PairedVectors(HemisphericCentredBredVector):
        def create_generator(self, time, generator_size, device):
            for _ in range(generator_size // 2):
                noise = torch.rand((1, 1, 1, 1, 2, 2), device=device)
                yield noise
                yield -noise

    def forecast(batches):
        backend = XarrayBackend()
        runner = EnsembleBase(
            time=[np.datetime64("2024-09-25")],
            nsteps=0,
            nensemble=4 * batch_size,
            prognostic=Forecast(),
            data=Source(),
            io_dict={"global": backend},
            output_coords_dict={"global": {}},
            perturbation=PairedVectors(Forecast(), Source(), perturb),
            batch_size=batch_size,
            batch_ids_produce=batches,
            device=torch.device("cpu"),
            base_seed_string="test",
        )
        runner()
        return backend.root

    complete = forecast(None)
    for batch in [1, 2, 3]:
        xr.testing.assert_equal(
            forecast([batch]),
            complete.isel(ensemble=slice(batch * batch_size, (batch + 1) * batch_size)),
        )


def test_cuda_initial_condition_transfer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from earth2studio.run import WorkItem
    from recipes.hens.src.hens_ensemble import HENSRunner

    runner = HENSRunner(Forecast(), Source(), perturb, {}, 2, 2, "test")
    runner.to(torch.device("cuda:0"))
    field = runner._fetch(
        WorkItem(np.datetime64("2024-09-25"), np.timedelta64(0, "h"), (0, 1))
    )
    assert field.e2s.to_torch()[0].is_cuda
