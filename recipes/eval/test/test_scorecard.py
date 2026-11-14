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

"""Scorecard baselines, source adapters and campaign configs.

The climatology baseline is a prognostic model whose forcing slot is the
climatology at each valid time; it runs here through the library's
:class:`earth2studio.run.PrognosticRunner` and the recipe's forecast builder
with fake sources on the small test grid.  Hydra composes the campaign
configs from the scorecard's config directory without loading a model.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import get_object
from omegaconf import OmegaConf, open_dict
from scorecard import run_scorecard
from scorecard.run_scorecard import (
    compose_campaign,
    labelled_score_stores,
    run_path,
    score_stores,
    source_labels,
)
from scorecard.utils.baselines import ClimatologyForecast, PersistenceBaseline
from scorecard.utils.sources import (
    STORMCAST_LAT_LIM,
    STORMCAST_LON_LIM,
    stormcast_conditioning,
    stormcast_grid,
)

from earth2studio.data import RegriddedSource
from earth2studio.grids import LatLonGrid, resolve_grid
from earth2studio.run import PrognosticRunner, work_items
from test.conftest import SMALL_LAT, SMALL_LON, VARIABLES, compose_cfg

RECIPE_DIR = Path(__file__).resolve().parents[1]
SCORECARD_CFG = RECIPE_DIR / "scorecard" / "cfg"
CAMPAIGNS = sorted(p.stem for p in (SCORECARD_CFG / "campaign").glob("*.yaml"))

T0 = np.datetime64("2025-01-01T00:00:00")
H6 = np.timedelta64(6, "h")


class _Zeros:
    """Fake analysis source on the small test grid."""

    def __init__(self) -> None:
        self.times: list[np.datetime64] = []

    def _field(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            np.zeros(
                (len(time), len(variable), len(SMALL_LAT), len(SMALL_LON)), np.float32
            ),
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            },
        )

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        self.times.extend(np.asarray(time, dtype="datetime64[ns]"))
        return self._field(time, variable)


class _Climatology(_Zeros):
    """Fake climatology: every point holds the valid time in hours after T0."""

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        da = super().__call__(time, variable)
        hours = (
            (
                np.asarray(da.time.values, dtype="datetime64[ns]")
                - np.datetime64(T0, "ns")
            )
            / np.timedelta64(1, "h")
        ).astype(np.float32)
        da.values[:] = hours[:, None, None, None]
        return da


def _hours(time: np.datetime64) -> float:
    return float(
        (np.datetime64(time, "ns") - np.datetime64(T0, "ns")) / np.timedelta64(1, "h")
    )


class TestClimatologyForecast:
    def test_signatures(self, small_grid: LatLonGrid) -> None:
        model = ClimatologyForecast(VARIABLES, grid=small_grid)
        inputs = model.input_coords()
        assert tuple(inputs.dims) == ("batch", "lead_time", "variable", "lat", "lon")
        assert inputs.lead_time.values.tolist() == [np.timedelta64(0, "h")]
        forcing = model.forcing_coords()
        assert tuple(forcing.dims) == tuple(inputs.dims)
        assert forcing.lead_time.values.tolist() == [H6]
        assert list(forcing["variable"].values) == VARIABLES
        outputs = model.output_coords(inputs)
        assert outputs.lead_time.values.tolist() == [H6]
        assert outputs.sizes["lat"] == len(SMALL_LAT)

    def test_load_model(self) -> None:
        model = ClimatologyForecast.load_model(variables=VARIABLES, step_hours=12)
        assert model.dt == np.timedelta64(12, "h")
        assert model.input_coords().sizes["lat"] == 721
        assert ClimatologyForecast.load_default_package() is None
        with pytest.raises(ValueError, match="variables"):
            ClimatologyForecast.load_model(variables=[])

    def test_default_sources(self, small_grid: LatLonGrid) -> None:
        model = ClimatologyForecast(VARIABLES, grid=small_grid)
        analysis, climatology = model.default_sources()
        assert analysis is None
        assert type(climatology).__name__ == "WB2Climatology"

    def test_runner_forecasts_climatology_at_valid_times(
        self, small_grid: LatLonGrid
    ) -> None:
        model = ClimatologyForecast(VARIABLES, grid=small_grid)
        analysis, climatology = _Zeros(), _Climatology()
        runner = PrognosticRunner(model, source=analysis, forcing=(climatology,))
        assert runner.step == H6
        (item,) = work_items([T0], 3 * H6)

        forecasts = [out["forecast"] for out in runner.run_item(item)]
        leads = [f.lead_time.values[0] for f in forecasts]
        assert leads == [k * H6 for k in range(4)]
        for k, f in enumerate(forecasts):
            assert f.sizes["lead_time"] == 1
            assert list(f["variable"].values) == VARIABLES
            np.testing.assert_allclose(f.values, 6.0 * k)
        # The runner reads the climatology at the forecast valid times only.
        assert sorted({_hours(t) for t in climatology.times}) == [6.0, 12.0, 18.0]
        assert sorted({_hours(t) for t in analysis.times}) == [0.0]

    def test_data_requests_cover_forcing(self, small_grid: LatLonGrid) -> None:
        model = ClimatologyForecast(VARIABLES, grid=small_grid)
        climatology = _Climatology()
        runner = PrognosticRunner(model, source=_Zeros(), forcing=(climatology,))
        (item,) = work_items([T0], 3 * H6)
        requests = [r for r in runner.data_requests(item) if r.source is climatology]
        valid = sorted(
            {_hours(t + lead) for r in requests for t in r.time for lead in r.lead_time}
        )
        assert valid == [6.0, 12.0, 18.0]

    def test_forecast_runner_broadcasts_members(self, small_grid: LatLonGrid) -> None:
        model = ClimatologyForecast(VARIABLES, grid=small_grid)
        runner = PrognosticRunner(model, source=_Zeros(), forcing=(_Climatology(),))
        items = work_items([T0], 2 * H6, members=2, member_batch=2)
        forecasts = [out["forecast"] for out in runner.run_item(items[0])]
        assert forecasts[-1].dims[0] == "ensemble"
        assert forecasts[-1].ensemble.values.tolist() == [0, 1]
        np.testing.assert_allclose(forecasts[-1].values, 12.0)

    def test_forecast_builder_wires_forcing_sources(self, tmp_path: Path) -> None:
        from src.runners import forecast

        cfg = compose_cfg(tmp_path, ["nsteps=2"])
        domain = {"lat": SMALL_LAT.tolist(), "lon": SMALL_LON.tolist()}
        with open_dict(cfg):
            cfg.model = {
                "_target_": "scorecard.utils.baselines.ClimatologyForecast",
                "variable": list(VARIABLES),
                "grid": "latlon-small",
            }
            cfg.data_source = {
                "_target_": "earth2studio.data.Random",
                "domain_coords": domain,
            }
            cfg.forcing_sources = [
                {"_target_": "earth2studio.data.Random", "domain_coords": domain}
            ]
            cfg.output.variables = list(VARIABLES)
        runner = forecast(cfg, prefer_predownloaded=False)
        assert len(runner.forcing) == 1
        assert set(runner.source_roles) == {"data", "forcing0"}
        schema = runner.output_coords(2 * H6)["forecast"]
        assert schema.lead_time.values.tolist() == [k * H6 for k in range(3)]
        (item,) = work_items([T0], 2 * H6)
        assert len(list(runner.run_item(item))) == 3


class TestPersistenceBaseline:
    def test_loader(self) -> None:
        assert PersistenceBaseline.load_default_package() is None
        model = PersistenceBaseline.load_model(variables=["t2m", "z500"])
        inputs = model.input_coords()
        assert list(inputs["variable"].values) == ["t2m", "z500"]
        assert inputs.sizes["lat"] == 721 and inputs.sizes["lon"] == 1440
        assert inputs.lat.values[0] == 90.0 and inputs.lat.values[-1] == -90.0
        with pytest.raises(ValueError, match="variables"):
            PersistenceBaseline.load_model(variables=None)


class TestStormCastSources:
    def test_grid_is_hrrr_window(self) -> None:
        grid = stormcast_grid()
        parent = resolve_grid("hrrr")
        assert grid.shape == (512, 640)
        np.testing.assert_array_equal(grid.y, parent.y[slice(*STORMCAST_LAT_LIM)])
        np.testing.assert_array_equal(grid.x, parent.x[slice(*STORMCAST_LON_LIM)])
        assert grid.crs == parent.crs

    def test_conditioning_regrids_onto_model_grid(self) -> None:
        source = _Zeros()
        wrapped = stormcast_conditioning(source)
        assert isinstance(wrapped, RegriddedSource)
        assert wrapped.source is source
        assert wrapped.regridder.target_grid.shape == (512, 640)


def _targets(node, path: str = ""):
    """Yield ``(path, mapping)`` for every mapping holding a ``_target_``."""
    if isinstance(node, dict):
        if "_target_" in node:
            yield path, node
        for key, value in node.items():
            yield from _targets(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _targets(value, f"{path}[{i}]")


def test_score_stores_follow_the_composed_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCORECARD_OUT", str(tmp_path))
    run = tmp_path / "fcn3_2025_scorecard"

    def stores(*overrides: str) -> list[Path]:
        return score_stores(compose_campaign("fcn3_2025_scorecard", list(overrides)))

    assert stores() == [run / "scores.zarr"]
    # Hydra reads null as None, so the online scores land in store_name.
    assert stores("scoring.online.scores_store=null") == [run / "scores.zarr"]
    assert stores("scoring.online.scores_store=online.zarr") == [run / "online.zarr"]
    assert stores("scoring.online.verification_sources={era5:null,ifs:null}") == [
        run / "scores__era5.zarr",
        run / "scores__ifs.zarr",
    ]
    assert stores(
        "scoring.mode=offline", "scoring.output.store_name=scores_offline.zarr"
    ) == [run / "scores_offline.zarr"]
    # A relative output.path is relative to the recipe root, where the
    # scripts run.
    assert stores("output.path=outputs/elsewhere") == [
        run_path("outputs/elsewhere/scores.zarr")
    ]


def test_each_reference_store_names_its_own_truth(tmp_path, monkeypatch) -> None:
    """Online references label their stores with their own archives."""
    monkeypatch.setenv("SCORECARD_OUT", str(tmp_path))
    references = (
        "scoring.online.verification_sources={era5:null,"
        "ifs:{_target_:earth2studio.data.IFS},"
        "staged:{_target_:src.data.PredownloadedSource,store_path:'/data/ifs.zarr'}}"
    )
    cfg = compose_campaign("fcn3_2025_scorecard", [references])
    truth = {
        store.name: labels.get("verification_source")
        for store, labels in labelled_score_stores(cfg)
    }
    assert truth == {
        "scores__era5.zarr": "ARCO_ERA5",  # null keeps the default truth
        "scores__ifs.zarr": "IFS",
        "scores__staged.zarr": None,  # a staged store does not name its archive
    }
    run = tmp_path / "fcn3_2025_scorecard"
    assert [store for store, _ in labelled_score_stores(cfg)] == [
        run / "scores__era5.zarr",
        run / "scores__ifs.zarr",
        run / "scores__staged.zarr",
    ]


def test_scorecard_configs_shadow_no_recipe_config() -> None:
    # The scripts search the scorecard's cfg first and the launcher the
    # recipe's; with no file in both, the two compose the same config.
    recipe = {
        p.relative_to(RECIPE_DIR / "cfg") for p in (RECIPE_DIR / "cfg").rglob("*.yaml")
    }
    scorecard = {p.relative_to(SCORECARD_CFG) for p in SCORECARD_CFG.rglob("*.yaml")}
    assert not recipe & scorecard


def test_launcher_composes_what_the_scripts_compose(tmp_path, monkeypatch) -> None:
    """The launcher's config equals the one score.py itself prints."""
    overrides = [
        "scoring.online.scores_store=null",
        "scoring.online.verification_sources={era5:null}",
    ]
    monkeypatch.setenv("SCORECARD_OUT", str(tmp_path))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(RECIPE_DIR), env.get("PYTHONPATH", "")) if p
    )
    printed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [
            sys.executable,
            "score.py",
            "--config-dir",
            str(SCORECARD_CFG),
            "campaign=fcn3_2025_scorecard",
            *overrides,
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=RECIPE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert printed.returncode == 0, printed.stderr[-4000:]
    composed = OmegaConf.to_container(
        compose_campaign("fcn3_2025_scorecard", overrides), resolve=True
    )
    assert composed == yaml.safe_load(printed.stdout)


@pytest.mark.parametrize("pin", [False, True])
def test_launcher_pins_ranks_through_the_wrapper(monkeypatch, pin) -> None:
    calls = []
    monkeypatch.setattr(
        run_scorecard.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw))
    )
    if pin:
        monkeypatch.setenv("SCORECARD_PIN_GPU", "1")
    else:
        monkeypatch.delenv("SCORECARD_PIN_GPU", raising=False)
    run_scorecard.launch("main.py", "fcn3_2025_scorecard", [], 8)
    [(cmd, kw)] = calls
    wrapper = str(SCORECARD_CFG.parent / "utils" / "rank_entry.sh")
    if pin:
        at = cmd.index("--no-python")
        assert cmd[at + 1 : at + 3] == [wrapper, "main.py"]
        assert kw["env"]["SCORECARD_PYTHON"] == sys.executable
    else:
        assert "--no-python" not in cmd and kw["env"] is None


@pytest.mark.parametrize(
    ("local_rank", "visible", "expected"),
    [("1", "4,5", "5"), ("0", "GPU-a,GPU-b", "GPU-a"), ("2", None, "2")],
)
def test_rank_entry_picks_the_rank_gpu(local_rank, visible, expected) -> None:
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env.update(LOCAL_RANK=local_rank, SCORECARD_PYTHON=sys.executable)
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    shown = subprocess.run(  # noqa: S603 - fixed script and interpreter
        [
            str(SCORECARD_CFG.parent / "utils" / "rank_entry.sh"),
            "-c",
            "import os; e = os.environ; "
            "print(e['CUDA_VISIBLE_DEVICES'], e['LOCAL_RANK'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert shown.returncode == 0, shown.stderr
    assert shown.stdout.split() == [expected, "0"]


def test_rank_entry_rejects_a_rank_beyond_the_visible_gpus() -> None:
    env = {**os.environ, "LOCAL_RANK": "2", "CUDA_VISIBLE_DEVICES": "4,5"}
    shown = subprocess.run(  # noqa: S603 - fixed script
        [str(SCORECARD_CFG.parent / "utils" / "rank_entry.sh"), "-c", "pass"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert shown.returncode == 1 and "lists 2 device(s)" in shown.stderr


def test_launcher_stamps_the_store_scoring_wrote(tmp_path, monkeypatch) -> None:
    import zarr

    monkeypatch.setenv("SCORECARD_OUT", str(tmp_path))
    # main() extends these for the scripts it launches; restore them after.
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    )
    run = tmp_path / "fcn3_2025_scorecard"
    launched: list[tuple[str, list[str]]] = []

    def fake_launch(entry, campaign, overrides, ngpu) -> None:
        launched.append((entry, list(overrides)))
        # The online scores land in scoring.output.store_name.
        zarr.open_group(str(run / "scores.zarr"), mode="w")

    monkeypatch.setattr(run_scorecard, "launch", fake_launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scorecard.py",
            "fcn3_2025_scorecard",
            "score",
            "--ngpu",
            "1",
            "--",
            "scoring.online.scores_store=null",
        ],
    )
    assert run_scorecard.main() == 0
    assert launched == [("score.py", ["scoring.online.scores_store=null"])]
    assert run_scorecard.scores_complete(run / "scores.zarr")
    assert not (run / "null").exists()
    # The run's source labels travel with the provenance, and the exporter
    # reads them back from the store.
    from scorecard.export_scores import data_sources

    expected = {
        "ic_source": "ARCO_ERA5",
        "verification_source": "ARCO_ERA5",
        "lat_weights": True,
    }
    assert dict(zarr.open_group(str(run / "scores.zarr")).attrs["sources"]) == expected
    assert data_sources(run) == expected


@pytest.mark.parametrize(
    "campaign, ic, truth, lat_weights",
    [
        # A composite of two archives names both.
        ("fuxi_2025_scorecard", "ARCO_ERA5+NCAR_ERA5", "ARCO_ERA5", True),
        # A cropping adapter names the source it wraps; uniform weights.
        ("stormcast_2025_events", "HRRR", "HRRR", False),
        # Stores of the fcn3 run read as the source that filled them.
        ("climatology_2025_scorecard", "ARCO_ERA5", "ARCO_ERA5", True),
        ("aurora_2025_scorecard", "ARCO_ERA5", "ARCO_ERA5", True),
        ("aurora1p5ens_2025_scorecard", "ARCO_ERA5", "ARCO_ERA5", True),
    ],
)
def test_source_labels_follow_the_recipes_resolution(
    campaign, ic, truth, lat_weights
) -> None:
    cfg = OmegaConf.to_container(compose_campaign(campaign, []), resolve=False)
    assert source_labels(cfg) == {
        "ic_source": ic,
        "verification_source": truth,
        "lat_weights": lat_weights,
    }


class TestCampaignConfigs:
    @pytest.fixture(params=CAMPAIGNS)
    def campaign(self, request) -> tuple[str, object]:
        overrides = [
            f"campaign={request.param}",
            f'hydra.searchpath=["file://{SCORECARD_CFG}"]',
        ]
        with initialize_config_dir(
            config_dir=str(RECIPE_DIR / "cfg"), version_base=None
        ):
            cfg = compose(config_name="default", overrides=overrides)
        return request.param, cfg

    @pytest.mark.parametrize("model", ["aurora1p5ens", "atlas_crps"])
    def test_healda_campaign_differs_from_era5_only_in_its_ics(self, model) -> None:
        # The two campaigns compare initializations like for like, so everything
        # but the initial conditions has to stay in step.
        def compose_campaign(name: str) -> object:
            with initialize_config_dir(
                config_dir=str(RECIPE_DIR / "cfg"), version_base=None
            ):
                return compose(
                    config_name="default",
                    overrides=[
                        f"campaign={name}",
                        f'hydra.searchpath=["file://{SCORECARD_CFG}"]',
                    ],
                )

        era5 = compose_campaign(f"{model}_2025_scorecard")
        healda = compose_campaign(f"{model}_healda_2025_scorecard")
        assert healda.model.da.architecture == "earth2studio.models.da.HealDAv2"
        assert healda.model.forecast.architecture == era5.model.architecture
        for key in (
            "start_times",
            "nsteps",
            "ensemble_size",
            "random_seed",
            "perturbation",
            "output.variables",
            "output.regrid",
            "scoring.variables",
            "scoring.lat_weights",
            "scoring.nan_policy",
            "scoring.online.members_per_rank",
            "scoring.online.mae",
            "scoring.online.lsd",
            "verification_source",
        ):
            assert OmegaConf.select(healda, key) == OmegaConf.select(era5, key), key

    def test_no_deleted_references(self) -> None:
        for name in CAMPAIGNS:
            text = (SCORECARD_CFG / "campaign" / f"{name}.yaml").read_text()
            assert "scorecard.utils.pipelines" not in text, name
            assert "src.pipelines" not in text, name
            assert "\npipeline:" not in text, name

    def test_targets_resolve_and_accept_their_arguments(self, campaign) -> None:
        # Every ``_target_`` imports, and takes the keys given next to it.
        name, cfg = campaign
        for path, node in _targets(OmegaConf.to_container(cfg, resolve=False)):
            target = get_object(node["_target_"])
            params = inspect.signature(target).parameters
            var_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            keys = [k for k in node if not str(k).startswith("_")]
            unknown = [k for k in keys if k not in params]
            assert var_kw or not unknown, f"{name}:{path} -> {unknown}"

    def test_composes_on_forecast_runner(self, campaign) -> None:
        name, cfg = campaign
        assert "pipeline" not in cfg
        # Campaigns initialized from data-assimilation analyses run the DA model
        # ahead of the same forecast runner.
        expected = "assimilation_forecast" if "da" in cfg.model else "forecast"
        assert cfg.runner == expected
        assert name.startswith(f"{cfg.project}_")
        assert cfg.scoring.mode == "online"
        assert cfg.output.retain == "none"
        if name.startswith(("aurora", "fcn_")):
            assert cfg.output.regrid.grid == "latlon-0.25deg"
            assert cfg.output.regrid.method == "nearest"
        else:
            assert not cfg.output.get("regrid")
        if name.startswith("climatology"):
            assert (
                cfg.model.architecture
                == "scorecard.utils.baselines.ClimatologyForecast"
            )
            assert len(cfg.forcing_sources) == 1
            assert (
                cfg.forcing_sources[0]["_target_"] == "earth2studio.data.WB2Climatology"
            )
        if name.startswith("persistence"):
            assert (
                cfg.model.architecture
                == "scorecard.utils.baselines.PersistenceBaseline"
            )
            assert list(cfg.model.load_args.variables) == list(cfg.output.variables)
        if name.startswith("stormcast"):
            assert cfg.forcing_sources[0]["_target_"] == (
                "scorecard.utils.sources.stormcast_conditioning"
            )
            assert (
                cfg.forcing_sources[0].source["_target_"]
                == "earth2studio.data.ARCO_ERA5"
            )
            assert cfg.ic_source["_target_"] == "earth2studio.data.HRRR"
            assert cfg.verification_source["_target_"] == (
                "scorecard.utils.sources.stormcast_window"
            )
            assert "conditioning_data_source" not in (cfg.model.get("load_args") or {})
