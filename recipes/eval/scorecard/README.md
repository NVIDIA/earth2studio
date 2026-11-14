# Scorecard

Generates the source score JSON files behind the documentation's
[Scorecards pages](../../../docs/scorecard). Every model runs the same
campaign: 48 initial conditions over a 14-day horizon, with ERA5
verification through ARCO_ERA5. The four monthly initial conditions rotate
through the 00Z/06Z/12Z/18Z hours.
The campaigns use the evaluation recipe's **online scoring**: the inference
loop reduces every forecast to a small set of statistics and never writes
the raw forecast store.

The scorecard computes every metric globally and for the regional splits in
`cfg/regions/standard.yaml`. Any named lat/lon boxes work; that file is
just the scorecard's choice. The exports also carry seasonal, monthly, and
per-init-hour breakdowns, a per-initial-condition skill grid, and the
persistence and climatology baseline campaigns. An
[event campaign](#event-campaigns) adds skill during named weather
events: a time window paired with a region.

## Layout

```text
scorecard/
  cfg/campaign/<model>_2025_scorecard.yaml   self-contained evaluation campaigns
  cfg/campaign/<model>_2025_events.yaml      event campaigns: named windows + boxes
  cfg/regions/standard.yaml     the regional splits every campaign shares
  run_scorecard.py        predownload -> infer (scores online) -> score/prune
  export_scores.py        scores.zarr -> exports/eval_scores_<model>*.json
  utils/baselines.py      persistence and climatology baselines (prognostic models)
  utils/sources.py        source adapters for the StormCast grid (see below)
  models/<model>/outputs/ run data: stats.zarr + scores.zarr (not tracked)
  data/                   shared ERA5 stores (not tracked)
```

Note: Some of the following folders will be generated after running the scorecard recipe.

## Usage

On a GPU node with the recipe environment:

```bash
# 1. Run a campaign from cfg/campaign/
python run_scorecard.py fcn3_2025_scorecard

# 2. Export the scores JSONs and copy them into the docs
python export_scores.py fcn3 --docs

# 3. Regenerate the docs pages
python ../../../docs/generate_scorecard.py
```

The score JSONs are not stored in the git repository. The docs build
fetches them from the `scorecard/` folder of the
[Earth2Studio assets dataset](https://huggingface.co/datasets/nvidia/earth2studio-assets)
on Hugging Face, where each model keeps its exports in its own folder:
`scorecard/<model>/eval_scores_<model>*.json`. Log in once with
`hf auth login`, or set `HF_TOKEN`, then publish straight from the
export step:

```bash
python export_scores.py fcn3 --docs --upload
```

Without a repository name, `--upload` writes to
`<your user>/earth2studio-assets`, creating that public dataset on first
use, or to `$SCORECARD_DATA_REPO` when set. A docs build reads the same
dataset when `SCORECARD_DATA_REPO` names it. The fork's docs preview
workflow does, and falls back to the upstream dataset until the personal
one exists. To propose the files to the upstream dataset instead, open a
pull request there:

```bash
m=fcn3  # the model whose campaign was run
hf upload nvidia/earth2studio-assets docs/_static/scorecard scorecard/$m \
  --repo-type dataset --create-pr \
  --include "eval_scores_$m.json" "eval_scores_${m}_*.json"
```

Because the campaigns score online, the `infer` stage already derives
`scores.zarr` from the accumulated `stats.zarr`. The `score` stage is a
cheap idempotent re-derivation, and `prune` does nothing because no raw
store exists.

## Scaling and portability

Launches default to a single node (`torchrun --standalone`, all local
GPUs). For multi-node runs, set `TORCHRUN_ARGS` with your rendezvous flags,
or invoke the recipe entry points (`predownload.py`, `main.py`, `score.py`)
with your own launcher; the driver adds nothing they do not already support.
Set `SCORECARD_PIN_GPU=1` to start each rank through `utils/rank_entry.sh`,
which makes only that rank's GPU visible before CUDA starts. Without it,
library threads that never select a device put their memory on the first
GPU. In an 8-rank HealDA run, each other rank held about 2.6 GiB there.
Every stage resumes (`resume: true` in the campaigns), so a large campaign
can advance incrementally across short queue allocations rather than
needing one long job.

The JSON exports carry everything a docs page needs: metric curves per
variable and lead time aggregated over initial conditions, units, and
variable groups. The exporter writes one file per regional split
(`eval_scores_<model>_region_<name>.json`) plus monthly, hourly, and
per-initial-condition grid files. The docs plot fetches these lazily when
the matching selector is first used, so the initial page load only pays
for the main file.

Multiple campaigns per model are supported by construction: a campaign is just
another self-contained config, and the docs plot selects its data file by URL
key. Only the default file naming (`eval_scores_<model>.json`) assumes one
campaign per model — exporting a second campaign under a distinct name (and
listing it in `mkdocs.yml`) is the whole extension.

## Event campaigns

An event campaign scores a model on named weather events. Each event is
a time window plus a region, declared under `scoring.events`. The
[evaluation recipe README](../README.md#event-scoring) documents the
block. `cfg/campaign/stormcast_2025_events.yaml` is the template: the
regional StormCast model on severe convective episodes of 2025. It sets
`start_times: null` and adds one initial condition every 3 h, from 12 h
before each window to its end. Every lead time out to 12 h then has
valid times inside the window. The model domain is the event region. Boxes on
the HRRR grid would use its projection coordinates rather than latitude
and longitude.

```bash
python run_scorecard.py stormcast_2025_events
python export_scores.py stormcast --docs
```

StormCast runs through the recipe's default forecast runner,
`runner: forecast`. The model declares its global conditioning as a
*forcing slot*, so the campaign lists one entry under `forcing_sources`.
That entry is ERA5 from ARCO wrapped by
`scorecard.utils.sources.stormcast_conditioning`, which interpolates the
fields onto the model's HRRR window. The model cuts that window from the
registered `hrrr` grid, and the window has no registered name of its own,
so the helper rebuilds it. The runner selects the window out of the full
HRRR grid for the initial conditions, and
`scorecard.utils.sources.stormcast_window` crops the HRRR truth the same
way. The campaign streams everything:
initial conditions and truth from HRRR, conditioning from ERA5, with no
stores fetched ahead of time, so its only artifacts are `stats.zarr` and
`scores.zarr`. Run it with the `infer` and `score` stages only. Truth is
the HRRR analysis and the scores carry uniform weights.

`export_scores.py` looks for `models/<model>/outputs/<model>_2025_events`
next to the main run, or takes `--events-run PATH`. A model without a
main campaign run, such as StormCast, exports the events run as its main
file: the headline curves pool every event's initial conditions. Either
way it writes `eval_scores_<model>_events.json`: one curve set per event,
restricted to the event's region and windowed on valid time, with the
number of initial conditions behind it. The docs plot lists the events in
its Event selector, next to the regional, monthly, and hourly splits. It
draws each event against the main file's all-IC curve. Events declared
in a global campaign export the same way, and their boxes stay out of
the year-round Region selector.

Keep an event campaign's ensemble size and horizon comparable to the
model's main campaign when it has one, since the docs plot draws each
event against that campaign's curve.

## Off-grid models and baselines

Every campaign runs the recipe's `runner: forecast`. Models off the
721 × 1440 ERA5 grid, such as Aurora and FCN with 720 rows, set

```yaml
output:
    regrid:
        grid: latlon-0.25deg
        method: nearest
```

so the recipe gathers their output onto the shared grid before scoring
and reuses the shared verification store. The block is a no-op for models
already on that grid. The ERA5 verification store itself comes from one
of two places. The `predownload` stage writes the run's own
`<output.path>/verification.zarr`, always a separate store. Or a
bring-your-own `verification_source` points a
`src.data.PredownloadedSource` at an existing store, as the Aurora and
baseline campaigns do with the fcn3 run's store.

The baselines in `utils/baselines.py` are prognostic models on the 1.0
protocol, selected through `model.architecture`:

* `PersistenceBaseline`: `earth2studio.models.px.Persistence` on the
  registered `latlon-0.25deg` grid.
* `ClimatologyForecast`: the climatology at each valid time is the
  model's forcing slot. The campaign names the climatology source under
  `forcing_sources` and `predownload` caches it into `forcing0.zarr`.
