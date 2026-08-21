# Scorecard

Generates the source score JSON files behind the documentation's
[Scorecards pages](../../../docs/scorecard). Each model is evaluated on the
same campaign — 24 initial conditions (1st and 15th of each month of 2025,
00 UTC), 14-day horizon, ERA5 verification via ARCO — using the eval recipe.

## Layout

```text
scorecard/
  cfg/campaign/<model>_2025_scorecard.yaml   self-contained evaluation campaigns
  run_scorecard.py        predownload -> infer -> score -> prune
  export_scores.py        scores.zarr -> exports/eval_scores_<model>.json
  utils/pipelines.py      history / off-grid pipeline variants (see below)
  models/<model>/outputs/ run data: forecast.zarr -> scores.zarr (not tracked)
  data/                   shared ERA5 stores (not tracked)
```

Note: Some of the following folders will be generated after running the scorecard recipe.

## Usage

On a GPU node with the recipe environment:

```bash
# 1. Run a campaign from cfg/campaign/
python run_scorecard.py fcn3_2025_scorecard

# 2. Export the scores JSON and copy it into the docs
python export_scores.py fcn3 --docs

# 3. Regenerate the docs pages
python ../../../docs/generate_scorecard.py
```

## Scaling and portability

Launches default to a single node (`torchrun --standalone`, all local GPUs);
for multi-node runs set `TORCHRUN_ARGS` with your rendezvous flags, or invoke
the recipe entry points (`predownload.py`, `main.py`, `score.py`) with your
own launcher — the driver adds nothing they don't already support. All stages
are resumable (`resume: true` in the campaigns), so large campaigns can be
advanced incrementally across short queue allocations rather than needing one
long job.

The JSON export carries everything a docs page needs — metric curves per variable
and lead time (aggregated over initial conditions), units, and variable groups.

Multiple campaigns per model are supported by construction: a campaign is just
another self-contained config, and the docs plot selects its data file by URL
key. Only the default file naming (`eval_scores_<model>.json`) assumes one
campaign per model — exporting a second campaign under a distinct name (and
listing it in `mkdocs.yml`) is the whole extension.

## Model-specific pipelines (`utils/pipelines.py`)

* `RegriddedForecastPipeline` — models off ERA5's 721x1440 grid are gathered
  onto it so the shared verification store is reused. A no-op on-grid.
