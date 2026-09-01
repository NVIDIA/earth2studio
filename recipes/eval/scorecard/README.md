# Scorecard

Generates the source score JSON files behind the documentation's
[Scorecards pages](../../../docs/scorecard). Every model runs the same
campaign: 48 initial conditions over a 14-day horizon, with ERA5
verification through ARCO_ERA5. The four monthly initial conditions rotate
through the 00/06/12/18 UTC synoptic hours, so no lead time always
verifies at the same local time.
The campaigns use the evaluation recipe's **online scoring**: the inference
loop reduces every forecast to a small set of statistics and never writes
the raw forecast store.

The scorecard computes every metric globally and for the regional splits in
`cfg/regions/standard.yaml`. Any named lat/lon boxes work; that file is
just the scorecard's choice. The exports also carry seasonal, monthly, and
per-init-hour breakdowns, a per-initial-condition skill grid, and the
persistence and climatology baseline campaigns.

## Layout

```text
scorecard/
  cfg/campaign/<model>_2025_scorecard.yaml   self-contained evaluation campaigns
  cfg/regions/standard.yaml     the regional splits every campaign shares
  run_scorecard.py        predownload -> infer (scores online) -> score/prune
  export_scores.py        scores.zarr -> exports/eval_scores_<model>*.json
  utils/pipelines.py      history / off-grid pipeline variants (see below)
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
on Hugging Face; files already present under `docs/_static/scorecard/`
(for example, fresh local exports from step 2) take precedence, so the
build works offline while iterating. To publish updated exports:

```bash
hf upload nvidia/earth2studio-assets docs/_static/scorecard scorecard \
  --repo-type dataset --include "eval_scores_*.json"
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

## Model-specific pipelines (`utils/pipelines.py`)

* `RegriddedForecastPipeline` — models off ERA5's 721x1440 grid are gathered
  onto it so the shared verification store is reused. A no-op on-grid.
