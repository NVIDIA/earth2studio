<!-- markdownlint-disable MD013 -->
# DSX Exchange connector

This recipe runs AI weather models with Earth2Studio and publishes forecasts for configured sites
to NVIDIA **DSX Exchange** over MQTT. The published forecasts can be consumed by building management
systems (BMS), for example as an input to data-center cooling optimization.

The recipe includes two example workflows:

| `workflow:` | Model | Coverage | Horizon / step | Initial condition | Publishes | Config |
| --- | --- | --- | --- | --- | --- | --- |
| `stormcast-conus` *(default)* | StormCast CONUS | U.S. (CONUS), ~3 km | 24 h / 1 h | hourly HRRR + AI conditioning | Temperature, RelativeHumidity, WetBulb, WindU, WindV | `cfg/dsx-connector.stormcast.yaml` |
| `sfno` | SFNO | Global | 14 d / 6 h | 6-hourly GFS | Temperature, WindU, WindV | `cfg/dsx-connector.sfno.yaml` |

## How it works

A workflow can combine compatible Earth2Studio data sources, models, and processing steps.

```text
workflow -> collector -> ForecastSeries -> validated DSX messages -> MQTT
```

1. **Workflow** (`src/<workflow>/workflow.py`) loads input data and runs the model-specific forecast
   process.
2. **Collector** (`src/<workflow>/collector.py`) extracts values for the configured sites and applies
   workflow-specific transformations. For example, StormCast rotates winds and calculates relative
   humidity and wet-bulb temperature using `src/shared/wind_rotation.py` and
   `src/shared/derived_variables.py`.
3. **ForecastSeries** (`src/shared/forecast.py`) provides the shared internal format.
4. **DSX messages** (`src/dsx/contract_adapter.py`, `src/dsx/coordinator.py`, `src/dsx/schema.py`) are
   created and validated against the contract.
5. **MQTT** (`src/dsx/publisher.py`, `src/dsx/publish_loop.py`, `src/dsx/bus.py`) publishes the
   validated messages to DSX Exchange.

A cycle is one complete forecast run for a single model initialization time. The included StormCast
and SFNO workflows support `run.init_time: latest`, which selects the newest available analysis, or
a fixed initialization time. Custom workflows own their scheduling behavior and may use different
initialization options. Each cycle publishes one `Forecast` bundle per configured site **and
variable**; `Metadata` for each is published at startup and on every heartbeat. The `workflow:`
setting selects which workflow a process runs. To run both workflows, start two processes with
different config files (see *Running two workflows*).
For the default StormCast workflow, install the tested plain-venv GPU stack described in
[`docs/stormcast-environment.md`](docs/stormcast-environment.md).

```bash
python main.py                                        # stormcast-conus (default config)
python main.py --config cfg/dsx-connector.sfno.yaml   # sfno global medium-range
```

## Prerequisites

Common to both workflows (per-workflow specifics are under *Workflows*).

### Software

- Install **Earth2Studio** and the dependencies for the workflow you want to run. StormCast also
  requires a global conditioning model (SFNO by default). GPU package versions must match your CUDA
  environment; for StormCast, follow the tested installation steps in
  [`docs/stormcast-environment.md`](docs/stormcast-environment.md).

### Hardware

- **GPU:** A CUDA GPU is required; CPU is unsupported. StormCast with SFNO conditioning is validated
  on a 48 GB GPU. Standalone SFNO uses less memory.
- **Network:** Outbound access to NOAA, model registries, and the DSX MQTT broker. No inbound access
  is required.
- **Disk:** Use disk-backed storage for the Earth2Studio cache and, for StormCast,
  `conditioning.scratch_dir`.

## Quick start

Install the StormCast recipe environment (see
[`docs/stormcast-environment.md`](docs/stormcast-environment.md) for the GPU stack), then run from
this directory. By default, Earth2Studio downloads and caches the required model weights on first
run. To use pre-staged weights for the workflow's main model, set `model.path`. The connector still
requests HRRR/GFS weather inputs at runtime.

The examples below use the default `stormcast-conus` config; add `--config cfg/dsx-connector.sfno.yaml`
for the SFNO workflow.

1. **Dry-run** — validates config + contract, loads the model(s), runs one cycle, and prints the
   `Forecast`/`Metadata` bundles per site. No broker needed. The recommended first check:

   ```bash
   python main.py --dry-run
   ```

   Model loading and inference may take several minutes. The abbreviated output looks like:

   ```text
   [RETAINED] Weather/v1/PUB/Metadata/conus-site-weather/dc-site-1/Temperature
     {"unit":"K", ...}
   [live    ] Weather/v1/PUB/Forecast/conus-site-weather/dc-site-1/Temperature
     {"initTime":..., "leadSeconds":[...], "values":[...], "memberCount":1, ...}
   ```

   Dry-run prints one Metadata and Forecast message per site and variable.

2. **Create a local configuration.** Copy the config for the workflow you want to run:

   ```bash
   # StormCast
   cp cfg/dsx-connector.stormcast.yaml cfg/local.yaml

   # Or SFNO
   cp cfg/dsx-connector.sfno.yaml cfg/local.yaml
   ```

   In `cfg/local.yaml`, set:

   - `bus.broker_host`: hostname or address of the DSX MQTT broker, for example
     `"dsx-broker.example.com"`.
   - `sites`: one or more locations, each with an `id`, `lat`, and `lon`, for example
     `{id: dc-omaha-1, lat: 41.26, lon: -95.94}`.
   - `conditioning.scratch_dir` (StormCast only): a writable, disk-backed directory for the
     temporary conditioning file, for example `"/var/lib/dsx-connector/conditioning"`.

   The config comments describe the remaining options. Do not edit the shipped configs directly.
   Then validate your config:

   ```bash
   python main.py --dry-run --config cfg/local.yaml
   ```

3. **Publish one cycle then exit** (`--once`) — confirms broker connectivity + auth:

   ```bash
   python main.py --once --config cfg/local.yaml
   ```

4. **Run as a persistent service** — a forecast each cycle plus a heartbeat that republishes
   Metadata and the latest forecasts. The connector exits on unrecoverable bus loss or repeated
   cycle failures, so run it under a supervisor / restart policy:

   ```bash
   python main.py --config cfg/local.yaml
   ```

Three modes: `--dry-run` (validate + print, no bus), `--once` (one cycle then exit), and no flag
(persistent loop).

Each workflow provides the input needed to initialize its model. The included
StormCast and SFNO workflows use weather analyses: gridded estimates of the atmospheric state at a
specific time. With `run.init_time: latest`, the connector selects the newest available analysis at
startup. After publishing that forecast, it waits until an analysis with a later initialization time
becomes available. The heartbeat republishes the latest forecast, and a forecast may also be sent
again after a process restart, so consumers must handle duplicates (see *Reading the data*).

## Testing

The test suite mocks the model and GPU integrations, so it runs without Torch, CUDA, or model
weights. In a clean Python environment:

```bash
python -m pip install -r requirements-test.txt
python -m pytest test/
```

Tests are grouped by ownership under `test/dsx/`, `test/stormcast/`, `test/sfno/`, and
`test/shared/`.

The MQTT broker integration tests are skipped when no test broker is available.

## Configuration

All settings live in one YAML file (`--config` selects it; default
`cfg/dsx-connector.stormcast.yaml`). Each workflow has one copyable configuration:
`cfg/dsx-connector.stormcast.yaml` for StormCast and `cfg/dsx-connector.sfno.yaml` for SFNO. Both
workflows use these sections:

- **`workflow`** selects a registered forecast workflow. Built-in options are `stormcast-conus` and
  `sfno`; custom workflows can be added as described in *Adding a workflow*.
- **`sites`** lists the forecast locations.
  - `id`: unique site identifier used in MQTT topics, such as `dc-site-1`. The contract allows
    lowercase letters, digits, and single hyphens, up to 63 characters. Duplicate IDs are rejected.
  - `lat`, `lon`: geographic coordinates in degrees.
- **`model`** configures the primary forecast model.
  - `id`: model identifier included in published messages.
  - `path`: optional path to pre-staged model weights, such as `/ckpt`.
  - `device`: execution device, such as `cuda:0`.
- **`run`** controls forecast scheduling.
  - `init_time`: `latest` or a fixed ISO timestamp, such as `"2026-07-05T14:00:00"`.
  - `nsteps`: number of model forecast steps. Step length depends on the workflow: StormCast uses
    1-hour steps (`24` = 24 hours), while SFNO uses 6-hour steps (`56` = 14 days).
  - `poll_interval_seconds`: how often persistent mode checks for new input, such as `600`.
  - `max_consecutive_failures`: number of consecutive failures allowed before the process exits.
  - `max_lookback_hours` (SFNO): how far back to search for available GFS input.
  - `cache_retention_hours`: age at which cached HRRR/GFS input files may be removed after a
    successful cycle. The default is `48`; use `0` to disable cleanup.
- **`bus`** configures the MQTT connection.
  - `broker_host`: broker hostname, such as `dsx-broker.example.com`.
  - `broker_port`: broker TCP port, such as `1883`.
  - `auth`: `noauth` for local testing or `oauth2` for a secured broker.
  - `qos`: MQTT Quality of Service, default `0` as specified by the contract. `0` sends without
    broker confirmation, so a message may be lost; the heartbeat republishes Metadata and the
    latest forecasts to recover from loss. `1` waits up to `bus.publish_timeout` (default 10 s)
    for the broker to acknowledge each message.
  - `client_id`: unique identifier for this connector process, such as
    `"dsx-connector-stormcast"`.
  - `heartbeat_seconds`: interval for republishing Metadata and the latest forecasts (default 90,
    at most 100).
  - `tls`: enables TLS; required when `auth` is `oauth2`.
  - `token_file`: path to a file containing the OAuth token.
  - `ca_certs`: optional path to the trusted CA certificate bundle.
- **`topics`** configures the MQTT namespace.
  - `forecast_prefix`: topic prefix; normally `Weather/v1/PUB`.
  - `product`: name used in MQTT topics so consumers can choose which forecasts to receive, such as
    `"conus-site-weather"`. Like site IDs, it may contain lowercase letters, digits, and single
    hyphens, up to 63 characters. Ask the DSX Exchange administrator which name to use.
- **`inputs`** describes the workflow's data sources in published Metadata; it does not configure
  data loading. Leave it unchanged for the included workflows. A custom workflow should list its
  actual sources and roles, such as `{source: HRRR, role: initialCondition}`.

Each included workflow has additional settings. StormCast uses `conditioning`, `subregion`, and
`performance`; SFNO uses `ensemble`. These settings are explained in the workflow sections below.

`topics.forecast_prefix` sets the beginning of every MQTT topic. Keep the default `Weather/v1/PUB`
unless the DSX Exchange administrator provides another value. Changing it also requires updating consumer
subscriptions and broker permissions.

## Workflows

The sections below describe the included StormCast and SFNO examples. Custom workflows can also be
added; see [Adding a workflow](#adding-a-workflow).

### StormCast CONUS (`stormcast-conus`, default)

```text
GFS analysis -> global model -> hourly conditioning --+
                                                       +-> StormCast -> collector -> ForecastSeries
Hourly HRRR analysis ---------------------------------+
```

StormCast produces forecasts at about 3 km resolution over the continental U.S. Because its regional
domain does not include weather outside that area, it also uses a forecast from a global model. This
additional model input is called *conditioning*.

Many global models could provide conditioning. This recipe currently supports **SFNO** (the default,
tested option) and **FCN3** (implemented but not yet tested end to end). The conditioning model starts
from a 6-hourly GFS analysis. If the expected GFS cycle is not available yet, the connector tries an
older cycle within the configured lookback period.

Each StormCast forecast starts from the latest hourly **HRRR** analysis and uses the global
forecast as conditioning. This gives StormCast an hourly starting state while retaining global
weather context. StormCast with SFNO conditioning is validated on a single 48 GB GPU; FCN3 requires
more GPU memory and is not validated here.

StormCast-specific config sections:

- **`conditioning`** configures the global conditioning model.
  - `model`: `sfno` or `fcn3`.
  - `device`: execution device, such as `cuda:0`.
  - `max_lookback_hours`: how far back to search for an available GFS analysis to initialize the
    conditioning model.
  - `scratch_dir`: disk-backed directory for the multi-GB conditioning file.
- **`subregion`** optionally limits StormCast inference to part of the CONUS grid.
  - `enabled`: enables or disables cropping; disabled by default.
  - `center`: optional latitude and longitude at the center of the crop.
  - `size_cells`: crop height and width in grid cells.

  Cropping is experimental and has model-specific size, alignment, and northern-edge restrictions.
  See [StormCast CONUS subregion cropping](docs/stormcast-subregion.md) before enabling it.
- **`performance`** controls the trade-off between StormCast inference speed and potential forecast
  quality.
  - `num_diffusion_steps`: lower values run faster but may reduce forecast quality.

### SFNO global medium-range (`sfno`)

```text
GFS analysis -> optional ensemble perturbation -> SFNO -> collector -> ForecastSeries
                (only when members > 1)
```

```bash
python main.py --config cfg/dsx-connector.sfno.yaml
```

SFNO is a global model initialized directly from a 6-hourly GFS analysis. A 56-step rollout produces
57 six-hourly leads per site, from lead 0 through 14 days. SFNO's weights are pulled from **NGC** on
the first run (a few GB, cached thereafter) unless `model.path` points at a local checkpoint.

#### SFNO ensemble demonstration (optional)

The SFNO workflow includes a ready-to-run initial-condition ensemble that demonstrates Earth2Studio
ensemble execution and DSX summary publishing. It is **off by default**: omit the `ensemble:`
section, or set `members: 1`, and the run stays deterministic (the *values* are unchanged — every
payload just also carries `memberCount: 1`). Set `members > 1` to run `earth2studio.run.ensemble`:

```yaml
ensemble:
  members: 8              # ensemble size (> 1 enables the ensemble)
  batch_size: 2           # members per GPU batch (VRAM vs throughput); defaults to members
  # seed: 0               # optional, for reproducible perturbations
  include_members: false  # also publish raw per-member curves (payload grows linearly with members)
```

For this demonstration, the recipe adds spatially correlated noise only to `z500` (500 hPa
geopotential) before running the ensemble. The noise amplitude, `39.27`, comes from an upstream HENS
example and is an uncalibrated starting point for SFNO. Other input variables are unchanged because
they use different physical units and require separately chosen and validated perturbations.

For each forecast time, an ensemble message includes:

- `values`: ensemble mean.
- `standardDeviation`: population standard deviation across members.
- `minimum` and `maximum`: lowest and highest member values.
- `percentiles`: fixed ensemble percentiles `p10`, `p50`, and `p90`. They are not configurable in
  YAML. To change the published set, update `_PERCENTILES` in `src/dsx/contract_adapter.py` and the
  corresponding tests. The contract already accepts keys from `p0` through `p100`; coordinate a
  changed set with consumers.
- `memberCount`: number of ensemble members.
- `members`: optional raw forecast curve for each ensemble member when `include_members: true`.

If any member is missing or invalid at a forecast time, all summary values for that time are `null`;
the connector never calculates statistics from only part of the ensemble.

Compute cost grows approximately with the number of members. `batch_size` controls how many members
run together: larger batches may run faster but use more GPU memory.

### Running two workflows

To run both included workflows at the same time, launch each in a separate process. Configure each
process with:

- a unique `bus.client_id`;
- an explicit GPU assignment; and
- a separate data cache when automatic cleanup is enabled. Set `EARTH2STUDIO_DATA_CACHE` to a
  different directory for each process; do not clean a cache used by another active process.

StormCast also needs its own `conditioning.scratch_dir`.

If both workflows publish the same site and variable, give them different `product` ids. A product
identifies one forecast stream and determines its topic namespace and DSX access rule. The included
configs already use different ids: `conus-site-weather` and `global-medium-range-weather`.

### Adding a workflow

To add another model or forecast process:

1. Create `src/<name>/workflow.py` with a `run(cfg, args, stop)` function. This function loads the
   model and input data, runs the forecast, and passes the results to the shared publishing code.
2. Add model-specific collection and variable definitions in the same `src/<name>/` package.
3. Register the workflow name in `_WORKFLOWS` in `main.py`.
4. Add a matching `cfg/dsx-connector.<name>.yaml` configuration file.

Use `src/dsx/` for DSX publishing and `src/shared/` for reusable helpers such as site extraction,
lead-time conversion, and the `ForecastSeries` output format. This keeps model-specific code
separate and avoids implementing the publishing path again.

The new workflow validates its own settings. Put scheduling and retry settings under `run`, and put
model-specific settings in their own section. New workflow settings do not require changes in
`src/dsx/`.

#### Repository layout

```text
main.py                    workflow dispatcher
src/dsx/                   model-independent DSX publishing core
src/stormcast/             StormCast workflow, collector, conditioning, variables
src/sfno/                  SFNO workflow, collector, variables
src/shared/                model-independent workflow helpers
  forecast.py              producer-to-DSX handoff format
  site_extraction.py       curvilinear and regular-grid site extraction
  lead_time.py             lead-time conversion
  derived_variables.py     relative humidity and wet-bulb calculations
  wind_rotation.py         Lambert-Conformal wind rotation
  cycle_availability.py    latest-available-cycle scheduling
  data_cache.py            shared input-cache cleanup
cfg/                       workflow configuration
data/weather.yaml          vendored DSX weather contract
test/                      torch-free test suite
```

`dsx/` never imports a producer, the producers never import each other, and `shared/` never imports
`dsx/` or a producer. Only each producer's `workflow.py` plus `stormcast/conditioning.py` import
Earth2Studio or Torch; the publishing path and reusable helpers remain GPU-independent.

## What it publishes

For each configured site and weather variable, the connector publishes two DSX message types:
Metadata describing the forecast and a Forecast bundle containing its values over time. Metadata
is also sent as a retained message, so consumers that connect later usually receive it at once.
Forecast bundles are not retained; instead, the connector republishes the latest bundle for each
topic at every heartbeat, so a consumer that connects later receives it within one heartbeat.

The `stormcast-conus` workflow publishes all five variables below. The `sfno` workflow publishes
Temperature, WindU, and WindV:

| Variable | Unit | SFNO | Notes |
| --- | --- | --- | --- |
| Temperature | K | ✓ | 2 m |
| RelativeHumidity | percent | | 2 m temp; humidity/pressure from ~8 m (lowest level), small bias |
| WetBulb | K | | from temperature, humidity and local pressure; the evaporative-cooling target |
| WindU / WindV | m s-1 | ✓ | **10 m**, earth-relative (true east / north) |

In MQTT, a topic is the address used to publish and subscribe to messages, similar to a channel
name. Each topic contains a `{product}` id that identifies the forecast stream a consumer subscribes
to. The producing model is not part of the topic. Instead, both message types contain a `model`
field, such as `"model": "stormcast-conus"`. Its value comes from `model.id` in the configuration
and is visible in `--dry-run` output and in messages received from DSX.

The connector uses these topics:

- **Forecast** (not retained): `Weather/v1/PUB/Forecast/{product}/{site}/{variable}` — one bundle per
  site/variable carrying the whole lead-time curve.
- **Metadata** (retained): `Weather/v1/PUB/Metadata/{product}/{site}/{variable}` — units, location,
  model.

In persistent mode, both are republished every `bus.heartbeat_seconds` (default 90 s). A
republished Forecast is byte-for-byte identical to the original, including `publicationTime`, until
the next cycle replaces it. `--once`/`--dry-run` publish each message once.

**Why the heartbeat:** the contract uses QoS 0, so any message can be lost, and a retained copy is
only an optimization for late subscribers. Periodic republishing recovers lost or missing messages.
The recipe caps `heartbeat_seconds` at 100 to match the 100 s cadence of the BMS contract.

For example, the complete topic
`Weather/v1/PUB/Forecast/conus-site-weather/dc-omaha-1/Temperature` contains:

- product: `conus-site-weather`;
- site: `dc-omaha-1`; and
- variable: `Temperature`.

### Reading the data (for consumers)

This guidance is for applications that subscribe to and use the published messages.

#### Interpreting messages

- **Read Metadata first.** It provides the unit, location, model, and expected forecast cadence for
  each site and variable.
- **Match each value to its forecast time by position.** For example, `leadSeconds: [0, 3600]` and
  `values: [280.0, 281.5]` mean 280.0 at the model initialization time and 281.5 one hour later.
  `initTime` is a Unix timestamp in milliseconds, while `leadSeconds` is in seconds, so calculate a
  value's time as `initTime + leadSeconds × 1000`.
- **`null` means missing.** Do not interpret it as zero.
- **Use `memberCount` to distinguish deterministic and ensemble forecasts.**
  - `memberCount: 1` means deterministic. `values` contains the single forecast curve, and ensemble
    summary fields are absent.
  - A value greater than 1 means ensemble. `values` contains the ensemble mean. Additional fields
    contain `standardDeviation`, `minimum`, `maximum`, and the `p10`, `p50`, and `p90` percentiles.
  - Each summary array follows the same positions as `leadSeconds`.
  - If any member is invalid at a forecast time, every summary value at that time is `null`.
  - Raw member curves are included under `members` only when the producer enables them.
- **Cadence is nominal, not guaranteed.** The retained Metadata's `issueCadenceSeconds` is the
  *nominal* product cycle (hourly for StormCast, 6-hourly for SFNO), not a promise of publication
  spacing: a new Forecast appears only when a newer analysis lands and a cycle completes, and a cycle
  may be late or skipped. `publicationTime` shows when a message was sent, while `initTime` identifies
  its model cycle. Compare these fields across messages to see the actual delivery timing and whether
  any model cycles were skipped.

#### Consumer safeguards

- **Check forecast age using `initTime`.** `publicationTime` records when a bundle was first
  published; heartbeat republishes keep it, but a restart can publish the same `initTime` again with
  a new value. Reject forecasts whose initialization or valid times are too old for your use case.
- **Handle duplicate forecasts.** The heartbeat repeats the latest Forecast, and a restart can
  publish the same `initTime` again. Identify a forecast by `product`, `site`, `variable`, and
  `initTime`, and use the one with the latest `initTime`. If a repeat contains the same values,
  ignore it. If the values differ, the contract does not indicate which version is newer, so define
  a policy, such as keeping the most recently received message.
- **Assemble a full cycle before acting.** Variables are sent as separate messages and can arrive in
  any order. Group them by `site` and `initTime`, wait for the variables you need, and never combine
  different initialization times. Use a timeout and safe fallback for an incomplete cycle.
- **Allowlist expected inputs.** Accept only the products and variables your application supports,
  verify their units and `standardName` values from Metadata, and ignore unknown inputs.
- **Use safe fallbacks.** Define how the application handles missing values, stale or incomplete
  forecasts, and communication failures. Treat forecasts as safeguarded inputs rather than acting on
  them without checks.

Every message is validated against the vendored DSX weather **AsyncAPI** contract (`data/weather.yaml`)
before publish. The contract is **early / pre-stable (v0.1.0)** and may change.

## Runtime and maintenance

- **Startup:** the connector connects to the broker first (fails fast if unreachable), then loads the
  workflow's model(s) — a few minutes on first run, including the weight download. The StormCast
  workflow additionally builds its conditioning model during startup, **before** entering the cycle
  loop (`building sfno conditioning model …`).
- **Per cycle:** a start line when a cycle begins (`stormcast: …` / `sfno: …`), then `cycle published
  (initTime=…)` once every bundle for that init has been accepted by the local MQTT client for
  sending (with `qos: 1`, acknowledged by the broker).
- **Automatic cache cleanup:** after a successful cycle, each workflow deletes its old input files
  (GFS for SFNO; HRRR and GFS for StormCast). Files older than `run.cache_retention_hours` are removed;
  the default is 48 hours. Cleanup also runs for fixed initialization times and one-shot runs. Files
  are aged by modification time, so an old analysis downloaded today remains available for immediate
  reruns. The effective retention period is never shorter than the input lookback plus one six-hour
  GFS cycle. Set the value to `0` to disable cleanup. Model weights are never removed.
- **Shutdown:** SIGTERM or SIGINT (Ctrl+C) asks the connector to stop safely. If a model is loading or
  a forecast is running, that work finishes before the connector exits, so shutdown may take several
  minutes. Messages still waiting in memory are not forced onto the bus. After a restart, the
  connector selects its input again and may publish the same `initTime` another time. Consumers
  should therefore handle duplicate forecasts.

## Authentication

For an unsecured test broker use `auth: noauth`. For a secured broker set `auth: oauth2` and `tls:
true`: the connector connects with MQTT username `oauthtoken` and a bearer token as the password, over
TLS. The token is read from `bus.token_file` (a mounted secret, preferred) or the `DSX_OAUTH_TOKEN`
environment variable, and re-read before every (re)connect (so a rotated file is picked up on the next
reconnect, not mid-session).

The connector only reads an access token; it does not obtain or refresh one. The DSX Exchange
administrator must provide:

- a process for obtaining and rotating a JWT access token;
- a token accepted by the broker, with the required issuer, audience, expiry, identity, and scope;
- permission to publish only to the configured product, for example
  `pub.allow: ["Weather.v1.PUB.*.<product>.>"]` (the `*` spans the `Forecast`/`Metadata`
  message kind that precedes `<product>` in the topic); and
- the TLS MQTT endpoint and its CA certificate, if it uses a private certificate authority.

The exact token claims and publish permissions depend on the deployment. Confirm them with the DSX
Exchange administrator before going live, mount the token as a secret file, and never commit it.

Running more than one connector against the same broker? Give each a unique `bus.client_id` (a shared
id makes the broker disconnect the older session).

## Limitations (pre-stable)

- **Messages can be lost or repeated (QoS 0)** — the heartbeat republishes Metadata and the latest
  forecasts; see *Reading the data*.
- **The DSX `{site}` convention is provisional** — agree on a durable site id with the DSX team before
  consumers build against it.
- **Fixed-cadence models only.** Each producer has a single, uniform native timestep, owned by the
  model (StormCast hourly, SFNO 6-hourly), so `horizonSeconds = nsteps × step`. The wire itself
  (integer `leadSeconds` arrays) supports sub-hourly and irregular spacing, but the recipe does not yet
  *produce* irregular cadences — a variable-step producer would need its own horizon/lead logic.
- **The weather contract is early / pre-stable (v0.1.0)** and may change (including breaking changes).

## References

- [Earth2Studio recipes guide](https://nvidia.github.io/earth2studio/userguide/developer/recipes.html)
- [`docs/stormcast-environment.md`](docs/stormcast-environment.md) — StormCast GPU stack install
- [`docs/stormcast-subregion.md`](docs/stormcast-subregion.md) — StormCast subregion configuration
- `data/weather.yaml` — the vendored DSX weather AsyncAPI contract
