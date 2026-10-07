# Supporting Agentic Earth System Decision Workflows

Three changes that would most improve Earth2Studio's ability to support
agentic pipelines that run end-to-end earth system decision workflows,
based on the current state of the repository (`run.py`, `serve/`,
`lexicon/`, `statistics/`, and the `skills/` directory).

## 1. A machine-readable capability registry that works without loading weights

Today, discovering what a model consumes and produces requires
instantiating it and calling `input_coords`/`output_coords`
([earth2studio/models/px/base.py:73-84](earth2studio/models/px/base.py#L73-L84)),
which usually means downloading a checkpoint onto a GPU. Data source
coverage (variables, time range, latency, grid) lives in docstrings and
per-source lexicons under [earth2studio/lexicon/](earth2studio/lexicon/).
The `earth2studio-discover` skill compensates with prose, but planning
should not require an LLM to read source files.

**Build:** a static registry, one entry per model, data source,
diagnostic, and statistic, with: variables in/out, grid and resolution,
lead-time range and step, time tolerance, data latency and archive
window, license, hardware needs, and dependency extra. Add a
compatibility checker that answers whether source X can feed model Y and
diagnostic Z can chain onto it.

This is the difference between an agent guessing a pipeline and an agent
verifying one before spending GPU hours.

## 2. A declarative, validated, resumable workflow spec

The built-in workflows in [earth2studio/run.py](earth2studio/run.py) are
imperative Python with three fixed shapes: deterministic, diagnostic,
ensemble. The serve layer
([earth2studio/serve/server/main.py:351-366](earth2studio/serve/server/main.py#L351-L366))
already exposes named workflows with a schema endpoint, but the
workflows themselves are hand-coded. An agent's natural output is a
plan, not code.

**Build:** a serializable pipeline spec (source, prognostic, chain of
diagnostics, perturbation, statistics, IO, times, lead times) that can
be:

- statically validated against the registry from item 1, with a dry
  run that estimates memory, wall time, and data volume;
- executed identically locally or through `serve`;
- checkpointed and resumed using the existing `Checkpoint` hook;
- emitted with provenance: model versions, package hashes, data source
  and fetch time, and the spec itself embedded in the output store.

This turns the LLM into a planner and keeps execution deterministic and
auditable, which matters for anything called a decision workflow.

## 3. A decision-layer output module that returns small, structured results

Every workflow currently ends in a gridded tensor in Zarr or NetCDF.
Decisions are made on thresholds, regions, points, and events, not
arrays. The pieces exist but are disconnected: `tc_tracking` in
`models/dx`, CRPS and Brier in
[earth2studio/statistics/](earth2studio/statistics/), and observation
frame sources like GHCN and ISD.

**Build:** a post-processing layer that takes an IO backend and
produces compact, typed outputs an agent can reason over directly:

- exceedance probabilities over regions or points,
- time-of-onset and duration for thresholds,
- event tables (tracks, extremes) as GeoJSON or DataFrames,
- forecast-versus-observation verification summaries.

Each function should be small enough to be a single tool call with a
JSON return. Without this, an agent has to write ad hoc xarray code at
the end of every run, which is where most pipeline errors and
hallucinated conclusions happen.

## Runner-up: expose 1-3 as an MCP server

Rank just below the top three: expose the registry, spec runner, and
decision-layer outputs as an MCP server alongside the existing
`skills/`, so any agent framework, not just Claude Code, gets the same
discover, plan, validate, run, and summarize loop.

---

## Part 2: Coverage against the planned 1.0 work

Read against [alignment.md](alignment.md) and the three proposals it
covers ([cupy_design.md](cupy_design.md), [pipelines.md](pipelines.md),
[Impact_modeling.md](Impact_modeling.md)), plus the forward-looking
[unification.md](unification.md).

**Scoping note.** The phasing in `alignment.md` is a rough draft, not a
commitment. Priorities and ordering there are movable, so anything
required to make the agentic story credible for an end-of-November 1.0
can be pulled into scope even where it contradicts a deferral or an
ordering decision in the supporting docs. Several recommendations below
do exactly that, and say so explicitly.

## 4. Summary of coverage

| Recommendation | Covered | Gap |
| --- | --- | --- |
| 1. Capability registry | Substantially | Model entries need weights; no checker |
| 2. Workflow spec | Partially | No spec, no dry run, no provenance |
| 3. Decision outputs | Barely | Named in coupling, out of November scope |
| Runner-up. MCP | Not at all | Not mentioned in any proposal |

The pattern is consistent: the planned program builds the ingredients an
agent needs and stops short of the surfaces an agent calls. That is a
good position to be in, because the remaining work is thin layers over a
foundation rather than new design, but it does not happen by itself.

## 5. Recommendation 1 against the plan

Aligned in spirit, partially in scope.

**Already planned.** The September foundation track builds a grid
registry with coordinate reference systems, a temporal aggregation
vocabulary reconciled with the existing lexicon, and a static
data-requirement declaration on models readable without an
initialization time. Those are three of the registry's columns, and the
last one exists specifically because graph validation has to run before
any time is chosen. `unification.md` adds source *advertisement* of
schema with no live fetch, and `alignment.md` asks that advertisement be
preserved when interfaces freeze in September. Between them that is most
of the data-source half of the registry.

**Not planned.** Model entries still require instantiation. The
declaration is a method on a loaded model, so discovering what a model
consumes still means downloading a checkpoint. Nothing covers lead-time
range and step, time tolerance, data latency and archive window,
license, hardware needs, or dependency extra. No proposal plans a
compatibility checker; the closest thing is the `unification.md`
planner, which is explicitly out of the current window.

**Proposed scope change.** Add to the September freeze the requirement
that a model's declaration be readable from the class or the package
manifest without loading weights, and that requirement and advertisement
objects serialize to JSON. Both are cheap at freeze time and expensive
afterwards. Then the registry is a build step over existing
declarations rather than a parallel artifact, which is the outcome
`alignment.md` section 7 is already asking for.

### 5.1 Implementation sketch: `earth2studio.catalog`

A follow-up audit of the model tree found the registry is more buildable
than it first looks, provided entries store *variance* rather than fixed
values. 39 of 64 exported models have coordinates that depend on a
constructor argument or checkpoint contents, so a flat per-model record
is the wrong shape.

- **Entry shape.** Invariants that hold for every instance, a declared
  set of knobs with what each ranges over, and a resolver reference.
  Knobs split into three kinds: enumerable (a small literal set, e.g.
  DLESyM's `version`, StormScope's `model_name` — precompute every
  combination), checkpoint-derived (read from package metadata once per
  package version, then cached), and open/delegating (SFNO's free
  variable list, `InterpModAFNO` inheriting another model's coords —
  record the knob and the delegation, not a value).
- **Build in tiers, cheapest first.** Badge tokens and dependency extras
  are already scraped from docstrings for the docs build
  (`docs/generate_api.py`, `docs/userguide/about/install_options.yml`)
  and just need promoting into a shipped data file rather than a
  docs-only artifact. Static models (25 of 64) resolve from a bare class
  import, since `check_optional_dependencies` only raises at `__init__`.
  Checkpoint-derived entries need an instance, which the conformance
  suite's mock-weight fixtures already build with no network — reuse
  that as the generation hook, run in CI where extras are installed.
- **Real obstacles, not the configurability question.** The harder work
  is metadata trapped in control flow: cadence as a bare `% 21600`
  inside a validator, HRRR's archive window built as a closure inside
  `__init__`, units as free text in `E2STUDIO_VOCAB` parentheticals,
  lexicon `VOCAB` values in seven different shapes. None of this blocks
  starting the registry, but it is most of the effort. A drift check
  against real weights is also needed for the checkpoint-derived
  entries, since nothing today catches the registry silently disagreeing
  with a model. And it must consume the existing docs catalog and
  `__init__` export lists rather than sit beside them as a fourth
  registry, which is the exact failure mode section 7 of `alignment.md`
  already warns against.

## 6. Recommendation 2 against the plan

Aligned on execution, not on the agent-facing form.

**Already planned.** Work-identity-keyed resume that survives a
world-size change. A typed `Pipeline` constructor with no `DictConfig`
in the core signature. The serving layer composing over `Pipeline`
through a thin adapter, which is what makes local and served execution
identical. A plan preview that touches no values, and a content hash
over source identity, operation chain, grid identity, and time range.
The plan-versus-apply rule in `alignment.md` section 3.1 is the same
instinct as the dry run, arrived at from the differentiability
direction.

**Not planned.** The pipeline stays imperative Python. No serializable
end-to-end spec covering source, prognostic, diagnostic chain,
perturbation, statistics, IO, times, and lead times. No dry run that
estimates memory, wall time, or data volume. No provenance embedded in
the output store; `Impact_modeling.md` NF5 asks for graph-derived
provenance, but only for coupled runs, and it is not in the November
scope.

**Scheduling conflict, stated plainly.** `alignment.md` puts the
pipeline abstraction last, in November, and `pipelines.md` defers it
deliberately because `run_item_batched` and member batching are still
moving. Under that ordering, a spec cannot be built before December. If
a declarative spec is part of the 1.0 agentic story, the pipeline
abstraction has to move earlier, or the spec has to target the
pre-`Pipeline` substrate and be re-pointed later. The first is cleaner
and is a real reordering of the execution track, not a free addition.

**Proposed scope change.** Treat the spec as the serialization of the
`Pipeline` constructor arguments rather than a second description of the
same run. That makes it a by-product of the Phase 3 typed-argument work
instead of an independent artifact, and it keeps one description of a
run rather than two that drift. Provenance then comes nearly free: the
spec, plus package hashes and fetch times, written into the output
store.

## 7. Recommendation 3 against the plan

This is the largest gap and the one least served by existing work.

**Acknowledged but unscheduled.** `Impact_modeling.md` FR9 names a
decision-product stage as a first-class pipeline member, and FR7 names
exceedance dictionary conventions and collapse only at the product
boundary. Its sequencing puts `Application` API hardening and catalog
metadata at item 7, built last from evidence. `alignment.md` section 6
then states that item 7 does not fit by the end of November.

**Absent elsewhere.** Nothing in `cupy_design.md`, `pipelines.md`, or
`alignment.md` touches thresholds, time-of-onset and duration, event
tables, or verification summaries. `tc_tracking`, the CRPS and Brier
implementations in `earth2studio/statistics/`, and the observation frame
sources stay disconnected from each other, which is the state the
original recommendation describes.

**Proposed scope change, contradicting the November deferral.** Do not
wait for `Application` hardening. A small post-processing module that
reads an IO backend and returns compact typed results does not depend on
the component, connector, and driver core, does not depend on the
labelled container, and does not depend on the pipeline abstraction. It
can be built against today's Zarr output in parallel with all three
tracks. Scope it to four functions: exceedance probability over a region
or point, threshold onset and duration, an event table from existing
tracking output, and a forecast-versus-observation summary against a
frame source. That is a fraction of the coupling proposal's item 7 and
it is the piece an agent actually calls.

The argument for doing this now rather than after 1.0 is that it is the
only recommendation whose absence causes wrong answers rather than
merely inconvenient ones. Without it, every agent run ends in
model-authored xarray code over a gridded store.

## 8. The MCP runner-up

Not mentioned in any proposal. The only place agents appear as a
motivation is `cupy_design.md` section 2, which frames the payoff as
LLM-consumable metadata travelling with the array. That is real and
useful, but it is context for an agent reading output, not a tool
surface for an agent driving a run.

If items 1 through 3 land with JSON-serializable inputs and outputs, the
MCP server is a wrapper, not a project. If they do not, it cannot be
built at all. So the dependency runs one way and the runner-up needs no
schedule of its own, only the serialization constraints named in section
5.

## 9. Where the plans already reinforce this

Worth recording, because these were arrived at independently and should
not be traded away during the September freeze.

- The plan-versus-apply rule. Every value-touching operation having a
  resolver that returns a plan is what makes a dry run and a cache key
  possible, quite apart from its differentiability motivation.
- One declaration read by two resolvers. A model declaring its needs in
  exactly one place is what stops a registry from being a third
  restatement that drifts from the other two.
- Static advertisement split into schema, availability, and validity.
  The planner needs schema only, which is precisely the split a
  compatibility checker needs.
- The contract specification and conformance test landing first. A
  registry over components that do not agree on iterator and coordinate
  semantics would encode the disagreement.

## 10. Minimum set for a credible 1.0 agentic story

Ordered by ratio of agentic value to cost, not by dependency.

1. Serialization and no-weights constraints on the September foundation
   interfaces. Near zero cost at freeze time.
2. The decision-layer module, four functions, built in parallel against
   today's output. Independent of all three tracks.
3. A generated registry over the frozen declarations, plus a
   compatibility checker. Small once item 1 holds.
4. Spec serialization of the `Pipeline` arguments, with provenance
   written to the output store. Requires moving the pipeline abstraction
   earlier in the execution track, which is the one genuine schedule
   change in this list.

Items 1 through 3 fit comfortably before the end of November. Item 4 is
the one that needs a decision, and it is a decision about the execution
track's ordering rather than about total effort.
