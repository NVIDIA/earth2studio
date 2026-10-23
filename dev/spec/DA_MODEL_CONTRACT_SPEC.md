<!-- markdownlint-disable MD013 -->
# Data assimilation model contract

- `AssimilationModel`: `__call__`, `input_coords`, `forcing_coords`, `output_coords`, `default_sources`, `to`, `stochastic`, `set_rng(seed, reset=True)`.
- `StatefulAssimilationModel` extends it: `initialize(*inputs) -> (y, state)`, `step(*outputs, *forcing, state) -> (y, state)`, `create_iterator(*inputs)`.
- Concrete signatures use named slots: backgrounds before forcing; step state last, positional-or-keyword.
- Slots: DataArray/CoordinateSystem or DataFrame/FrameSchema; optional observations accept `None`; multiple slots preserve tuple order.
- Initialization computes the first analysis; iteration yields it first, then receives dynamic forcing via `send`. Required forcing is never reused implicitly.
- Outputs plus serializable state fully determine continuation. Steps replay exactly without mutating inputs/state; execution hooks are iterator-only.
- RNG: calls/new trajectories advance a local stream; reset restarts it, `reset=False` initializes only if absent. Seeded execution preserves global RNG; reseeding leaves live trajectories unaffected. Deterministic models use a no-op.
- Sources follow input-then-forcing order; callers fetch and align data. Models document observation times/windows.
- Independent: HealDA, interpolation (deterministic), CorrDiff (stochastic). Stateful: StormCast (stochastic).
- Migration: remove `create_generator`; use independent calls or forecast-first `create_iterator`. Replace CorrDiff `.seed` with `set_rng`.
