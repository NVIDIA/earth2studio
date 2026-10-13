# Flexible-domain models

Supporting notes for the optional `DomainAdaptable` protocol and `Region` type. The
contract itself, including rules `G1`–`G5`, lives in Domain selection in
[MODEL_CONTRACT_SPEC.md](../spec/MODEL_CONTRACT_SPEC.md). Fixed-domain wrappers need
no changes.

## Motivation

| Wrapper | Current configuration |
| --- | --- |
| `StormCastCONUS` | HRRR index slices in the constructor; cropping re-registers `pos_embed` |
| `CorrDiffCosmoEra5` | `set_domain(lat_min, lat_max, lon_min, lon_max, margin_deg, halo)` view |
| `cBottleSR` | `set_super_resolution_window` changes the instance in place |

- `StormCastCONUS`: a different crop requires reloading weights or changing shared
  network state.
- `CorrDiffCosmoEra5`: `set_domain` returns a new instance sharing networks, but its
  arguments are geographic bounds, and a view cannot select another part of the full
  footprint.
- `cBottleSR`: previously captured coordinate signatures become stale.

A future model may place a patch-local Lambert azimuthal equal-area (LAEA) grid
anywhere on the globe, with a chosen orientation. Geographic bounds alone cannot
express that placement: the same bounds can be covered by differently oriented
grids. The selection request should describe either an area to cover or a local
frame to honor.

## Region representation

Keep the original tagged representation rather than adding public classes for
each shape. The constructors validate and normalize it, so callers need only
know the classmethods. Model authors can inspect the documented fields.

Direct construction must enforce the same invariants as the classmethods (or
be disabled). In particular, polygon edge interpretation needs the separate
`edges` field: a tuple of floats alone cannot encode it. This is an internal
storage detail for most callers, but a stable, documented representation for
models implementing `domain_for`.

A full-longitude request uses `band` rather than `box`: a ring between two
latitudes has two boundaries, so no single-ring `polygon` can describe it, and
`around` has no single center for it. cBottleSR windows spanning every longitude
are bands. `from_track(points, radius_m)` can later be a convenience
constructor that produces a region covering the track. It does not need a separate
region kind.

## Implementation guidance

Views should reuse loaded network weights and keep domain-dependent invariants,
coordinates, and caches separate. The protocol specifies observable behavior,
not a required shared-core class or cache layout. In particular:

- StormCast must not re-register a cropped positional-embedding parameter on
  a network shared with other views. Keep the full embedding and select the
  needed portion during execution.
- Networks that store shape metadata must bind it for every forward, including
  an A, B, A sequence of views. Shared caches must include every input that
  changes their result; a grid `fingerprint()` alone does not account for halo,
  source grid, device, or other settings.
- A view's stochastic stream must not be shared accidentally with another
  view. To check reproducibility, seed before each A rollout, run A, B, A, and
  compare the two A results; also check that the root `state_dict` is unchanged.
- Moving a network shared by views can leave sibling view buffers on an old
  device. Wrappers must keep those buffers consistent when supporting `.to()`.
  Shape-varying views may also cause recompilation under `torch.compile`.

A covering selection helper may be useful for crop models, but is not part of the
model protocol.

## Wrapper migration

- `CorrDiffCosmoEra5`: add `domain_for` and `with_domain`; keep a reference to the
  full footprint in views and adapt the current `set_domain` API. Treat its run grid
  and trimmed output grid separately.
- `StormCastCONUS` / `StormCast`: share the full weights; select crops without
  replacing network parameters. Keep existing constructor limits as an adapter.
- `cBottleSR`: return an independently configured view from `with_domain`; retain
  the in-place setter for compatibility while discouraging its use after signatures
  have been captured.
- Future globe-wide patch models: build a local grid from `Region.around`, including
  its orientation, and validate explicit grids in `with_domain`.

The existing `ProjectedGrid` represents 1-D projected axes under one CRS. An
arbitrarily rotated LAEA patch may need another `GridDefinition` if its frame
cannot be expressed by that CRS; this does not change the model protocol.
