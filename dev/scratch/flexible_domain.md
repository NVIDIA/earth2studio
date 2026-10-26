# Flexible-domain models

Supporting notes for the optional `DomainAdaptable` protocol and `Region` type. The
contract itself, including rules `G1`–`G4`, lives in Domain selection in
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

## Design decisions

- **One method.** `with_region(region)` replaced an earlier `domain_for(region)` /
  `with_domain(grid)` pair. A single grid intermediary does not fit multi-grid models
  whose slots use different grids; the model now chooses every slot's grid itself.
- **Region, not grid-specific arguments.** Each model's grids still own selection
  (`subset_indexers`, snapping, local projection). `Region` is only the request, so a
  driver can configure models on different grids, or a model with no parent grid,
  without model-specific code.
- **Tagged representation.** Keep `kind` plus `params` rather than public classes
  for each shape. The constructors validate and normalize it, so callers need only
  know the classmethods; model authors can inspect the documented fields. Direct
  construction enforces the same invariants.
- **Full longitude.** A full-longitude request uses `band` rather than `box`. A ring
  between two latitudes has two boundaries and no single center, so neither a polygon
  nor a `rectangle` can describe it. cBottleSR windows spanning every longitude are
  bands.
- **Deferred kinds.** Polygons were dropped: interior/exterior resolution, concave or
  self-intersecting rings and canonical vertex order add complexity no current model
  needs. A `from_track(points, radius_m)` constructor can be added later without
  changing the protocol.
- **Circle.** `circle` normalizes the longitude of a polar center to zero so that
  equal requests compare equal; `rectangle` keeps it, because there it orients the
  local frame.

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

- `StormScopeGOES` / `StormScopeMRMS` (implemented): `with_region` accepts boxes.
  Flash variants crop the full footprint with `region_padding` context, a 200-point
  minimum and four-point alignment; baseline variants return full-footprint views.
  Views share experts, crop grids, invariants and valid masks from the full
  footprint, and rebind the experts' image size on every forward.
- `CorrDiffCosmoEra5`: add `with_region`; keep a reference to the full footprint in
  views and adapt the current `set_domain` API through `Region.box`. Treat its run grid
  and trimmed output grid separately.
- `StormCastCONUS` / `StormCast`: share the full weights; select crops without
  replacing network parameters. Keep existing constructor limits as an adapter.
- `cBottleSR`: return an independently configured view from `with_region`; retain
  the in-place setter for compatibility while discouraging its use after signatures
  have been captured.
- Future globe-wide patch models: build a local grid from `Region.rectangle`,
  including its orientation, and ship a documented default region.

The existing `ProjectedGrid` represents 1-D projected axes under one CRS. An
arbitrarily rotated LAEA patch may need another `GridDefinition` if its frame
cannot be expressed by that CRS; this does not change the model protocol.
