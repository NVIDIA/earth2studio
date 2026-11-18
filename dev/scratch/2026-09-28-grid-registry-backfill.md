# Grid Registry Backfill Implementation Plan

**Goal:** Register the fixed global grids used by models and reuse their names in model signatures.

**Architecture:** Keep the existing grid classes and registry API. Add 1-degree and
1.5-degree north-to-south global LatLonGrid definitions, an ascending F90 Gaussian
LatLonGrid with half-degree longitude centers, and HEALPix levels 3, 6, and 10.
HEALPix variants include nested and ring flat layouts and north-origin clockwise
XY flat/face layouts. Preserve configurable model geometry. HRRR regional domains
remain selections of the existing registered parent, not separate registrations.

**Tech Stack:** NumPy, xarray, pytest, existing Earth2Studio grid definitions.

## Tasks

- [x] Add registry tests for global axes, Gaussian quadrature geometry, HEALPix
  ordering/layout and selected pixel coordinates, and HRRR domain selections.
- [x] Run the tests before implementation to confirm missing registrations.
- [x] Extend `earth2studio/grids/__init__.py` using existing grid definitions.
- [x] Update fixed model signatures in GraphCast's shared signature helper,
  FuXiS2S, UCast, ACE2, DLESyM, and CBottle; preserve custom/test geometries.
- [x] Keep ACE2 data and model coordinates consistent with the Gaussian registry.
- [x] Document built-in names and model coverage in the grid user guide.
- [x] Run grid and affected coordinate/model tests, Black, Ruff, and diff checks.

No commits or pushes until requested.
