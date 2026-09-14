# CAMulator -> Earth2Studio prognostic model: design for review

Source of truth for inference behaviour: HuggingFace repo `willychap/camulator`
(revision `4da83abd466aae4f7f39473c7f4bef83dd5a2ea0`) + the CREDIT toolbox on
GitHub `WillyChap/miles-credit` branch `camulator_huggingface` (`climate/`,
`credit/postblock/_postblock.py`, `credit/transforms/transforms_global.py`,
`credit/models/camulator.py`, `credit/boundary_padding.py`, `credit/physics_core.py`).
Local copies of the branch files: (HF branch files).
Downloaded assets for inspection: `dev/camulator/assets/`.
NCAR-main clone (differs from the HF branch in the energy fixer and wind filter): `/Users/nsobhani/Desktop/work/credit-2/miles-credit`.
Earth2Studio clone (target): `/Users/nsobhani/Desktop/work/credit-2/earth2studio` (branch `main`, clean).

## Facts established about the shipped model

- Grid: 192 lat x 288 lon, 1 deg CESM FV. File latitude is **south-to-north** (-90..90), lon 0..358.75.
- 32 hybrid sigma-pressure levels, index 0 = top (~3.6 hPa), 31 = bottom (~992 hPa). hyai/hybi (33 interfaces) in statics file.
- Time step 6 h. Forcing files use a **noleap** calendar (cftime.DatetimeNoLeap).
- Checkpoint `checkpoint.pt00069.pt` (4.8 GB): dict with `model_state_dict` (+optimizer etc.). Keys are the CREDIT `Camulator` module tree with spectral-norm parametrization (`*.weight_orig`, `*.weight_u`, `*.weight_v`), plus `cube_embedding.*` (constructed but unused for patch=1), plus a parameter-less `postblock.*` (TracerFixer).
- Architecture (must match): frames=1, dim=(256,512,1024,2048), depth=(2,2,18,2), global_window=(4,4,2,1), local_window=3, cross_embed_kernel_sizes=((4,8,16,32),(2,4),(2,4),(2,4)), strides=(2,2,2,2), use_spectral_norm=True, interp=True, earth padding pad_lat=(48,48) pad_lon=(48,48) (poles: 180-deg roll + flip; lon: circular).
- Model input tensor: `[B, 136, 1(time), 192, 288]`, **normalized**, channel order (static_first=True):
  `U[0:32] V[32:64] T[64:96] Qtot[96:128] PS[128] TREFHT[129] z_norm[130] LANDM_COSLAT[131] SOLIN[132] SST[133] ICEFRAC[134] co2vmr_3d[135]`.
- Model output tensor: `[B, 147, 1, 192, 288]`, normalized:
  `U V T Qtot PS TREFHT` (0..129) then diagnostics `PRECT[130] TS[131] CLDHGH[132] CLDLOW[133] CLDMED[134] TAUX[135] TAUY[136] U10[137] QFLX[138] FSDS_J[139] FLDS_J[140] FSUS[141] FLUS[142] SHFLX[143] LHFLX[144] FSUTOA[145] FLUT[146]`.
- Normalization: z-score with `mean_*.nc` / `std_*.nc`. U/V/T/Qtot have per-level (32,) stats; TREFHT and all diagnostics/forcing are scalars; **PS mean/std are (lat,lon) maps**. Statics `z_norm`, `LANDM_COSLAT` are fed **raw** (not normalized); dynamic forcing SOLIN/SST/ICEFRAC/co2vmr_3d **is** normalized with scalar stats. Verified against the shipped IC tensor (channel 130 mean ~0, 131 mean 0.55 = raw land fraction, 135 = -1.54 normalized CO2).
- Forcing time alignment (Quick_Climate.py): the step from t to t+6h uses forcing valid at **t** (input time). The IC tensor already carries forcing at the init time.
- Per-step postprocessing chain in the shipped toolbox (order matters):
  1. Inside `model.forward`: gen1 `PostBlock` = **TracerFixer** only (mass/water/energy have `activate_outside_model: True`). Tracer fixer: per channel denorm -> clip -> renorm. Channels: Qtot (96..127) min 0; PRECT(130) min 0; CLDHGH/CLDLOW/CLDMED (132..134) min 0 max 1; FSDS_J(139), FSUS(141), FSUTOA(145), FLUT(146) min 0.
  2. **Wind artifact filter** (WindPP.py, HF-branch version): in normalized space, in place. Mask from |(U,V)| at level 14 > 2.8 (normalized units), anisotropic dilation 15 (zonal) x 5 (meridional), Gaussian falloff sigma 4 (lon spread 2x), data smoothing separable Gaussian sigma_zonal=2.0 / sigma_meridional=0.5, `preserve_amplitude=True` (RMS rescale, alpha clamp 4), applied to U,V,T,Qtot at levels 9..20. Zero padding in conv2d (not circular). Original code assumes batch=1 (`.squeeze()`).
  3. **GlobalMassFixer** (sigma grid, midpoint=True, denorm inside): with sigma coordinates only **PS is rescaled** so that global dry-air mass matches t0 (the q-fix branch is pressure-level only). Uses hyai/hybi, area from lat2d/lon2d via R^2 * d(sin lat) * d(lon) with `torch.gradient(edge_order=2)`.
  4. **GlobalWaterFixer**: rescales PRECT so global d(TWC)/dt = E - P closes (precip/evap in m per 6 h -> flux via RHO_WATER/21600).
  5. **GlobalEnergyFixerUpDown** (HF-branch version): rescales T (all 32 levels) so column total energy tendency matches TOA - surface fluxes. SOLIN is read from the **input** tensor (index 132, denormalized) and multiplied by N_seconds; FSUTOA/FLUT (W m-2) from output x N_seconds; surface terms are J m-2 accumulations; `F_S = FSDS_J - FSUS + FLDS_J - FLUS + SHFLX + LHFLX` (note **plus** SH/LH: CAMulator stores them positive-into-surface). Physics constants: CP_DRY 1004.64, CP_VAPOR 1810, LH_WATER 2.501e6, GRAVITY 9.80665, RAD_EARTH 6371000, RHO_WATER 1000. Branch accumulates area-weighted sums in float64.
  6. Output written denormalized; the next-step state is `prediction[:, :130]` (normalized) + fresh forcing.
- Units of outputs (camulator_metadata.yaml): PRECT m per 6 h step; QFLX m per 6 h step (positive into surface, i.e. negative = evaporation); SHFLX/LHFLX/FSDS_J/FLDS_J/FSUS/FLUS J m-2 per 6 h step (positive into surface for SH/LH); FSUTOA/FLUT W m-2; TAUX/TAUY N m-2; U10 m s-1 (speed); cloud fractions 0-1; PS Pa; TREFHT/TS K; SOLIN W m-2; SST K; ICEFRAC 0-1; co2vmr_3d mol mol-1.

## Proposed Earth2Studio implementation

### Dependency choice: vendor, do not depend on `miles-credit`
`pip install miles-credit` drags cartopy, pygrib, torch-harmonics<0.9 (conflicts with E2S git-pinned torch-harmonics like `ace2`), segmentation-models-pytorch, etc., and the HF toolbox explicitly says *not* to pip-install it (needs the branch + pinned env). The architecture is ~450 lines of plain torch+einops; E2S already vendors architectures under `earth2studio/models/nn/`. Optional extra: `camulator = ["einops>=0.8.1"]`, included in `all`, no conflicts.

### Files
- `earth2studio/models/nn/camulator.py` — vendored CrossFormer (`CubeEmbedding`, `CrossEmbedLayer`, `DynamicPositionBias`, `Attention`, `FeedForward`, `Transformer`, `UpBlockPS`, `LayerNorm`, `EarthPadding`, `apply_spectral_norm`, `CamulatorNet`). Forward: `[B,C_in,1,H,W] -> [B,C_out,1,H,W]`, identical math to CREDIT (post block removed).
- `earth2studio/models/nn/camulator_physics.py` — `HybridSigmaPhysics` (area, integrals, TWC, dry-air mass; float64 weighted sums), `tracer_clip`, `global_mass_fix`, `global_water_fix`, `global_energy_fix_updown`, `wind_artifact_filter` (batched re-implementation of the HF-branch WindPP).
- `earth2studio/lexicon/camulator.py` — `CAMulatorLexicon` (E2S name <-> CESM name, with modifier for CO2 units).
- `earth2studio/data/camulator.py` — `CAMulatorForcing` data source.
- `earth2studio/models/px/camulator.py` — `CAMulator(torch.nn.Module, AutoModelMixin, PrognosticMixin)`.
- Tests: `test/models/px/test_camulator.py`, `test/data/test_camulator.py`, `test/lexicon/test_camulator_lexicon.py`; `test/conftest.py` `_TEST_DEPENDENCIES` entries.
- Docs: `docs/modules/models_px.md`, `docs/modules/datasources_analysis.md`, `docs/userguide/about/install_options.yml`, `CHANGELOG.md`; `pyproject.toml` extra.

### Variable naming (E2S side)
E2S does not enforce `E2STUDIO_VOCAB` for model variables; the ACE lexicon defines its own model-level names (`u0k`..`u7k`, `mtdwswrf`, `global_mean_co2`, ...) outside the base vocab. Proposal:
- 3D prognostic, k = 0..31 (0 = top): `u{k}k`, `v{k}k`, `t{k}k`, `qtot{k}k` (same `{var}{k}k` model-level convention as ACE; different levels, model-scoped).
- 2D prognostic: PS -> `sp`, TREFHT -> `t2m`.
- Diagnostics (17): PRECT -> `tp06` (m, 6 h accumulation, exists); TS -> `skt`; CLDHGH/CLDMED/CLDLOW -> `hcc`/`mcc`/`lcc`; U10 -> `ws10m`; TAUX/TAUY -> **new** `iews`/`inss` (ECMWF instantaneous eastward/northward turbulent surface stress, N m-2); QFLX -> **new** `e06` (evaporation accumulated over past 6 h, m water equivalent, negative = evaporation) — sign convention matches ERA5 `e`.
  Flux fields converted from J m-2 per 6 h to **mean W m-2 over the 6 h step** (divide by 21600) and mapped to existing ACE-style mean-rate names: FSDS_J -> `msdwswrf`, FLDS_J -> `msdwlwrf`, FSUS -> `msuwswrf`, FLUS -> `msuwlwrf`, SHFLX -> `msshf`, LHFLX -> `mslhf` (positive downward, as ERA5). FSUTOA -> `mtuwswrf`, FLUT -> `mtuwlwrf` (already W m-2). The conversion is applied only at the E2S boundary; the physics fixers operate on native units.
  Alternative considered: keep J m-2 accumulations with `ssrd06`/`strd06` (exist) + new `ssru06`/`stru06` + `sshf`/`slhf`. Rejected for inconsistency (mixed suffixes) — reviewers may disagree.
- Forcing (data source): SOLIN -> `mtdwswrf` (W m-2), SST -> `sst` (K), ICEFRAC -> `sic` (0-1), co2vmr_3d -> `global_mean_co2` in **ppm** (lexicon modifier x1e6; wrapper divides by 1e6 before normalizing).
- Statics (`z_norm`, `LANDM_COSLAT`) are **internal buffers** loaded from the package, not exposed as input variables (unlike ACE2, whose forcing source supplies `z`, `land_abs`). Rationale: `z_norm` is an already-normalized geopotential with no physical E2S counterpart.

### `CAMulatorForcing` data source (`earth2studio/data/camulator.py`)
- `CAMulatorForcing(mode="cyclic" | "transient", cache=True, verbose=True, forcing_file: str | None = None)`. Default `cyclic` downloads `forcing_data/b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc` (1.3 GB, one climatological year, labelled year 2000); `transient` downloads `forcing_data/b.e21.CREDIT_climate_branch_1980_2014.nc` (9.7 GB). Download via `huggingface_hub.HfFileSystem.get_file` from the pinned revision into `datasource_cache_root()/camulator`, like `ACE2ERA5Data`. `forcing_file` allows a local file (tests, custom scenarios).
- Calendar: requested `np.datetime64` (Gregorian) is matched on **(month, day, hour)** for cyclic, **(year, month, day, hour)** for transient, via integer arithmetic on the noleap axis (no cftime construction). Feb 29 maps to Feb 28 (logged once). Hour must be 0/6/12/18 else `ValueError`. Transient out-of-range year -> `ValueError`.
- Returns `xr.DataArray [time, variable, lat, lon]` in physical units, **lat flipped to 90..-90**, coords set to the model constants `CAMULATOR_GRID_LAT`/`CAMULATOR_GRID_LON` so `handshake_coords` passes; exposes `.lat`, `.lon` attributes.
- Note: with `mode="cyclic"` a multi-year rollout sees the same forcing each year (CREDIT's default); a Gregorian E2S clock will step over Feb 29 (which maps to Feb 28's forcing) and therefore drifts by one 6 h step per leap year relative to CREDIT's noleap clock — documented, not "fixed".

### `CAMulator` prognostic model (`earth2studio/models/px/camulator.py`)
- `__init__(core_model, center, scale, ps_center, ps_scale, forcing_center, forcing_scale, statics, hyai, hybi, area_lat2d/lon2d, phis, forcing_data_source=CAMulatorForcing(), conservation_fixers=True, wind_filter=True)`. All stats/statics stored as buffers in **model (S->N) latitude order**; flipping happens only at the E2S boundary. `center`/`scale` shape (147,1,1); PS handled via `(H,W)` maps at channel 128.
- `input_coords`: batch, time, lead_time=[0h], variable = 130 prognostic names, lat = `np.linspace(90,-90,192)`, lon = `np.linspace(0,358.75,288)`.
- `output_coords`: same but variable = 147 names (prognostic + 17 diagnostics), lead_time + 6 h. Validates via `handshake_dim`/`handshake_coords` on lead_time/variable/lat/lon.
- `__call__` (`@batch_func`): x `[B,T,1,130,H,W]` physical -> flip lat -> normalize -> forcing at `time + lead_time` for each `time` (same across batch), CO2 ppm -> mol/mol, normalize -> concat statics + forcing -> `[B*T,136,1,H,W]` -> core -> tracer clip -> wind filter (if enabled) -> mass -> water -> energy-updown (if enabled) -> denormalize 147 -> unit conversion (6 flux fields /21600) -> flip lat -> `[B,T,1,147,H,W]`.
- `_default_generator`: yields IC in **output schema** with diagnostics NaN-filled (same as `ACE2ERA5`), then loops `front_hook -> __call__ -> rear_hook -> yield`, next input = output[..., :130, :, :] with lead_time advanced.
- `load_default_package`: `hf://willychap/camulator@4da83abd466aae4f7f39473c7f4bef83dd5a2ea0` with `same_names` cache. `load_model(package, checkpoint="checkpoint.pt00069.pt", forcing_data_source=..., conservation_fixers=True, wind_filter=True)` resolves the checkpoint, `normalization/mean_*.nc`, `normalization/std_*.nc`, `normalization/statics_b_credit_runs_f32_02.nc` (z_norm, LANDM_COSLAT), `normalization/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc` (hyai, hybi, lat2d, lon2d, PHIS). Loads `model_state_dict` with `postblock.*` keys dropped and `strict=True`.
- Badges: `region:global class:climate product:wind product:precip product:temp product:atmos year:2025 gpu:40gb provider:ncar backend:pytorch` (check `docs/conf.py` for allowed values; `provider:ncar` may not exist).

### Tests
- `test_camulator.py`: tiny real `CamulatorNet` (dim=(8,16,32,64), depth=(1,1,1,1), same windows/padding — all stage sizes divide window sizes: 288x384 -> 144x192 -> 72x96 -> 36x48 -> 18x24) as the "Phoo" core, identity stats, synthetic hyai/hybi/PHIS, `Random` forcing source on the model grid. Tests: call (cpu/cuda, 1-2 times), iter (ensemble 1,2; IC first with NaN diagnostics; lead_time 6h increments), exceptions (wrong variable, wrong lat order), fixers close budgets on synthetic states (mass: dry mass equal to t0; water: E-P-dTWC residual ~0; energy: TE_t1 == TE_t0 + fluxes), wind filter shape/no-op when below threshold, package test (`@pytest.mark.package`, downloads 4.8 GB, uses the shipped IC tensor semantics: neutral input = normalization center denormalized).
- `test/data/test_camulator.py`: write a small noleap NetCDF fixture (cyclic 1460 steps and transient subset), check month/day/hour lookup, Feb 29 -> Feb 28, lat flip, variable order, coords equal to grid constants, CO2 ppm modifier.
- Lexicon test: bidirectional mapping, 147 output names unique.

### Open questions for reviewers
1. Is folding the 17 output-only diagnostics into the prognostic model's output (with NaN at lead 0) the right E2S pattern, or should they be a separate `DiagnosticModel`? (ACE2ERA5 precedent does the former.)
2. Variable naming: model-level `{var}{k}k` names shared with ACE's convention; new base-vocab entries `iews`, `inss`, `e06`; converting J m-2/6h accumulations to mean W m-2 rates.
3. Calendar policy (noleap forcing vs Gregorian E2S time).
4. Statics internal vs exposed through the forcing source.
5. Vendoring vs `miles-credit` dependency.
