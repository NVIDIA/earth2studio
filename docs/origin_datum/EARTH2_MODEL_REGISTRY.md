# EARTH-2 CANDIDATE MODEL REGISTRY

**Program:** EO-OBSCURA / OD-EARTH2-001
**Organization:** ORIGIN DATUM
**Hardware Profile:** AMD Ryzen 5 3600 (CPU inference), AMD Radeon RX 9060 XT (non-CUDA)

---

## 0. Baseline Persistence (`earth2studio.models.px.Persistence`)
- **Official Source:** NVIDIA Earth2Studio (`earth2studio.models.px.Persistence`)
- **Supported Application:** Zero-skill atmospheric baseline; identity persistence across arbitrary forecast lead times
- **Checkpoint Availability:** None required (algebraic persistence operator)
- **License:** Apache-2.0
- **Dependencies:** `torch`, `earth2studio`
- **Required Hardware:** CPU or CUDA (minimal memory, operates seamlessly on standard CPU)
- **Input Data Requirements:** Any valid coordinate system with time and meteorological variable dimensions (e.g. NOAA GFS Analysis 0.25°)
- **Output Variables:** Preserves all input prognostic variables (`t2m`, `u10m`, `v10m`, etc.)
- **Runs on Current Environment:** YES (Fully tested and verified on CPU under WSL2 Ubuntu)
- **Current Status:** TESTED / VERIFIED (Benchmark baseline establishing ground-truth MAE/RMSE for neural model comparison)

---

## 1. DLWP (Deep Learning Weather Prediction)
- **Official Source:** Weyn et al. / NVIDIA Earth2Studio `earth2studio.models.px.DLWP`
- **Supported Application:** Medium-range global atmospheric forecast (1.4-degree resolution)
- **Checkpoint Availability:** Public checkpoint hosted on NVIDIA NGC catalog (`https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/dlwp`)
- **License:** Apache-2.0
- **Dependencies:** `torch`, `netcdf4`, `xarray`
- **Required Hardware:** CPU or CUDA (minimal memory, ~2 GB RAM for inference)
- **Input Data Requirements:** 2-step history at 6-hour interval (`t2m`, `t850`, `z1000`, `z700`, `z500`, `z300`)
- **Output Variables:** 6 prognostic surface and geopotential levels
- **Runs on Current Environment:** Untested on this machine (Checkpoint not yet downloaded or executed locally)
- **Current Status:** NOT TESTED (Do not label tested until the checkpoint is downloaded, loaded, and genuinely executed locally)

---

## 2. FourCastNet / FourCastNet3
- **Official Source:** NVIDIA Research / Pathak et al. (`earth2studio.models.px.FCN` / `FCN3`)
- **Supported Application:** High-resolution global forecast (0.25-degree, 721x1440 grid)
- **Checkpoint Availability:** Public on NVIDIA NGC
- **License:** Apache-2.0
- **Dependencies:** `torch`, `torch-harmonics` (AFNO)
- **Required Hardware:** Dedicated NVIDIA GPU (minimum 16 GB VRAM) for local weights; alternatively NVIDIA Hosted API / Earth2Studio Server
- **Input Data Requirements:** 73 surface and upper-air variables from GFS or ERA5
- **Output Variables:** Global winds (`u10m`, `v10m`), temperature (`t2m`), pressure levels
- **Runs on Current Environment:** NO for local execution (blocked due to non-CUDA AMD GPU and VRAM/compute constraints); client integration configured via Earth2Studio remote inference API (`earth2studio.serve.client.Earth2StudioClient`)
- **Current Status:** BLOCKED (Local Execution on Non-CUDA GPU) / CONFIGURED (Remote API Client Integration)

---

## 3. StormScope EU
- **Official Source:** NVIDIA Earth-2 High-Resolution Regional Models
- **Supported Application:** Kilometer-scale regional convective storm and cloud forecasting across Europe
- **Checkpoint Availability:** Research access on NVIDIA NGC
- **License:** NVIDIA Research License
- **Dependencies:** `earth2studio`, high-resolution boundary-condition processors
- **Required Hardware:** Multi-GPU NVIDIA A100/H100 clusters
- **Input Data Requirements:** High-resolution NWP boundary forcing (e.g. ECMWF IFS or ICON)
- **Output Variables:** Radar reflectivity, precipitation, convective parameters
- **Runs on Current Environment:** NO
- **Current Status:** FUTURE EVALUATION

---

## 4. Atlas
- **Official Source:** NVIDIA Earth-2 Atlas / Stochastic Interpolant Models (`earth2studio.models.px.Atlas`)
- **Supported Application:** Generative continuous representation downscaling, super-resolution interpolation, and continuous spatial-temporal field representation on global 0.25° lat-lon grids using stochastic interpolants and latent autoencoders
- **Checkpoint Availability:** NVIDIA NGC Catalog (`https://huggingface.co/nvidia/atlas-era5`)
- **License:** Apache-2.0 / NVIDIA Open Model License
- **Dependencies:** `earth2studio`, `physicsnemo`, `torch`
- **Required Hardware:** NVIDIA GPU clusters (VRAM intensive; Tensor Core acceleration required)
- **Input Data Requirements:** 2-step history at 6-hour interval (t-6h, t) across full-resolution physical variables (u, v, z, t, q levels, sst, tp06)
- **Output Variables:** Continuous arbitrary-resolution and full 0.25° atmospheric prognostic fields at t+6h
- **Runs on Current Environment:** NO (Requires CUDA and `physicsnemo` module)
- **Current Status:** FUTURE EVALUATION

---

## 5. CorrDiff
- **Official Source:** NVIDIA Earth-2 Generative Diffusion Downscaling (`earth2studio.models.dx.CorrDiff`, `CorrDiffCOSMOERA5`, `CorrDiffERA5HRRR`)
- **Supported Application:** Kilometer-scale generative super-resolution downscaling from coarse numerical forecasts using score-based diffusion models
- **Checkpoint Availability:** Pre-trained regional checkpoints (e.g., Taiwan, Europe, CONUS) on NGC
- **License:** Apache-2.0 / NVIDIA Open Model License
- **Dependencies:** `torch`, generative diffusion toolkits, CUDA runtime
- **Required Hardware:** NVIDIA RTX 4090 / A100 (24 GB+ VRAM required for diffusion sampling passes)
- **Input Data Requirements:** Coarse resolution atmospheric fields (0.25 deg)
- **Output Variables:** 2 km radar reflectivity, extreme wind speed, 2 m temperature
- **Runs on Current Environment:** NO (Requires CUDA tensor cores and high VRAM)
- **Current Status:** BLOCKED (Hardware Incompatible)

---

## 6. HealDA (Data Assimilation)
- **Official Source:** NVIDIA PhysicsNeMo / Earth2Studio Data Assimilation (`earth2studio.models.da.HealDA`, `HealDA_v2`)
- **Supported Application:** Neural data assimilation and Ensemble Kalman Filter coupling on HEALPix and spherical grids for sparse observational integration
- **Checkpoint Availability:** Earth2Studio DA modules (`earth2grid`, `physicsnemo.experimental.models.healda`)
- **License:** Apache-2.0
- **Dependencies:** `earth2studio`, `physicsnemo`, `scipy`, `numpy`, optional `cupy`/`cudf`
- **Required Hardware:** CPU or CUDA (CPU viable for small prototype ensemble dimensions, CUDA required for production assimilations)
- **Input Data Requirements:** Sparse in-situ or satellite observation frames + background forecast state
- **Output Variables:** Assimilated atmospheric analysis state
- **Runs on Current Environment:** UNTESTED (Requires `physicsnemo` and `earth2grid` optional dependencies)
- **Current Status:** NOT TESTED
