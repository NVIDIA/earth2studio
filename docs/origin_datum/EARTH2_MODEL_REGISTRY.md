# EARTH-2 CANDIDATE MODEL REGISTRY

**Program:** EO-OBSCURA / OD-EARTH2-001
**Organization:** ORIGIN DATUM
**Hardware Profile:** AMD Ryzen 5 3600 (CPU inference), AMD Radeon RX 9060 XT (non-CUDA)

---

## 1. DLWP (Deep Learning Weather Prediction)
- **Official Source:** Weyn et al. / NVIDIA Earth2Studio `earth2studio.models.px.DLWP`
- **Supported Application:** Medium-range global atmospheric forecast (1.4-degree resolution)
- **Checkpoint Availability:** Public anonymous checkpoint hosted on NVIDIA NGC catalog
- **License:** Apache-2.0
- **Dependencies:** `torch`, `netcdf4`, `xarray`
- **Required Hardware:** CPU or CUDA (minimal memory, ~2 GB RAM for inference)
- **Input Data Requirements:** 2-step history at 6-hour interval (`t2m`, `t850`, `z1000`, `z700`, `z500`, `z300`)
- **Output Variables:** 6 prognostic surface and geopotential levels
- **Runs on Current Environment:** YES (via CPU execution)
- **Current Status:** TESTED / AVAILABLE

---

## 2. FourCastNet / FourCastNet3
- **Official Source:** NVIDIA Research / Pathak et al. (`earth2studio.models.px.FCN` / `FCN3`)
- **Supported Application:** High-resolution global forecast (0.25-degree, 721x1440 grid)
- **Checkpoint Availability:** Public on NVIDIA NGC
- **License:** Apache-2.0
- **Dependencies:** `torch`, `torch-harmonics` (AFNO)
- **Required Hardware:** Dedicated NVIDIA GPU (minimum 16 GB VRAM) for local weights; alternatively NVIDIA Hosted API
- **Input Data Requirements:** 73 surface and upper-air variables from GFS or ERA5
- **Output Variables:** Global winds (`u10m`, `v10m`), temperature (`t2m`), pressure levels
- **Runs on Current Environment:** NO (exceeds local CPU memory/speed constraints for local inference; suitable via Hosted API)
- **Current Status:** BLOCKED (Local) / FUTURE EVALUATION (Hosted API)

---

## 3. StormScope EU
- **Official Source:** NVIDIA Earth-2 High-Resolution Regional Models
- **Supported Application:** Kilometer-scale regional convective storm and cloud forecasting across Europe
- **Checkpoint Availability:** Proprietary / Restricted Research Access on NGC
- **License:** NVIDIA Proprietary Research License
- **Dependencies:** `earth2studio`, high-resolution boundary-condition processors
- **Required Hardware:** Multi-GPU NVIDIA A100/H100 clusters
- **Input Data Requirements:** High-resolution NWP boundary forcing (e.g. ECMWF IFS or ICON)
- **Output Variables:** Radar reflectivity, precipitation, convective parameters
- **Runs on Current Environment:** NO
- **Current Status:** FUTURE EVALUATION

---

## 4. Atlas
- **Official Source:** NVIDIA Climate Simulation Group
- **Supported Application:** Multi-decadal climate projection and statistical downscaling
- **Checkpoint Availability:** NVIDIA Climate Research Catalog
- **License:** Apache-2.0 / Research Use
- **Dependencies:** `earth2studio`, `scipy`, `cftime`
- **Required Hardware:** NVIDIA GPU clusters (VRAM intensive)
- **Input Data Requirements:** CMIP6 or long-term reanalysis (ERA5)
- **Output Variables:** Climate indicators, long-term temperature, precipitation anomalies
- **Runs on Current Environment:** NO
- **Current Status:** FUTURE EVALUATION

---

## 5. CorrDiff
- **Official Source:** NVIDIA Earth-2 Generative Diffusion Downscaling
- **Supported Application:** Kilometer-scale generative super-resolution downscaling from coarse forecasts
- **Checkpoint Availability:** Pre-trained regional checkpoints (e.g., Taiwan, Europe) on NGC
- **License:** Apache-2.0 / NVIDIA Open Model License
- **Dependencies:** `torch`, generative diffusion toolkits
- **Required Hardware:** NVIDIA RTX 4090 / A100 (24 GB+ VRAM)
- **Input Data Requirements:** Coarse resolution atmospheric fields (0.25 deg)
- **Output Variables:** 2 km radar reflectivity, extreme wind speed, 2 m temperature
- **Runs on Current Environment:** NO (Requires CUDA tensor cores)
- **Current Status:** BLOCKED

---

## 6. HealDA (Data Assimilation)
- **Official Source:** NVIDIA PhysicsNeMo / Earth2Studio Data Assimilation
- **Supported Application:** Ensemble Kalman Filter and neural atmospheric data assimilation
- **Checkpoint Availability:** Codebase models in Earth2Studio DA modules
- **License:** Apache-2.0
- **Dependencies:** `earth2studio`, `scipy`, `numpy`
- **Required Hardware:** CPU or CUDA (CPU viable for small ensemble dimensions)
- **Input Data Requirements:** Sparse in-situ or satellite observation frames + background forecast
- **Output Variables:** Assimilated atmospheric analysis state
- **Runs on Current Environment:** YES (Small prototype ensembles)
- **Current Status:** NOT TESTED
