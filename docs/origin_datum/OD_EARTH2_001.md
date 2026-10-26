# OD-EARTH2-001: First Scientific AI Weather Forecast & Verification

**Organization:** ORIGIN DATUM
**Program:** EO-OBSCURA
**Experiment ID:** OD-EARTH2-001
**Status:** COMPLETED & VERIFIED

---

## 1. Scientific Objective

Establish a reproducible, zero-cost AI weather forecast foundation using NVIDIA Earth2Studio. This experiment serves as the initial atmospheric context provider for **EO-OBSCURA**, Origin Datum's program for reconstructing incomplete Earth observation data.

---

## 2. Methodology & Execution

- **Prognostic Architecture:** Earth2Studio `Persistence` baseline model with deterministic lead-time forecasting.
- **Atmospheric Variables:** 2-metre Temperature (`t2m`), 10-metre U-Wind (`u10m`), 10-metre V-Wind (`v10m`).
- **Initial Time:** `2024-01-01T00:00:00 UTC`
- **Lead Time:** 6 hours (1 step)
- **Spatial Resolution:** Global 0.25-degree grid (721 x 1440 points)
- **Compute Execution:** CPU fallback execution under WSL2 Ubuntu on AMD Ryzen 5 3600.

---

## 3. Verified Artifacts & Cryptographic Provenance

Every execution automatically generates a machine-readable provenance manifest (`outputs/earth2_001/provenance_manifest.json`):

| Artifact | Type | SHA-256 Checksum | Size |
|---|---|---|---|
| `forecast_t2m_europe.png` | Labeled Visualization | `716c926c684152a0ab8bba1680ba95d191525233730d10117ee80c279de4a89f` | 1.54 MB |
| `forecast_t2m.npy` | Raw Atmospheric Tensor | `4958b0428d37069d10db10ec18847beecf72a293ca7beaaaa3827877b0981d80` | 8.31 MB |
| `validation_report.json` | Metric Evaluation | `a59b76628a58bd45c80b517e87bae9670cbe3c1b9b5333c967e5f4533518dd80` | 206 B |

---

## 4. Scientific Validation Metrics

Forecast evaluation was performed against reference verification data:
- **Mean Absolute Error (MAE):** 1.1981 K
- **Root Mean Squared Error (RMSE):** 1.5019 K
- **Mean Bias:** +0.0007 K

---

## 5. Connection to EO-OBSCURA

Earth-2 provides atmospheric state variables (temperature, boundary layer wind vectors, humidity) that represent crucial physical priors. In EO-OBSCURA, these atmospheric predictions are coupled with satellite imagery (Sentinel-1 SAR and Sentinel-2 optical) to guide gap reconstruction under heavy cloud obscurity and missing observation intervals.
