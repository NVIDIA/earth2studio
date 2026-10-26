# OD-EARTH2-001: Genuine AI Weather Forecast & Scientific Verification

**Organization:** ORIGIN DATUM  
**Program:** EO-OBSCURA  
**Experiment ID:** OD-EARTH2-001  
**Execution Date:** 2026-10-26 (Simulated Atmospheric Time: 2024-01-01 00:00 UTC)  
**Classification:** GENUINE_ATMOSPHERIC_EXPERIMENT  
**Model Type:** BASELINE_PERSISTENCE  
**Data Source:** HISTORICAL_GFS (NOAA GFS Analysis via AWS Open Data S3)  
**Verification Reference:** NOAA GFS Analysis at 2024-01-01T06:00:00 UTC  
**Validation Status:** VALIDATED  

---

## 1. Scientific Verification Integrity & Distinctions

To ensure strict scientific and engineering rigor, Origin Datum enforces clear distinctions across experimental tiers:
1. **`SYNTHETIC_SMOKE_TEST`**: Run using synthetic random fields (`earth2studio.data.Random`). Validates pipeline execution, tensor plumbing, and Zarr I/O without asserting meteorological correctness.
2. **`GENUINE_ATMOSPHERIC_EXPERIMENT`**: Uses real historical weather observations (`earth2studio.data.GFS`), runs model inference, extracts predictions directly from the generated Zarr array store, and compares against independent meteorological analyses at the verification lead time.
3. **Execution vs. Validation**: Execution success proves software pipeline completion. Validation success requires genuine observation reference data and latitude-weighted error metric evaluation.

---

## 2. Methodology & Experimental Configuration

- **Prognostic Architecture:** Earth2Studio `Persistence` baseline model with deterministic lead-time forecasting.
- **Input Data Source:** NOAA GFS 0.25-degree Analysis (`s3://noaa-gfs-bdp-pds/gfs.20240101/00/atmos/gfs.t00z.pgrb2.0p25.f000`).
- **Initial Forecast Time:** `2024-01-01T00:00:00 UTC`.
- **Forecast Horizon:** +6 hours (1 step, lead time = 6h).
- **Prognostic Variables:**
  - 2-metre Temperature (`t2m`, K)
  - 10-metre U-Wind component (`u10m`, m/s)
  - 10-metre V-Wind component (`v10m`, m/s)
- **Spatial Resolution:** Global 0.25-degree grid (721 latitude x 1440 longitude coordinates).
- **Compute Hardware:** CPU inference under WSL2 Ubuntu (AMD Ryzen 5 3600 6-core processor).

---

## 3. Verified Artifacts & Cryptographic Provenance

All arrays were extracted directly from the Earth2Studio `forecast.zarr` store and cryptographically verified in `outputs/earth2_001/provenance_manifest.json`:

| Artifact | Description | SHA-256 Checksum | Size |
|---|---|---|---|
| `forecast_t2m.npy` | Forecast 2m Temperature tensor | `e63a97e30e71a98a3510499446e32b23a3946d5ace84d26e3ad9dd48173d39a4` | 4.15 MB |
| `forecast_t2m.png` | Labeled Temperature plot | `019fa67b79ff4ceb126eb8c9aea093015f947b6c5d60c0b78aa4f7eb8158adc1` | 534.6 KB |
| `forecast_u10m.npy` | Forecast 10m U-Wind tensor | `84e88d88f0287f5bd4ef7f15749779b420d432280e57df8d8cb2e8c66b50ce97` | 4.15 MB |
| `forecast_u10m.png` | Labeled U-Wind plot | `fcdffb41e57137d790b02d9158b96e26a73ff902097ea400f7463311d6ea5ca5` | 618.5 KB |
| `forecast_v10m.npy` | Forecast 10m V-Wind tensor | `8b3cdcff6e71d65de6212dd76fbd084b52822ba9c8ab08d7d17780efcc450b21` | 4.15 MB |
| `forecast_v10m.png` | Labeled V-Wind plot | `380b89b4fe77b71b006a3b73d8dc519c96822372d4308ea2eae75c2057dcd8c6` | 647.7 KB |
| `reference_t2m.npy` | Genuine GFS Analysis at +6h | `deddf7f7ee0d13afbcb534a754bdfba776f9650c0fc1eeb84cbf54378e3097af` | 8.31 MB |
| `reference_u10m.npy` | Genuine GFS Analysis at +6h | `49a204cf255f81351cacc5e75927eb8c58fc3d0c3b0e3857e2cf679ca0656a34` | 8.31 MB |
| `reference_v10m.npy` | Genuine GFS Analysis at +6h | `52bca94723e09456e6b6f44ee95ec2a8423009fc80b1173ef6c0bc4d5d5fd87c` | 8.31 MB |
| `validation_report.json` | Latitude-weighted metrics | `d667443282ed358901bab87ec1ede1ab4546f0e812de2af9a65d3fbde24bf4c6` | 451 B |

---

## 4. Scientific Validation Results (+6h Lead Time)

Verification against the genuine NOAA GFS analysis at `2024-01-01T06:00:00 UTC` using area/latitude weighting ($\cos(\text{lat})$):

- **2-metre Temperature (`t2m`):**
  - **MAE:** `1.2635 K`
  - **RMSE:** `2.3813 K`
  - **Mean Bias:** `-0.3313 K`
- **10-metre U-Wind component (`u10m`):**
  - **MAE:** `1.4975 m/s`
  - **RMSE:** `2.4188 m/s`
  - **Mean Bias:** `-0.0341 m/s`
- **10-metre V-Wind component (`v10m`):**
  - **MAE:** `1.6118 m/s`
  - **RMSE:** `2.5368 m/s`
  - **Mean Bias:** `+0.0719 m/s`

**Meteorological Analysis:**
These baseline values reflect realistic diurnal atmospheric variability over a 6-hour interval on a global scale. The persistence model preserves the initial state; temperature changes across sunrise/sunset zones create a characteristic global MAE around ~1.26 K, establishing our ground-truth reference baseline for future neural prognostic evaluations (e.g., DLWP).

---

## 5. Live NOAA GFS & Remote API Verification Procedures

### Live NOAA GFS Ingestion Verification
The genuine atmospheric execution pulls operational GRIB2 analysis datasets anonymously from NOAA's Big Data Program S3 bucket (`s3://noaa-gfs-bdp-pds/`).
1. **Initial Condition Ingestion:** `gfs.20240101/00/atmos/gfs.t00z.pgrb2.0p25.f000` (Analysis time: `2024-01-01T00:00:00 UTC`).
2. **Verification Analysis Ingestion:** `gfs.20240101/06/atmos/gfs.t06z.pgrb2.0p25.f000` (Verification time: `2024-01-01T06:00:00 UTC`).
3. **Execution Command:**
   ```bash
   python -m od_experiments.earth2_001.cli run --source gfs
   ```
4. **Validation Procedure:**
   ```bash
   python -m od_experiments.earth2_001.cli validate --output-dir outputs/earth2_001
   ```

### FourCastNet & Hosted API Verification Protocol
Because local FourCastNet and FourCastNet3 inference is blocked on non-CUDA AMD GPU hardware, remote execution pathways utilize Earth2Studio's client architecture (`earth2studio.serve.client.Earth2StudioClient`):
1. **Pre-flight Credential Isolation:** Ensure any remote access credentials (e.g., NGC or NVIDIA API keys) are sanitized in memory and redacted from all manifests via `od_experiments.earth2_001.security.sanitize_data`.
2. **API Endpoint Verification:** Connect to an operational Earth2Studio serve instance or hosted microservice:
   ```python
   from earth2studio.serve.client import Earth2StudioClient
   
   client = Earth2StudioClient(base_url="http://localhost:8000")
   health_resp = client.health_check()
   assert health_resp.status == "healthy"
   ```
3. **Remote Model Evaluation:** Submit prognostic inference requests specifying `model_name="fcn"` or `"fcn3"`, download the resulting remote Zarr tensors, and evaluate using `od_experiments.earth2_001.validate.evaluate_forecast`.
