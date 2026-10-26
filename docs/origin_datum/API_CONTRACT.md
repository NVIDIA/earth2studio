# NVIDIA Hosted FourCastNet API Contract Specification

**Service URL:** `https://climate.api.nvidia.com/v1/nvidia/fourcastnet`
**Underlying Architecture:** NVIDIA Cloud Functions (NVCF) Microservice
**Hosting Model:** NVIDIA Managed Hosted NIM Cloud Function
**Classification:** `HOSTED_NEURAL_SAMPLE_FORECAST`
**Verification Date:** October 26, 2026
**License & Cost:** NVIDIA Developer Program Free Tier ($0 development quota)

---

## 1. Architectural Distinction: Hosted API vs. Self-Hosted NIM

| Feature | NVIDIA Hosted Cloud API (`climate.api.nvidia.com`) | Self-Hosted NIM (`/v1/infer`) |
|---|---|---|
| **Base URL** | `https://climate.api.nvidia.com/v1/nvidia/fourcastnet` | `http://localhost:8000/v1/infer` |
| **Authentication** | `Bearer nvapi-...` header mandatory | Typically unauthenticated or local API key |
| **Input Mechanism** | Fixed pre-staged sample scenarios (`input_id: 0..4`) | Arbitrary NumPy file uploads (`input_array=@...npy`) |
| **Payload Content** | Small JSON configuration parameters | Multipart form-data with serialized arrays |
| **Response Format** | Content-negotiated (`application/json` or `application/x-tar`) | Tarball stream containing `.npy` step outputs |
| **Execution Pattern** | Synchronous with `nvcf-reqid` tracking | Direct local stream |

---

## 2. Request Contract

### 2.1 HTTP Headers

| Header | Required | Valid Values / Example | Purpose |
|---|---|---|---|
| `Authorization` | **Yes** | `Bearer nvapi-...` | Authenticates against NVIDIA Developer API gateway |
| `Content-Type` | **Yes** | `application/json` | Format of the submitted request body |
| `Accept` | **Yes** | `application/json` or `application/x-tar` | Desired output response format |
| `NVCF-POLL-SECONDS` | Optional | Integer $\ge 1$ (e.g., `5`) | Polling configuration hint for backend gateway |
| `User-Agent` | Optional | `OriginDatum-Earth2/1.0` | Client tracking identifier |

### 2.2 JSON Body Parameters

```json
{
  "input_id": 0,
  "variables": "t2m",
  "simulation_length": 1,
  "ensemble_size": 1,
  "noise_amplitude": 0.0
}
```

| Parameter | Type | Required | Valid Range / Values | Default | Description |
|---|---|---|---|---|---|
| `input_id` | Integer | **Yes** | `0, 1, 2, 3, 4` | None | Index referencing pre-loaded global meteorological sample state. Empirical probe verified indices 0–4 are accepted; indices $\ge 5$ and negative values trigger HTTP 400. |
| `variables` | String | **Yes** | Comma-delimited string subset of: `w10m,t2m,msl,tcwv,z500` | None | Atmospheric and surface variables to simulate and extract. |
| `simulation_length` | Integer | No | `1` to `40` | `4` | Number of forecast steps from initial state. Each step represents a 6-hour interval (e.g., 1 step = 6h, 4 steps = 24h). |
| `ensemble_size` | Integer | No | `1` to `4` | `1` | Number of ensemble members generated. |
| `noise_amplitude` | Float | No | `0.0` to `0.1` | `0.0` | Perturbation amplitude applied to initial condition for ensemble spread. |

---

## 3. Response Contract

### 3.1 Synchronous Execution & Asset Fulfillment
When a valid request is submitted with standard parameters:
1. HTTP Status `200 OK` is returned by the upstream uvicorn/fastapi service.
2. The response header exposes:
   - `Access-Control-Expose-Headers: nvcf-reqid`
   - `Nvcf-Reqid: <UUID>` (Unique NVCF invocation request identifier)
   - `Nvcf-Status: fulfilled`
3. Response payload:
   - In JSON format, the backend returns metadata indicating asset persistence: `{"message":"Large asset written"}`.
   - For tarball requests (`Accept: application/x-tar`), binary tar archive streams are delivered when compatible or fulfilled as large assets.

### 3.2 Error Status Codes Observed Live

| HTTP Status | Condition | Observed Response Body | Client Handling |
|---|---|---|---|
| **401 Unauthorized** | Missing or invalid `Authorization` header | `{"detail":"Header of type authorization was missing"}` | Fail immediately; do not retry with identical credentials. |
| **400 Bad Request** | Out-of-bounds `input_id` (e.g. $\ge 5$, negative) | `{"code":400,"message":"Provided input ID is beyond range of available sample inputs"}` | Expected-error validation; mark test passed if testing boundary. |
| **400 Bad Request** | Malformed JSON or invalid parameter types | `{"code":400,"message":"Bad request body"}` | Raise `ValueError` in client validation. |
| **429 Too Many Requests** | Rate limit reached | Gateway throttled header with optional `Retry-After` | Exponential backoff up to bounded limit; do not flood. |
| **504 Gateway Timeout** | Backend model worker timeout or dropped stream | Empty body or timeout gateway notice | Backoff and report service latency. |

---

## 4. Scientific Scope & Honest Validation Boundaries

The hosted endpoint operates **strictly on pre-staged sample inputs** (`input_id: 0..4`). It is a verification mechanism demonstrating:
1. End-to-end API connectivity and authentication against NVIDIA Earth-2 infrastructure.
2. Real Spherical Fourier Neural Operator (SFNO) neural forecasting execution on NVIDIA hardware.
3. Provenance and artifact tracking of neural forecasts.

**It does NOT:**
- Ingest live user GFS/ERA5 raw grids from local files (which requires the containerized self-hosted NIM).
- Provide independent ground-truth verification without corresponding observational reference data for those specific sample event timestamps.
