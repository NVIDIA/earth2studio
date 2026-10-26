# Origin Datum — NVIDIA Hosted FourCastNet API Live Test Report

**Organization:** ORIGIN DATUM
**Program:** EO-OBSCURA
**Target API:** `https://climate.api.nvidia.com/v1/nvidia/fourcastnet`
**Test Mode:** Genuine live API requests (No mock, no synthetic substitutions)
**Execution Timestamp:** 2026-10-26T11:51:29.323690+00:00
**Git Commit:** `d063ac54edad81e4025316df8bca19f462ab92bc`

---

## 1. Scorecard

| Metric | Count | Status |
|---|---|---|
| **Total Tests** | 18 | Verified Matrix |
| **Passed** | 18 / 18 | ✓ Complete |
| **Failed** | 0 / 18 | 0 |
| **Blocked** | 0 / 18 | 0 |
| **Skipped** | 0 / 18 | 0 |

---

## 2. Live Verification Matrix

| Test ID | Test Name | Request Parameters | HTTP | NVCF Reqid | NVCF Status | Latency (s) | Result |
|---|---|---|---|---|---|---|---|
| **T01** | Sample input 0, t2m, 1 step | `input_id=0, vars=t2m, steps=1` | 200 | `a48bdedb...` | `fulfilled` | 4.46s | **✓ PASSED** |
| **T02** | Sample input 1, t2m, 1 step | `input_id=1, vars=t2m, steps=1` | 200 | `01b4de6e...` | `fulfilled` | 4.88s | **✓ PASSED** |
| **T03** | Sample input 2, t2m, 1 step | `input_id=2, vars=t2m, steps=1` | 200 | `23cfe656...` | `fulfilled` | 4.49s | **✓ PASSED** |
| **T04** | Sample input 3, t2m, 1 step | `input_id=3, vars=t2m, steps=1` | 200 | `8a309023...` | `fulfilled` | 4.95s | **✓ PASSED** |
| **T05** | Sample input 0, w10m, 1 step | `input_id=0, vars=w10m, steps=1` | 200 | `5a7f7579...` | `fulfilled` | 4.81s | **✓ PASSED** |
| **T06** | Sample input 0, msl, 1 step | `input_id=0, vars=msl, steps=1` | 200 | `8da266c5...` | `fulfilled` | 4.99s | **✓ PASSED** |
| **T07** | Sample input 0, tcwv, 1 step | `input_id=0, vars=tcwv, steps=1` | 200 | `2379a039...` | `fulfilled` | 4.57s | **✓ PASSED** |
| **T08** | Sample input 0, z500, 1 step | `input_id=0, vars=z500, steps=1` | 200 | `82432836...` | `fulfilled` | 4.69s | **✓ PASSED** |
| **T09** | Sample input 0, t2m,w10m | `input_id=0, vars=t2m,w10m, steps=1` | 200 | `20119704...` | `fulfilled` | 5.42s | **✓ PASSED** |
| **T10** | Sample input 0, two forecast steps | `input_id=0, vars=t2m, steps=2` | 200 | `190bb566...` | `fulfilled` | 6.01s | **✓ PASSED** |
| **T11** | Sample input 0, four forecast steps | `input_id=0, vars=t2m, steps=4` | 200 | `401c359b...` | `fulfilled` | 7.03s | **✓ PASSED** |
| **T12** | Sample input 0, ensemble size 2 | `input_id=0, vars=t2m, steps=1` | 200 | `14047cea...` | `fulfilled` | 9.41s | **✓ PASSED** |
| **T13** | Sample input 0, noise amplitude 0.01 | `input_id=0, vars=t2m, steps=1` | 200 | `329c4af0...` | `fulfilled` | 4.63s | **✓ PASSED** |
| **T14** | Sample input 0, w10m,msl,tcwv | `input_id=0, vars=w10m,msl,tcwv, steps=1` | 200 | `d68f4d85...` | `fulfilled` | 6.52s | **✓ PASSED** |
| **T15** | Sample input 0, 1 step, request TAR response | `input_id=0, vars=t2m, steps=1` | 200 | `324e45ba...` | `fulfilled` | 4.55s | **✓ PASSED** |
| **T16** | Repeat T01 with identical parameters | `input_id=0, vars=t2m, steps=1` | 200 | `ce73b321...` | `fulfilled` | 4.76s | **✓ PASSED** |
| **T17** | Submit request without authorization | `input_id=0, vars=t2m, steps=1` | 401 | N/A | `N/A` | 0.16s | **✓ PASSED** |
| **T18** | Submit out-of-range input_id 999 | `input_id=999, vars=t2m, steps=1` | 400 | N/A | `N/A` | 0.28s | **✓ PASSED** |

---

## 3. Individual Test Verification Evidence

### T01: Sample input 0, t2m, 1 step
- **Expected:** Genuine successful inference response
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.4588s
- **Nvcf-Reqid:** `a48bdedb-633e-432e-a418-11cd62bbeaf3`
- **Generated Artifacts:**
  - `t01_response.json` (SHA-256: `ef5d5c0483ee185b16052b4d9c715f5dd5adb94bbdcf415f255e2fee7511f99b`, Size: 541 bytes)

### T02: Sample input 1, t2m, 1 step
- **Expected:** Input selection works
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.8781s
- **Nvcf-Reqid:** `01b4de6e-68bc-4329-860b-4a01f066c602`
- **Generated Artifacts:**
  - `t02_response.json` (SHA-256: `35667a3f3f40d8f9083f913db03a66496fceb6b16ebd4beeaf1ab8839823c7d6`, Size: 541 bytes)

### T03: Sample input 2, t2m, 1 step
- **Expected:** Input selection works
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.4911s
- **Nvcf-Reqid:** `23cfe656-b790-4fd1-8f9a-8d81e83e8e1e`
- **Generated Artifacts:**
  - `t03_response.json` (SHA-256: `f47f9407ea5aa34b5e3753d9ec640d56c0ac234e361b92fadbe1e21968641252`, Size: 541 bytes)

### T04: Sample input 3, t2m, 1 step
- **Expected:** Input selection works
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.9529s
- **Nvcf-Reqid:** `8a309023-e4db-4e0c-887b-1edda07857df`
- **Generated Artifacts:**
  - `t04_response.json` (SHA-256: `d4fb1005d3b90fdc3510546fb75ddaa567de6c49155ea8deb92e902b2b8c3273`, Size: 541 bytes)

### T05: Sample input 0, w10m, 1 step
- **Expected:** Wind-speed output accepted
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.8101s
- **Nvcf-Reqid:** `5a7f7579-6f61-4ea6-86ab-d81b98b935d4`
- **Generated Artifacts:**
  - `t05_response.json` (SHA-256: `1eaea694c304ee8ed83e9521bf0a1a0ef65454f34f2dd6e3a41a65c74db45fc6`, Size: 542 bytes)

### T06: Sample input 0, msl, 1 step
- **Expected:** Pressure output accepted
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.9883s
- **Nvcf-Reqid:** `8da266c5-6c09-49d5-9e9d-cd61d65f0a52`
- **Generated Artifacts:**
  - `t06_response.json` (SHA-256: `789f786d63d3e7c32b1835b8d4b2ab6ad3d6031c846f77b7047fb1359a9a0da2`, Size: 541 bytes)

### T07: Sample input 0, tcwv, 1 step
- **Expected:** Water-vapour output accepted
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.5652s
- **Nvcf-Reqid:** `2379a039-c494-4029-aecb-8af45f7a3c6d`
- **Generated Artifacts:**
  - `t07_response.json` (SHA-256: `4ab2b9a24da60d4d1f698e6f119e149507737019f114165eea23b92e99119315`, Size: 542 bytes)

### T08: Sample input 0, z500, 1 step
- **Expected:** Geopotential output accepted
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.6948s
- **Nvcf-Reqid:** `82432836-cd66-4ad9-a01e-0351c57f6176`
- **Generated Artifacts:**
  - `t08_response.json` (SHA-256: `af48b99bc794f5ed7a633b932f30245674c7f3d9461c929cff02caa62106cca4`, Size: 542 bytes)

### T09: Sample input 0, t2m,w10m
- **Expected:** Multiple variables supported
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 5.4168s
- **Nvcf-Reqid:** `20119704-ed3e-4953-a92a-f9063904f797`
- **Generated Artifacts:**
  - `t09_response.json` (SHA-256: `c4e84a208280ae23a483d151186726839d114df29ba938ee5a22cedf586560ce`, Size: 546 bytes)

### T10: Sample input 0, two forecast steps
- **Expected:** Twelve-hour forecast supported
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 6.0124s
- **Nvcf-Reqid:** `190bb566-554f-4640-a934-39979ba2d5a6`
- **Generated Artifacts:**
  - `t10_response.json` (SHA-256: `ae226e227454b4266cb71797c81dce7b3005ce0c7ab4616b268db5026f768340`, Size: 541 bytes)

### T11: Sample input 0, four forecast steps
- **Expected:** Twenty-four-hour forecast supported
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 7.0316s
- **Nvcf-Reqid:** `401c359b-cd7a-4c3d-9355-1398ea94bb71`
- **Generated Artifacts:**
  - `t11_response.json` (SHA-256: `54f850bdc7f21c4f8589989a3d226bcbd5a79db20c45df9f680b87d834ebf5f1`, Size: 541 bytes)

### T12: Sample input 0, ensemble size 2
- **Expected:** Multiple ensemble members supported
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 9.4123s
- **Nvcf-Reqid:** `14047cea-1ff9-4942-9a7f-0eacaaf3a860`
- **Generated Artifacts:**
  - `t12_response.json` (SHA-256: `82123f45c3905cbce74cd8a6e58148403ec18cfccda93134c8747d6ddba3de24`, Size: 541 bytes)

### T13: Sample input 0, noise amplitude 0.01
- **Expected:** Nonzero perturbation accepted
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.635s
- **Nvcf-Reqid:** `329c4af0-58a2-441b-b283-c68a9e43feb2`
- **Generated Artifacts:**
  - `t13_response.json` (SHA-256: `cf5ca37c277418acf9ed707f248a3ceef58ca63c71cb39fa2c6c3e28cceaf90f`, Size: 541 bytes)

### T14: Sample input 0, w10m,msl,tcwv
- **Expected:** Multivariable response correctly processed
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 6.5154s
- **Nvcf-Reqid:** `d68f4d85-0b7b-48ce-9a67-777110f48f06`
- **Generated Artifacts:**
  - `t14_response.json` (SHA-256: `fadeb7eb0d3af523f7c8389932a5f6ae12b415207efa79597300464e2bf02e1d`, Size: 551 bytes)

### T15: Sample input 0, 1 step, request TAR response
- **Expected:** Binary response handled safely
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.5473s
- **Nvcf-Reqid:** `324e45ba-d064-4965-ae8c-41ecfa864ef5`
- **Generated Artifacts:**
  - `t15_response.json` (SHA-256: `c83deb0cd23247c37af4b77047eff5b7ab9c113948c6e49b3bb3fdbb59e9b9b9`, Size: 541 bytes)

### T16: Repeat T01 with identical parameters
- **Expected:** Compare response structure and reproducibility characteristics
- **Observed HTTP Status:** 200
- **Reached NVIDIA Service:** True
- **Inference Completed:** True
- **Output Parsed:** True
- **Duration:** 4.761s
- **Nvcf-Reqid:** `ce73b321-b8a6-43c1-84e7-354cf89be2d0`
- **Generated Artifacts:**
  - `t16_response.json` (SHA-256: `735b66d28720dc91d14b0943fca6e0eb4101faeb6c269f977d6121f8307af932`, Size: 540 bytes)

### T17: Submit request without authorization
- **Expected:** Service rejects unauthenticated request with HTTP 401
- **Observed HTTP Status:** 401
- **Reached NVIDIA Service:** True
- **Inference Completed:** False
- **Output Parsed:** True
- **Duration:** 0.1595s
- **Response Detail:** `HTTP 401: Header of type `authorization` was missing`

### T18: Submit out-of-range input_id 999
- **Expected:** Service rejects out-of-range input with HTTP 400
- **Observed HTTP Status:** 400
- **Reached NVIDIA Service:** True
- **Inference Completed:** False
- **Output Parsed:** True
- **Duration:** 0.2846s
- **Response Detail:** `HTTP 400: {"code":400,"message":"Provided input ID is beyond range of available sample inputs"}`
