---
date:
  created: 2026-08-11
  updated: 2026-08-11
readtime: 20
pin: true
authors:
  - skhajehei
links:
  - Homepage: index.md
  - Blog index: blog/index.md
  - External links:
    - PhysicsNeMo GitHub: https://github.com/NVIDIA/physicsnemo
    - NVIDIA Earth-2: https://www.nvidia.com/en-us/high-performance-computing/earth-2/
categories:
  - Earth-2
tags:
  - Blog
  - Earth-2
  - DLESym
  - ERA5
  - Agricultural AI
  - Soil Health
  - LSTM
  - Climate AI
slug: earth2-soil-health-prediction
---

# A Digital Twin for Soil Health Prediction, Powered by Subseasonal Climate AI

NVIDIA Earth-2 provides a platform foundation for weather and climate modeling, but agricultural
decisions rarely depend directly on atmospheric variables—they hinge on how the *living soil*
responds. Team-Soil, a collaboration between AreaandDee LLC, BioSensor Solutions, and NVIDIA, is
building that missing local-response layer: a compact attention-augmented LSTM trained on ERA5
climate drivers that predicts microbial CO₂ flux and colonization timing site-by-site. On a held-out
random-split test set the model reaches an R² of **0.982**, and a GPU optimization pass on a single
H100 cuts end-to-end training time by **2.5×**—turning overnight ablation experiments into
same-afternoon iteration.

<!-- more -->

## Background: Closing the Gap Between Climate Forecasts and Field Decisions

NVIDIA Earth-2 (E2) supports organizations developing weather and climate forecasting applications
with its models, acceleration technologies, and data workflows. Agriculture represents a natural
downstream expansion, but moving from global atmospheric variables to actionable field decisions
requires a connected modeling chain:

> **global weather & climate → regional/local environmental forcing → soil & agricultural response**

Building that chain is a two-step process. **Step 1** develops a local-response model that maps
environmental forcing and soil context to biological response. **Step 2** connects large-scale
weather and climate model outputs to the site-level variables that response model requires. This
post describes **Step 1** in full. Coupling to NVIDIA Earth-2 DLESym subseasonal forecasts (**Step
2**) remains future work, with a controlled transfer protocol already defined.

If successful, this global-to-local pattern could underpin future customer- and partner-developed
agricultural applications: soil-carbon management, irrigation support, crop-stress analysis,
agricultural risk assessment, and field-operation planning.

## Predictive Soil Health: The Team-Soil Pipeline

Soil is one of Earth's largest active carbon reservoirs, but the biological processes governing soil
carbon flux are intensely local. Temperature, moisture, radiation, organic carbon, and microbial
state interact with lags, thresholds, and site-specific sensitivity. A regional climate signal can
tell us what the atmosphere is doing, but field decisions often depend on a more specific question:
**how will the living soil respond?**

Team-Soil is building exactly that model. The project combines ERA5 reanalysis variables, site and
soil metadata, and optional biosensor traces to predict microbial CO₂ flux and colonization timing.
The model is deliberately compact—an attention-augmented LSTM that can be retrained quickly as new
sites, sensor traces, and input modes become available.

![End-to-end soil-response pipeline](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/fig1_pipeline.png)
*Figure 1. End-to-end soil-response pipeline. ERA5 climate drivers, MLRA / SSURGO / SoilGrids site
context, and optional biosensor traces feed a single 2-layer attention-LSTM that emits per-timestep
CO₂ flux, colonization probability, and attention weights for explainability.*

**Key results at a glance:**

| Evaluation | R² | RMSE (µmol m⁻² s⁻¹) |
| --- | --- | --- |
| Random-split (with attention) | **0.982** | 0.040 |
| Random-split (no-attention ablation) | 0.981 | — |
| Leave-one-year-out (mean ± std) | 0.741 ± 0.076 | — |
| Leave-one-site-out (median, IQR) | 0.445 [−0.023, 0.706] | — |

## 1. The Soil-Response Prediction Problem

ERA5 provides a rich environmental record—soil temperature, volumetric soil moisture, surface solar
radiation, air temperature, and other land-surface variables. But these variables are not the
soil-health indicators that agricultural and carbon workflows actually need. The missing layer is
*biological response*.

Soil microbes respond to climate forcing with lags, thresholds, and site-specific sensitivity. A
moisture pulse after rainfall can produce a respiration event, but timing and magnitude depend on
temperature, soil structure, organic carbon, microbial biomass, and colonization state. A model that
predicts that response must read the climate signal as a time series and learn which parts of recent
history matter most.

This shape of problem—gridded atmospheric forcing on one side, sparse and local biological response
on the other, with lags and site-specific sensitivity—is where attention-augmented LSTMs excel. In
the broader E2 expansion pathway, this model represents the downstream response layer that will
eventually consume site-level forcing derived from DLESym or other weather and climate model
outputs.

## 2. Data Sources

The pipeline uses three complementary data sources.

**Synthetic biosensor traces** provide a controlled cold-start environment. Early biosensor
deployments were still coming online, so synthetic traces gave Team-Soil a way to build and debug
the model architecture, loss functions, and analysis workflow before the field dataset was large
enough for broad validation.

**ERA5-derived site traces** provide real climate forcing. The LSTM-V5 pipeline supports xarray/zarr
training data with ERA5 variables: soil temperature, volumetric soil moisture, surface solar
radiation, and near-surface air temperature.

**Site and soil context** provide the local information that climate variables alone cannot capture.
The current workflow uses 20 USDA MLRA agricultural sites with soil-context data from SSURGO and
SoilGrids.

![Site coverage and representative ERA5 drivers](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/fig2_sites_and_traces.png)
*Figure 2. Left: the 20 USDA MLRA agricultural sites used for training, colored by
leave-one-site-out (LOSO) test R² clipped to [0, 1]; sites with negative LOSO R² are marked with a
red X overlay and discussed in Section 6.2. Right: hourly ERA5 traces (T_soil, θ_soil, SSRD, T_air)
for one representative site.*

![Data inventory](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/table1_data_inventory.png)
*Table 1. Data inventory—variable, source, unit, role, input mode, provenance, and availability at
inference time.*

## 3. Model Architecture

The core model is a **two-layer LSTM** with hidden size 64. A temporal attention head produces a
weight over the encoded sequence, letting the model focus on the parts of the recent time history
most relevant to the predicted response. The network then branches into three task-specific heads: a
**CO₂ flux head**, a **microbial colonization head**, and a **scale-shift component** that helps the
same backbone operate across different flux magnitudes.

![Attention-LSTM architecture](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/fig3_architecture.png)
*Figure 3. Attention-LSTM architecture. Multichannel inputs flow into a 2-layer LSTM (hidden=64),
then through a temporal attention layer to three task heads: CO₂ flux, colonization probability, and
a per-trace scale-shift correction.*

LSTM-V5 adds flexible input modes so the same model family can support multiple experimental
settings via a single `--input-mode` flag:

| Mode | Channels | Notes |
| --- | --- | --- |
| `sensor` | T_soil, θ_soil, MBC_init | Biosensor-style input |
| `era5` | T_soil, θ_soil, SSRD | Base ERA5 climate drivers |
| `era5-extended` | T_soil, θ_soil, SSRD, T_air, doy | **Default**; adds T_air + day-of-year |
| `hybrid` | ERA5 + sensor channels | Combined climate and sensor data |

The model is small by design: easier to retrain as new sites and traces arrive, faster to profile,
and suitable for iterative ablation.

| Parameter | Value |
| --- | --- |
| Architecture | 2-layer LSTM, hidden size 64 |
| Attention | Temporal attention head |
| Task heads | CO₂ flux · colonization probability · scale-shift |
| Default input mode | `era5-extended` |
| Training traces | 300 (90% train / 10% test) |
| CO₂ loss weight | 0.9 |
| Colonization loss weight | 0.1 |

*Table 3. Model and training configuration used for the era5-extended results in this post.*

## 4. From Synthetic Traces to ERA5-Compatible Training

The first stage of the project used synthetic sensor traces to establish the modeling
workflow—validating the basic sequence-to-sequence setup, attention mechanism, CO₂ flux target, and
colonization target, and producing the first analysis tools for plotting predictions, attention
rollouts, and colonization behavior.

The next stage moved the same model family onto ERA5-compatible training data. A data generator
extracts site-level climate traces, joins them with soil and site context, and writes the result
into an xarray/zarr format the LSTM trainer can consume. A subtle early issue: the default generator
path applied a sensor-noise pass that renamed clean ERA5 channels to `T_soil_true` /
`theta_soil_true`; for ERA5-driven training the generator runs with `--no-noise` so the trainer
reads raw climate variables directly.

![Synthetic biosensor vs ERA5 trace comparison](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/fig4_synthetic_vs_era5.png)
*Figure 4. Same backbone, two data worlds. Left: a synthetic biosensor trace and the model's CO₂
flux prediction vs ground-truth target. Right: an ERA5-driven test trace with T_soil and θ as
drivers.*

The result is a general pipeline that can train on synthetic traces for controlled debugging, ERA5
traces for climate-driven experiments, and hybrid traces when both climate context and sensor
information are available. This input-source independence matters for Step 2: ERA5-derived channels
can be replaced by aligned DLESym-derived forcing without changing the model.

## 5. Profiling the Training Bottleneck

After the ERA5 workflow was in place, the bottleneck shifted from model design to iteration speed.
NVIDIA Nsight Systems profiling revealed that the GPU was spending too much time waiting for
data—trace loading and CPU-to-GPU transfer created kernel gaps, leaving accelerator performance on
the table.

The optimization work focused on feeding the GPU more efficiently while keeping model behavior
intact. **The scientific architecture stayed the same; only the surrounding training pipeline became
faster.**

### 5.1 Vectorized Data Loading

The original path loaded traces one at a time, creating many small xarray/zarr reads and significant
Python overhead. The optimized loader reads all traces for a variable in bulk, then slices in
memory—changing the I/O pattern from thousands of small reads to a small number of larger vectorized
reads.

**Result:** Data-loading time dropped from **3.8 s → 0.9 s** per batch—a **4.2× speedup**.

### 5.2 Keeping the GPU Busy

Additional GPU-side optimizations included:

- **Mixed precision training** via Tensor Core acceleration
- **Pinned memory + non-blocking transfers** to remove CPU-to-GPU synchronization stalls
- **CUDA-stream prefetcher** to overlap transfer of the next batch with computation on the current
  batch
- **Persistent DataLoader workers** to reduce repeated worker-startup overhead across epochs

**Combined result:** Epoch time fell from **360 ms → 168 ms** (2.14×); GPU utilization rose from
**~20% → above 80%**; end-to-end training speed improved by **~2.5×** on a single H100.

![Profiling results before and after optimization](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/fig5_profiling.png)
*Figure 5. Profiling results for the opt_v5 training pipeline. Bar plots show data loading time,
epoch time, and GPU utilization before and after the optimization pass.*

![opt_v5 benchmark](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/table4_opt_v5_benchmark.png)
*Table 4. Optimization performance benchmark, including which optimizations are active in each
configuration.*

## 6. Validation Workflow

### 6.1 Random-Split Baseline and Attention Ablation

On the held-out random-split test set (300 traces, 10% test), the model achieves a CO₂ flux R² of
**0.982** and RMSE of **0.040 µmol m⁻² s⁻¹**.

A no-attention ablation run through the same harness with `--no-attention` reaches R² = 0.981.
Within three decimal places the two configurations are indistinguishable—indicating that at the
current 300-trace scale the temporal attention head is not contributing measurable signal beyond
what the LSTM hidden state already encodes, once `ra_fraction` is available as an input channel.
This is a **negative result for the attention module in particular**, but it does not weaken the
rest of the architecture.

That said, attention does concentrate around the moisture and temperature transitions that precede
each flux peak—consistent with what soil microbiologists expect—providing useful *explainability*
even when it is not strictly necessary for fit.

![Prediction and attention rollout for the highest-R² test trace](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/fig6_attention_rollout.png)
*Figure 6. Prediction and attention rollout for the highest-R² test trace. Top: ERA5 drivers (T_soil
in °C, θ_soil in m³/m³). Middle: normalized attention weights over the trace. Bottom: target vs.
predicted CO₂ flux with model-predicted colonization events marked.*

### 6.2 Cross-Validation: Geographic and Temporal Generalization

A random split tests whether the model fits the available distribution; it does not test whether it
generalizes to sites or years absent from training. Leave-one-year-out (LOYO) and leave-one-site-out
(LOSO) cross-validation quantify those gaps. Each fold re-trains from scratch on the remaining
traces.

**Temporal generalization (LOYO).** Across three LOYO folds, mean test R² is **0.741 ± 0.076**
(range 0.641–0.826). Year-to-year variability within a site is much smaller than between-site
variability, and the model recovers roughly three-quarters of the held-out year's variance from
training on only two years.

**Geographic generalization (LOSO).** Across 20 LOSO folds, the distribution of test R² is
heavy-tailed: **median 0.445**, IQR [−0.023, 0.706]; best fold 0.938 (Loess Uplands Iowa); worst
fold −22.4 (Columbia Basin). The arithmetic mean (−1.17) is dominated by the tail and is reported
for completeness only.

**Two distinct failure modes emerge:**

1. **Low-variance R² artefact.** At Columbia Basin, Southern Desertic Basins, and Sacramento/San
   Joaquin Valleys sites, the CO₂ flux signal has very narrow temporal variance (semi-arid or
   Mediterranean-irrigated). The model's RMSE on these folds is 0.017–0.030 µmol m⁻² s⁻¹—comparable
   to or better than the best-generalizing folds—but because SS_tot is tiny in the R² formula, even
   a small SS_res produces a strongly negative score. **These are not prediction failures; they are
   scaling artefacts of R² on low-variance targets.**

2. **Genuine extrapolation failure.** The Central Iowa Till Prairie fold (SOC = 4,800 gC/m², the
   maximum in the training set; RMSE = 0.74) and the Atlantic Coast Flatwoods fold (sand = 70%, a
   texture outlier; RMSE = 0.20) fail because the held-out sites sit at the edge of the training
   data's soil-parameter range with no convex-hull neighbours to interpolate from.

**Headline numbers:** 16 of 20 LOSO folds achieve RMSE < 0.1 µmol m⁻² s⁻¹. 45% of folds reach test
R² > 0.5; 25% reach R² > 0.7.

![Cross-validation and attention ablation results](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/fig7_cv_summary.png)
*Figure 7. Cross-validation and attention ablation. (a) Random-split vs LOSO vs LOYO test R², with
and without attention; error bars are 1σ across folds. (b) Per-site LOSO test R² (one bar per
held-out MLRA site); dashed line marks the LOSO mean. (c) Per-year LOYO test R².*

![Validation benchmark](https://huggingface.co/datasets/nvidia/earth2studio-assets/resolve/blog/blog/soil-health/table5_validation_benchmark.png)
*Table 5. Cross-validation and attention ablation summary. Each row reports the test-set metric mean
across all folds in that group; LOSO and LOYO use mean across held-out sites and years
respectively.*

## 7. Extending the Pipeline

Four concrete extensions follow directly from these results.

### Coupling to DLESym Subseasonal Forecasts (Step 2)

The trainer reads inputs by channel name from xarray/zarr, not by data source. Replacing ERA5
reanalysis with NVIDIA Earth-2 DLESym output requires:

1. Aligning DLESym surface variables (`stl1`, `swvl1`, `ssrd`, `t2m`) to the `T_soil` / `θ_soil` /
   `SSRD` / `T_air` channels the trainer already consumes.
2. Regridding from the DLESym native grid to MLRA site lat/lon coordinates via nearest-neighbor or
   bilinear interpolation.
3. Re-applying the per-channel z-score normalizer persisted at training time.

The existing checkpoint accepts the DLESym-driven input tensor directly, so a **first end-to-end
forecast requires no retraining**. The controlled transfer experiment is structured as: select a
DLESym hindcast window overlapping the 2023–2025 ERA5 training period; extract site time-series;
generate a parallel `training_data_dlesym.zarr`; evaluate the ERA5-trained checkpoint on it as a
zero-shot transfer test.

**Success criterion:** median LOSO R² under DLESym forcing within 0.05 of the ERA5 baseline at
zero-shot, or within 0.02 after a short fine-tune (≤20 epochs, lower learning rate).

### Closing the LOSO Gap

For the low-variance R² artefact, the primary metric to monitor is RMSE—not R²—on arid and
Mediterranean sites. For genuine extrapolation failure, the fix is broadcasting per-trace static
soil-property channels (`clay_pct`, `sand_pct`, `bulk_density`, `SOC_initial`) as constant inputs,
plus a small learned site embedding (4–8 dimensions per `site_id`) concatenated to the LSTM context
vector.

### Recalibrating the Colonization Head

The colonization head collapsed on ERA5 traces (the step-function label produces an always-on tail
that the BCE loss treats as signal). Two targeted changes address this without touching the rest of
the architecture: replace the step-function label with a soft Gaussian centred at the lag, and
increase `colon_loss_weight` from 0.1 to 0.5–1.0.

### Uncertainty and Multi-Task Heads

A variance head trained with Gaussian negative-log-likelihood would provide calibrated 90%
prediction intervals per timestep. Additional regression heads for MBC and SOC, trained jointly
under a weighted multi-task loss, would extend operational utility to carbon accounting workflows.

## Implications for the E2 Agricultural Ecosystem

Many agricultural applications require the same three layers: weather and climate information; local
environmental translation; a domain-specific response model. E2 currently provides the platform
foundation for the first layer; this project explores what must be added downstream to support full
agricultural response modeling.

In such a future ecosystem, E2 could provide the weather-and-climate foundation upon which
customers, partners, researchers, and agricultural-technology organizations build specialized
response models and operational products. This post is the downstream half of that pattern—it builds
the local response layer, measures where it generalizes, identifies where it fails, and defines a
controlled protocol for connecting it to DLESym forcing. Post 2 in this series will make the
upstream half concrete by driving the same trained model with DLESym subseasonal forecasts and
reporting the transfer loss against the ERA5 baseline established here.

## Authors and Acknowledgements

This project is a collaboration between AreaandDee LLC, BioSensor Solutions, and NVIDIA. Rich Loft
leads the machine-learning work at AreaandDee. Sam Walker and David Beitz lead the biosensor work at
BioSensor Solutions. Sepideh Khajehei mentors the project from NVIDIA.
