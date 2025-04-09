# Recovering HENS through Parallel Inference using Multiple Checkpoints (Python)

## Table of Contents

- [1. Pipeline overview](#1-pipeline-overview)
- [2. Prerequisites](#2-prerequisites)
- [3. Configuring the pipeline](#3-configuring-the-pipeline)
- [4. Executing the pipeline](#4-executing-the-pipeline)
- [5. Reference Workflows](#5-reference-workflows)
  - [5.1 Hurricane Helene](#51-hurricane-helene)
  - [5.2 Reproducing individual Batches of the Helene Ensemble](#52-reproducing-individual-batches-of-the-helene-ensemble)
  - [5.3 Precipitation Forecast](#53-precipitation-forecast)

## 1. Pipeline overview

- This project shows capabilities of multi-checkpoint inference pipeline
- Can recover HENS method described in: [Huge Ensembles Part I: Design of Ensemble Weather Forecasts
  using Spherical Fourier Neural Operators](https://arxiv.org/abs/2408.03100)
- HENS is
  - Most important features
    - Parallel
    - Reproducibility
    - Add diagnostic models
    - Add TC tracking
    - Regional output for saving data

## 2. Prerequisites

- Download model packages
- Download skill
- Download data [optional]

## 3. Configuring the pipeline

- Most important settings in config

## 4. Executing the pipeline

- Sequentially
- In parallel

This workflow implements a method for running large-scale inference in a highly
distributed setting. Model uncertainty is reflected by using multiple checkpoints.
The workflow is highly customisable through the config file, by configuring key
components including the forecast model and initial condition perturbation strategies.
The method described in [Huge Ensembles Part I: Design of Ensemble Weather Forecasts
using Spherical Fourier Neural Operators](https://arxiv.org/abs/2408.03100) can be
recovered by using the `HemisphericCentredBredVector` perturbation.

Additionally, diagnostic models and tropical cyclone tracking can be added to the
pipeline by setting `mode` to `diagnostic` or to `cyclone_tracking` in the config.

To run an inference, tailor the config file to your needs and do

```bash
python hens.py
```

To run on multiple GPUs using your personal config file, do

```bash
mpirun -n 2 python hens.py --config-name=custom_config.yaml
```

## 5. Reference Workflows

### 5.1 Hurricane Helene

- Does 234583 member ensemble of [Hurricane Helene](https://en.wikipedia.org/wiki/Hurricane_Helene)
- Use config `helene.yaml`
- Run in parallel, use up to X GPUs

### 5.2 Reproducing individual Batches of the Helene Ensemble

### 5.3 Precipitation Forecast

- Shows use of diagnostic models
- Various diagnostic models can be used simultaneously
- Order is important
