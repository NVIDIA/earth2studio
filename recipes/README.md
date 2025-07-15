<!-- markdownlint-disable MD007 -->
# Earth2Studio Cookbook

Welcome to the Earth2Studio cookbook.
This is a collection of different recipes that solve common problems or use cases.
Recipes are **not** designed to be turnkey solutions, but rather reference
boilerplate for users to copy and modify for their specific needs.
Many recipes involve more complex workflows that are too specific, complex or resource
intensive for constitute upstreaming into the broader package.

> [!NOTE]
> If you are new to Earth2Studio and are looking for samples to get started with, first
> visit the [examples page](https://nvidia.github.io/earth2studio/examples/index.html)
> in the documentation.

Recipes are not installable packages.
Users are expected to clone the repository and then interact with the source files
directly.

> [!WARNING]
> Earth2Studio recipes are in beta, thus may evolve as updates are added.

## Recipe Prerequisites

- Basic knowledge / understanding of Earth2Studio APIs
- Background on the particular recipe usecase
- Intermediate Python knowledge
- Any specific hardware requirements listed

## Index

- [Huge Ensembles (HENS)](./hens/)

    This recipe implements a multi-checkpoint inference pipeline for large-scale
    ensemble weather forecasting using the HENS (Huge Ensembles) method, which enables
    parallel processing of multiple model checkpoints for uncertainty quantification in
    weather prediction systems. The pipeline supports multi-GPU inference, tropical
    cyclone tracking, diagnostic models, and regional output.

    - Difficulty: Advanced
    - Inference Compute Type: Multi-GPU
    - Estimated Runtime: 2 minutes to 12+ hours

- [Subseasonal-to-Seasonal (S2S)](./s2s/)

    This recipe demonstrates how to run ensemble forecasts for subseasonal-to-seasonal
    (S2S) timescales using Earth2Studio, bridging the gap between weather forecasts
    (up to 2 weeks) and seasonal forecasts (3-6 months). The recipe supports multi-GPU
    distributed inference, parallel I/O using zarr format, diagnostic models, storage
    space reduction via regional output, and scoring capabilities including ECMWF AIWQ
    S2S metrics, with DLESyM and HENS-SFNO models being best-suited for S2S forecasting.

    - Difficulty: Advanced
    - Inference Compute Type: Multi-GPU
    - Estimated Runtime: 10 minutes to 2 hours

- [Recipe Template](./template/)

    Recipe template for developers.
