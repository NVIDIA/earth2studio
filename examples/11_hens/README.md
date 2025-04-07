# Parallel Inference using Multiple Checkpoints (Python)

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

``` bash
python hens.py
```

To run on multiple GPUs using your personal config file, do

``` bash
mpirun -n 2 python hens.py --config-name=custom_config.yaml
```
