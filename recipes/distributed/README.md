# Earth2Studio Distributed Inference Recipe

This recipe shows how to use the `DistributedInference` interface to distribute inference workloads
to multiple GPUs in a distributed computing environment (e.g. `torchrun` or MPI).

## Prerequisites

### Software

Installing Earth2Studio and [Hydra](https://hydra.cc/docs/intro/) is sufficient for running the
recipe. The commands below in Quick Start will install a tested environment.

### Hardware

- GPUs: Any type with >= 20 GB memory, at least 2 GPUs required to run the recipe
- Storage: A few GB to store inference results and model checkpoints.

## Quick Start

### Installation

Installing Earth2Studio is generally a sufficient prerequisite to use this recipe. The support
for models used by the recipe must be included in the installation. For the diagnostic model
example, this means installing Earth2Studio with

```bash
pip install earth2studio[fcn,precip-afno]
```

To install a full tested environment, you can use pip:

```bash
pip install -r requirements.txt
```

or set up a uv virtual environment:

```bash
uv sync
```

### Test distributed inference

Start an environment with at least 2 GPUs available. The run the distributed diagnostic model
example, substituting <NUMBER_OF_GPUS> with the number of GPUs you have:

```bash
# if you installed a uv environment
uv run torchrun --standalone --nnodes=1 --nproc-per-node=<NUMBER_OF_GPUS> main.py --config-name=diagnostic.yaml

# using default python
torchrun --standalone --nnodes=1 --nproc-per-node=<NUMBER_OF_GPUS> main.py --config-name=diagnostic.yaml
```

## Documentation

### Using the recipes

Specify the recipe you want to run using the `--config-name` command line argument to `main.py`.
This is used to select the relevant function in `main.py`. Currently, only `diagnostic.yaml` is
provided; more recipes will be added later.

The configuration of the recipes is managed with Hydra using YAML config files located in the
`cfg` directory. You can override default options by editing the config file, or from the command
line using the Hydra syntax: for example, to save the diagnostic model recipe output to
`output_file.zarr`:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=<NUMBER_OF_GPUS> main.py\
    --config-name=diagnostic.yaml ++parameters.output_path=output_file.zarr
```

### Supported distribution methods

In a single-node environment, we recommend using `torchrun`.

`DistributedInference` should also work with any distribution method supported by the
[`DistributedManager`](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.distributed.html)
of PhysicsNeMo. The startup commands will need to be modified to the distribution. For instance,
an MPI job using 2 GPUs on a single node can be started with Slurm using a script:

```bash
cd <EARTH2STUDIO_PATH>/recipes/distributed/
mpirun --allow-run-as-root python main.py --config-name=diagnostic.yaml
```

which can then be launched with
`srun --nodes=1 --ntasks-per-node=2 --gpus-per-node=2 <PATH_TO_SCRIPT>`,
replacing `<EARTH2STUDIO_PATH>` with the path where Earth2Studio is located and `<PATH_TO_SCRIPT>`
with the startup script path.

### Creating custom applications

To create custom applications using `DistributedInference`, you can use the provided recipes as a
starting point.

## References

- [PyTorch TensorPipe CUDA RPC](https://docs.pytorch.org/tutorials/recipes/cuda_rpc.html), the
PyTorch feature used to implement `DistributedInference`.
