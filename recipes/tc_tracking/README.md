# FCN# cyclone tracking

Baseline to acheive this is an ensembling setup. Due to the inherently stochastic nature of the model, initial condition perturbation is not required.

## Setting up the environment

There are two ways to set up the environment, either directly using an uv environment or using uv inside a container.
Since torch-harmonics has to be compiled and TempestExtreme is a commandline tool, the container is the recommended way.

1. **container**

    There is a Dockerfile provided in the top level directory of the repository.

2. **uv**

    The project contains a pyproject.toml and a uv.lock file for setting up the envirnment. All you need to do is run:

    ```bash
    uv sync --frozen
    ```

    This will create a uv environment in the .venv directory. The fronzen flag will make sure the exact version specified in the lock file is used.
    Optionally, trigger the virtual environment with:

    ```bash
    source .venv/bin/activate
    ```

    **issues with torch-harmoics**
    if `--frozen` flag in `uv sync` cannot be used eg for updating the dependencies to the latest versions, it can happen that torch harmonics fails during dependency resolution phase. In that phase, uv has to build th to resolve dependencies, but does not yet install it. In these cases it can happen that th build fails as other dependencies are not installed yet, Installation of packages happens in a second phase, the install phase. Additionally, there is a git_lfs error that appears every now and then. To adress both or either, split the installation up in a two-step process and skip GIT_LFS:
    ```bash
    GIT_LFS_SKIP_SMUDGE=1 uv sync --no-install-package torch-harmonics
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    ```



## Run

The project contains a script for running the code.

```bash
python tc_hunt.py --config-name=tc_selection_ens.yaml
```

## Modes

### extract Reference tracks from ERA5 using IBTrACS as ground truth

```yaml
mode: 'extract_baseline'
```

IBTrACS is used as ground truth for the reference tracks. However, due to various reasons simulation data is compared against extracting the tracks directly from ERA5...

### generate ensemble

```yaml
mode: 'generate_ensemble'
```

### reproduce individual members

```yaml
mode: 'reproduce_members'
```

Only works if exact same batch is reproduced, that means:
- produces full batches
- important to set batch size
- ensemble size required if final batch of ensemble shall be reproduced
- random seed has to be provided for each ensemble member, but the global random seed won't have effect here, as it mainly impacts the generation of random seeds for members
- works only on same machine using identical environment. No guarantees otherwise
