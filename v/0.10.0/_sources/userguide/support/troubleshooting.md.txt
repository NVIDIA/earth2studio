# Troubleshooting

## ONNX Runtime  Error when binding input

When running an ONNX based model, such as FengWu or Pangu, one may see a runtime error
where the model fails to bind input data when using a GPU. The error message may look
like.
> RuntimeError: Error when binding input: There's no data transfer registered for
>copying tensors from Device:[DeviceType:1 MemoryType:0 DeviceId:0]

or
> onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 :
>FAIL : Failed to load library libonnxruntime_providers_cuda.so with error:
> libcublasLt.so.11: cannot open shared object file: No such file or directory.

This is an error from ONNX runtime not being installed correctly.
If you are using CUDA 12 make sure you manually pip install following the instructions
on the ONNX [documentation](https://onnxruntime.ai/docs/install/#python-installs).
You may need to manually link the needed libraries, see this
[Github issue](https://github.com/microsoft/onnxruntime/issues/19616) for reference.

## ImportError: object is not installed, install manually or using pip

This is an error that arises typically when the proper optional dependencies are not
installed on the system.
For example:

```bash
>>> from earth2studio.data import CDS
>>> CDS()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/earth2studio/earth2studio/data/cds.py", line 90, in __init__
    raise ImportError(
ImportError: cdsapi is not installed, install manually or using `pip install earth2studio[data]`
```

The error message should indicate what the install group is that needs to be added.
In the above example, running the command:

```bash
uv pip install earth2studio[data]
# Or with pip
pip install earth2studio[data]
# Or if you are developer
uv sync --extra data
```

will fix the problem.
For additional information refer to the {ref}`optional_dependencies` section.

## Earth2Studio not authorized to download public models from NGC

Earth2Studio will attempt to use NGC CLI based authentication to download models.
Sometimes credential misconfiguration can impact access to even public models with
potential errors like:

```bash
ValueError: Invalid org. Choose from ['no-org', '0123456789']
# or
ValueError: Invalid team. Choose from ['no-team', '0123456789']
```

In these cases it's typically because there is an NGC API key on the system either via
the NGC config file located at `~/.ngc/config` by default or by environment variable
`NGC_CLI_API_KEY`.

One solution is to rename your config file or unset the API key environment variable so
Earth2Studio uses guest access.
Otherwise one can modify the config / environment variables to provide the needed
information.
For example:

```bash
export NGC_CLI_ORG=no-org
export NGC_CLI_TEAM=no-team
```

For more information see the [NGC CLI docs](https://docs.ngc.nvidia.com/cli/index.html).

Still having some problems? Open an issue.

## Flash Attention has long build time for AIFS

This is a known issue with the library with several [issues](https://github.com/Dao-AILab/flash-attention/issues/1038)
on the subject.
There are a few options to try outside of just waiting for the build to complete.

1. If using a docker container is possible, the PyTorch docker container on NGC has
  flash attention already built inside of it. See {ref}`pytorch_container_environment`
  for details on how to install Earth2Studio inside a container.

2. Speed up the compile time by increasing the number of jobs used during the build
  process. The upper limit depends on the systems memory, too large may result in
  a crash:

    ```bash
    # Ninja build jobs, increase depending on system memory
    export MAX_JOBS=8
    ```

3. Disable unused features in the library not needed for inference:

    ```bash
    # https://github.com/Dao-AILab/flash-attention/issues/1486
    export FLASH_ATTENTION_DISABLE_HDIM128=FALSE
    export FLASH_ATTENTION_DISABLE_CLUSTER=FALSE
    export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
    export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
    export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
    export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
    export FLASH_ATTENTION_DISABLE_FP16=TRUE
    export FLASH_ATTENTION_DISABLE_FP8=TRUE
    export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
    export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
    export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
    export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
    ```

## Failed to Install Earth2Grid `Python.h: No such file or directory`

[Earth2Grid](https://github.com/NVlabs/earth2grid) needs to be installed from source.
To build this dependency, the Python developer library is needed, on debian systems this
can be installed with:

```bash
sudo apt-get install python3-dev
```

## Torch Harmonics has long build time for FCNv3

This is a known challenge when building torch harmonics with cuda extensions which
require the compilation of discrete-continuous (DISCO) convolutions.
One method to speed up the install process is to limit the [cuda architectures](https://developer.nvidia.com/cuda-gpus)
that are built to the specific card being used.
For example, to compile for just Ada Lovelace and newer architectures, set the
following environment variables before installing:

```bash
export FORCE_CUDA_EXTENSION=1
export TORCH_CUDA_ARCH_LIST="8.9 9.0 10.0 12.0+PTX"
```

See the [torch harmonics repo](https://github.com/NVIDIA/torch-harmonics) for more
information.
If torch harmonics is already installed, you may need to force a re-install to build
the cuda extensions:

```bash
pip install --no-build-isolation --force-reinstall --upgrade --no-deps \
  --no-cache  --verbose torch-harmonics==0.8.0
# Or respective uv command
```
