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
You may need to manally link the need libraries, see this
[Github issue](https://github.com/microsoft/onnxruntime/issues/19616) for reference.

## ImportError: object is not installed, install manually or using pip

This is an error that arrises typically when the proper optional dependencies are not
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
