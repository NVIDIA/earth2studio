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
