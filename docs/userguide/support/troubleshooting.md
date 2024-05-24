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

Still having some problems? Open an issue.
