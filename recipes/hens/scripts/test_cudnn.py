import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(1, 3, 128, 128, device="cuda")
layer = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
y = layer(x)

print("cuDNN convolution succeeded")
print("Output shape:", tuple(y.shape))
