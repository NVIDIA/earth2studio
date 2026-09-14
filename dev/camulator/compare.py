import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

import torch

sys.path.insert(0, _HERE)
from credit_camulator import Camulator as CreditCam  # noqa: E402

from earth2studio.models.nn.camulator import CamulatorNet  # noqa: E402

torch.manual_seed(0)
kw = dict(
    image_height=192,
    image_width=288,
    channels=4,
    surface_channels=2,
    input_only_channels=6,
    output_only_channels=17,
    levels=32,
    dim=(8, 16, 32, 64),
    depth=(1, 1, 2, 1),
    dim_head=8,
    global_window_size=(4, 4, 2, 1),
    local_window_size=3,
    cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
    cross_embed_strides=(2, 2, 2, 2),
    use_spectral_norm=True,
    interp=True,
)
ref = CreditCam(
    frames=1,
    patch_height=1,
    patch_width=1,
    padding_conf={"activate": True, "mode": "earth", "pad_lat": [48, 48], "pad_lon": [48, 48]},
    post_conf={"activate": False},
    **kw,
).eval()
mine = CamulatorNet(pad_lat=(48, 48), pad_lon=(48, 48), **kw).eval()
sd_ref = ref.state_dict()
sd_mine = mine.state_dict()
kr, km = set(sd_ref), set(sd_mine)
print("keys ref-only:", sorted(kr - km)[:10])
print("keys mine-only:", sorted(km - kr)[:10])
assert kr == km, "state dict key mismatch"
for k in kr:
    assert sd_ref[k].shape == sd_mine[k].shape, (k, sd_ref[k].shape, sd_mine[k].shape)
mine.load_state_dict(sd_ref, strict=True)
x = torch.randn(2, 136, 1, 192, 288)
with torch.no_grad():
    yr = ref(x)
    ym = mine(x)
print("shapes", yr.shape, ym.shape)
print("max abs diff", (yr - ym).abs().max().item(), "ref scale", yr.abs().mean().item())
assert torch.allclose(yr, ym, atol=1e-5, rtol=1e-5)
print("EQUIVALENT")
