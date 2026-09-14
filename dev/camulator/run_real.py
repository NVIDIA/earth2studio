"""Real-checkpoint check: load CAMulator from HuggingFace (4.8 GB download on first
run), build a physical initial condition from the shipped 1981-01-01 IC tensor and
roll out N 6 h steps. Prints finiteness, PS sanity and peak memory per step.

    DEVICE=cuda STEPS=8 uv run python dev/camulator/run_real.py

Assumption (unverified against a CREDIT run): the shipped IC tensor is the
*normalized* model input, channels 0..129 = prognostic state in south-to-north
latitude order, same layout as the wrapper's center/scale buffers. The PS
range check below fails loudly if that is wrong.
"""
import os
import resource
import time as _time

import numpy as np
import torch

from earth2studio.models.px.camulator import CAMulator

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.environ.get("CAMULATOR_ASSETS", os.path.join(HERE, "assets"))
DEVICE = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
STEPS = int(os.environ.get("STEPS", "8"))
FIXERS = os.environ.get("FIXERS", "1") == "1"
WIND = os.environ.get("WIND", "1") == "1"


def rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1e9 if os.uname().sysname == "Darwin" else 1e6)


t0 = _time.time()
model = CAMulator.load_model(
    CAMulator.load_default_package(), conservation_fixers=FIXERS, wind_filter=WIND
).to(DEVICE)
print(f"loaded in {_time.time() - t0:.0f}s, RSS {rss_gb():.1f} GB, device {DEVICE}")

ic = torch.load(os.path.join(ASSETS, "ic_1981.pth"), map_location="cpu", weights_only=False)
assert ic.shape == (1, 136, 1, 192, 288), ic.shape
state_n = ic[0, :130, 0].to(DEVICE)  # normalized, S->N
center, scale = model.center[:130], model.scale[:130]
state = state_n * scale + center  # physical, S->N
state = torch.flip(state, dims=(-2,))  # E2S wants N->S
ps = state[128]
print(f"PS physical: min {ps.min():.0f} max {ps.max():.0f} mean {ps.mean():.0f} Pa")
assert 40_000 < ps.min() and ps.max() < 110_000, "PS out of range: IC layout assumption wrong"

x = state[None, None, None]  # [batch=1, time=1, lead=1, 130, H, W]
coords = model.input_coords()
coords["batch"] = np.array([0])
coords["time"] = np.array([np.datetime64("1981-01-01T00:00")])

it = model.create_iterator(x, coords)
for i, (y, c) in enumerate(it):
    y = y.cpu()
    prog = y[..., :130, :, :]
    msg = (
        f"step {i:3d} lead {c['lead_time'][0]}  finite(prog)={torch.isfinite(prog).all().item()} "
        f"finite(all)={torch.isfinite(y).all().item()}  "
        f"PS[{y[0,0,0,128].min():.0f},{y[0,0,0,128].max():.0f}]  "
        f"T0k mean {y[0,0,0,64].mean():.1f}K  RSS {rss_gb():.1f} GB"
    )
    if DEVICE.type == "cuda":
        msg += f"  CUDA peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    print(msg, flush=True)
    if i >= STEPS:
        break
