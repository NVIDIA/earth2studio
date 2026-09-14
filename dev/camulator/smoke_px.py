import os
import resource

import numpy as np
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.data.camulator import CAMULATOR_GRID_LAT, CAMULATOR_GRID_LON
from earth2studio.models.nn.camulator import CamulatorNet
from earth2studio.models.nn.camulator_physics import grid_area
from earth2studio.models.px.camulator import CAMulator, _TRACERS

torch.manual_seed(0)
H, W = 192, 288
net = CamulatorNet(dim=(8, 16, 32, 64), depth=(1, 1, 1, 1), dim_head=8).eval()
lat2d, lon2d = np.meshgrid(np.linspace(-90, 90, H), np.linspace(0, 358.75, W), indexing="ij")
hyai = torch.linspace(0.0, 400.0, 33)
hybi = torch.linspace(0.0, 1.0, 33) ** 2
model = CAMulator(
    net,
    center=torch.zeros(147, H, W),
    scale=torch.ones(147, H, W),
    forcing_center=torch.zeros(4),
    forcing_scale=torch.ones(4),
    tracer_center=torch.zeros(len(_TRACERS)),
    tracer_scale=torch.ones(len(_TRACERS)),
    statics=torch.zeros(2, H, W),
    hyai=hyai,
    hybi=hybi,
    area=grid_area(torch.tensor(lat2d, dtype=torch.float32), torch.tensor(lon2d, dtype=torch.float32)),
    phis=torch.zeros(H, W),
    forcing_data_source=Random({"lat": CAMULATOR_GRID_LAT, "lon": CAMULATOR_GRID_LON}),
    conservation_fixers=False,  # random state -> unphysical budgets
    wind_filter=True,
).to(torch.device(os.environ.get("DEVICE", "cpu")))
time = np.array([np.datetime64("2001-01-01T00:00"), np.datetime64("2001-07-01T06:00")])
dc = model.input_coords()
for k in ("batch", "time", "lead_time", "variable"):
    del dc[k]
x, coords = fetch_data(Random(dc), time, model.input_coords()["variable"], model.input_coords()["lead_time"])
print("x", x.shape)
out, oc = model(x, coords)
print("out", out.shape, oc["lead_time"], len(oc["variable"]), oc["lat"][:2], torch.isfinite(out).all().item())
it = model.create_iterator(x.unsqueeze(0).repeat(2, 1, 1, 1, 1, 1), {**{"ensemble": np.arange(2)}, **coords})
for i, (y, c) in enumerate(it):
    print(i, y.shape, c["lead_time"], torch.isnan(y).any().item())
    if i == 2:
        break
# fixers on a "physical" random state
model.conservation_fixers = True
model.center.copy_(torch.zeros(147, H, W))
model.center[64:96] = 250.0
model.center[96:128] = 1e-3
model.center[128] = 1e5
model.scale[96:128] = 1e-4
model.scale[128] = 100.0
model.scale[64:96] = 1.0
model.center[129] = 280.0
out2, _ = model(x, coords)
print("fixers finite:", torch.isfinite(out2).all().item())

print("peak RSS GB:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if os.uname().sysname == "Darwin" else 1e6))
if torch.cuda.is_available():
    print("peak CUDA GB:", torch.cuda.max_memory_allocated() / 1e9)
