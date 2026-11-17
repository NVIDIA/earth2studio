---
name: earth2studio-get-started
description: >
  Use when getting started with Earth2Studio or setting up a first forecast,
  from a Python environment through UCast inference, Zarr output, and plotting.
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
---

# Earth2Studio - Get Started

Keep replies brief. Guide one stage at a time; skip questions already answered.
After each successful stage, explicitly offer the next step rather than ending at setup.

## 1. Confirm the version and offer the example

Ask which Earth2Studio version to use, unless already specified: **latest release,
main, or a specific version?** Follow `earth2studio-install` to resolve the latest
stable release and present the version choices. Wait for the choice and carry it
through setup without asking again. Use matching-version sources and documentation.
Then ask:
**"Do you want to get started with a minimal runnable example?"** Explain briefly:
"We'll initialize UCast from ARCO ERA5 and save a five-day forecast to Zarr."
Wait for the answer. If no, stop; if yes, continue with environment setup.

## 2. Set up the environment

Read the selected
version's install options, `pyproject.toml`, `ucast.py`, and deterministic example.
Confirm `UCast` and the `ucast` extra exist before giving commands.

Ask: **"Are you already in a Python project, and does it use uv?"** Recommend uv.
Check GPU availability. A GPU with 40 GB of memory is recommended for UCast;
depending on the workflow, it may run on GPUs with less memory.
Model loading downloads weights and WeatherBench2 static fields.

Use `earth2studio-install` for version-specific commands and troubleshooting. Give commands
for the user to execute, then wait for their result. For a new uv project:

```bash
mkdir first-forecast && cd first-forecast
uv init --python=3.13
uv add "earth2studio[ucast] @ git+https://github.com/NVIDIA/earth2studio.git@<ref>"
uv run python -c "import earth2studio, torch; print(earth2studio.__version__, torch.cuda.is_available())"
```

Replace `<ref>` with the selected release tag or `main` before presenting commands;
use the selected branch for an explicit unreleased RC. Main may change incompatibly.
Use that version's supported Python. For an existing uv project, omit `mkdir` and
`uv init`; preserve its Python pin. For pip projects, give the matching pip command.
The current `ucast` extra is empty; it needs no additional model-specific dependencies.

Once the user confirms setup and verification succeeded, continue to the forecast step.
If the environment is already ready, skip installation and proceed directly.

## 3. Save and run the forecast

Use ARCO ERA5 for UCast's initial conditions, including sea surface temperature (`sst`).

ARCO ERA5 is delayed: do not use yesterday's date without verifying availability for
both initialization and its 12-hour history. Default to the historical example below;
it forecasts five days **from that date**, not the next five days from today.
For a verified recent date, round UTC down to a six-hour cycle and ensure both times exist.

Show this as `forecast.py`, adapting APIs to the selected version:

```python
import torch
from earth2studio.data import ARCO_ERA5
from earth2studio.io import ZarrBackend
from earth2studio.models.px import UCast
from earth2studio.run import deterministic

model = UCast.load_model(UCast.load_default_package())
store = ZarrBackend("ucast.zarr")
deterministic(["2023-01-01T00:00:00"], 10, model, ARCO_ERA5(), store,
              device=torch.device("cuda"))
```

UCast steps every 12 hours: 10 steps = five days, 20 = ten days. The single-member
workflow still uses UCast's default stochastic dropout. Its inputs include two times;
the workflow fetches that history. Use a fresh store path for a new run.

Give `uv run python forecast.py` (or their environment's Python) and ask:
**"Would you like to run the forecast now?"** If no, stop with the script and command
ready. If yes, have the user run the command and share the result. Wait for completion;
do not claim output exists before the user confirms or you verify it. On failure,
diagnose the error before moving on. Once successful, offer post-processing immediately.

## 4. Offer post-processing

After the store is saved, ask **"Would you like to do any post-processing, such as plotting?"**
If no, stop. If yes, ask **"What would you like to plot from this?"** Offer:
"Plot 2-metre temperature contours at day five." Let the user choose the variable,
lead time, and region; do not assume their answer. Inspect the store's variables and
coordinates before selecting. `t2m` is near-surface air temperature, not skin temperature.

Once chosen, provide a separate `plot_forecast.py` using Matplotlib and xarray.
If Matplotlib is missing, give `uv add matplotlib` and wait. For the example request:

```python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ds = xr.open_zarr("ucast.zarr", consolidated=False)
field = ds["t2m"].isel(time=0).sel(lead_time=np.timedelta64(120, "h")) - 273.15
field.plot.contourf(x="lon", y="lat", levels=20, cbar_kwargs={"label": "°C"})
init = ds.time.values[0]
valid = init + np.timedelta64(120, "h")
plt.title(f"UCast 2-metre temperature\nInit {init}; valid {valid} UTC (+120 h)")
plt.savefig("temperature.png", dpi=150, bbox_inches="tight")
plt.close()
ds.close()
```

Give `uv run python plot_forecast.py`. Adapt selectors to the actual store schema;
select a forecast lead, not lead zero, and label units and initialization/valid time.
Use Matplotlib alone for the first plot; add mapping libraries only if requested.
