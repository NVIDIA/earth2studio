"""Compare the E2S CAMulator post-processing chain against the HF-branch CREDIT
fixers + WindPP on identical normalized inputs with the real stats/statics files."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.join(_HERE, "stub"))
from credit.postblock._postblock import (  # noqa: E402
    GlobalEnergyFixerUpDown,
    GlobalMassFixer,
    GlobalWaterFixer,
    TracerFixer,
)
from WindPP import apply_wind_artifact_filter_to_tensor  # noqa: E402

from earth2studio.models.nn.camulator_physics import grid_area  # noqa: E402
from earth2studio.models.px.camulator import (  # noqa: E402
    _TRACERS,
    FORCING_VARIABLES,
    OUTPUT_VARIABLES,
    CAMulator,
    _stats_field,
)
from earth2studio.lexicon.camulator import CAMulatorLexicon  # noqa: E402

T = os.environ.get("CAMULATOR_ASSETS", os.path.join(_HERE, "assets"))
MEAN, STD = f"{T}/mean.nc", f"{T}/std.nc"
PHYS = f"{T}/statics_b.nc"
torch.manual_seed(0)
H, W = 192, 288

# ---------------- CREDIT side (config as the parser would emit) ----------------
post_conf = {
    "activate": True,
    "tracer_fixer": {
        "activate": True,
        "denorm": True,
        "mean_path": MEAN,
        "std_path": STD,
        "tracer_inds": [c for c, _, _, _ in _TRACERS],
        "tracer_thres": [lo for _, _, lo, _ in _TRACERS],
        "tracer_thres_max": [None if hi == float("inf") else hi for _, _, _, hi in _TRACERS],
        "tracer_var_names": [n for _, n, _, _ in _TRACERS],
    },
    "global_mass_fixer": {
        "activate": True, "activate_outside_model": True, "fix_level_num": 14, "simple_demo": False,
        "denorm": True, "grid_type": "sigma", "midpoint": True,
        "lon_lat_level_name": ["lon2d", "lat2d", "hyai", "hybi"], "save_loc_physics": PHYS,
        "mean_path": MEAN, "std_path": STD, "q_inds": [96, 127], "sp_inds": 128,
    },
    "global_water_fixer": {
        "activate": True, "activate_outside_model": True, "simple_demo": False, "denorm": True,
        "grid_type": "sigma", "midpoint": True, "lon_lat_level_name": ["lon2d", "lat2d", "hyai", "hybi"],
        "save_loc_physics": PHYS, "mean_path": MEAN, "std_path": STD, "lead_time_periods": 6,
        "q_inds": [96, 127], "sp_inds": 128, "precip_ind": 130, "evapor_ind": 138,
    },
    "global_energy_fixer_updown": {
        "activate": True, "activate_outside_model": True, "simple_demo": False, "denorm": True,
        "grid_type": "sigma", "midpoint": True, "lead_time_periods": 6, "save_loc_physics": PHYS,
        "mean_path": MEAN, "std_path": STD, "lon_lat_level_name": ["lon2d", "lat2d", "hyai", "hybi"],
        "surface_geopotential_name": ["PHIS"], "T_inds": [64, 95], "q_inds": [96, 127],
        "U_inds": [0, 31], "V_inds": [32, 63], "sp_inds": 128, "TOA_forcing_solar_ind": 132,
        "TOA_up_solar_ind": 145, "TOA_up_OLR_ind": 146, "surf_down_solar_ind": 139,
        "surf_up_solar_ind": 141, "surf_down_LW_ind": 140, "surf_up_LW_ind": 142,
        "surf_SH_ind": 143, "surf_LH_ind": 144,
    },
}
conf = {
    "data": {"variables": ["U", "V", "T", "Qtot"]},
    "model": {"levels": 32},
    "postprocessing": {
        "wind_artifact_filter": {
            "activate": True, "mask_level": 14, "target_levels": list(range(9, 21)),
            "target_vars": ["U", "V", "T", "Qtot"], "speed_threshold": 2.8, "smooth_sigma": 1.2,
            "smooth_sigma_zonal": 2.0, "smooth_sigma_meridional": 0.5, "dilation_zonal": 15,
            "dilation_meridional": 5, "falloff_sigma": 4.0, "preserve_amplitude": True,
        }
    },
}
tracer = TracerFixer(post_conf)
mass = GlobalMassFixer(post_conf)
water = GlobalWaterFixer(post_conf)
energy = GlobalEnergyFixerUpDown(post_conf)

# ---------------- E2S side ----------------
mean_ds, std_ds, phys = xr.open_dataset(MEAN), xr.open_dataset(STD), xr.open_dataset(PHYS)
center = torch.from_numpy(np.stack([_stats_field(mean_ds, v, (H, W)) for v in OUTPUT_VARIABLES]))
scale = torch.from_numpy(np.stack([_stats_field(std_ds, v, (H, W)) for v in OUTPUT_VARIABLES]))
fc = torch.tensor([float(mean_ds[CAMulatorLexicon[v][0]]) for v in FORCING_VARIABLES])
fs = torch.tensor([float(std_ds[CAMulatorLexicon[v][0]]) for v in FORCING_VARIABLES])
tc = torch.tensor([float(np.asarray(mean_ds[n].values).flatten()[0]) for _, n, _, _ in _TRACERS])
ts = torch.tensor([float(np.asarray(std_ds[n].values).flatten()[0]) for _, n, _, _ in _TRACERS])
lat2d = torch.from_numpy(phys["lat2d"].values.astype(np.float32))
lon2d = torch.from_numpy(phys["lon2d"].values.astype(np.float32))
model = CAMulator(
    torch.nn.Identity(), center, scale, fc, fs, tc, ts, torch.zeros(2, H, W),
    torch.from_numpy(phys["hyai"].values.astype(np.float32)),
    torch.from_numpy(phys["hybi"].values.astype(np.float32)),
    grid_area(lat2d, lon2d), torch.from_numpy(phys["PHIS"].values.astype(np.float32)),
    forcing_data_source=object(),
)

# Normalized random inputs near climatology (B=1 because WindPP assumes it)
x = 0.3 * torch.randn(1, 136, 1, H, W)
y = 0.3 * torch.randn(1, 147, 1, H, W)
y[:, 96:128] = y[:, 96:128].abs() * -1  # force some negative tracers to exercise the clip

# CREDIT chain: tracer (inside model) -> wind -> mass -> water -> energy
y_ref = y.clone()
y_ref = tracer({"y_pred": y_ref})["y_pred"]
apply_wind_artifact_filter_to_tensor(
    y_ref, ["U", "V", "T", "Qtot"], 32, mask_level=14, target_levels=range(9, 21),
    target_vars=["U", "V", "T", "Qtot"], speed_threshold=2.8, smooth_sigma=1.2,
    dilation_zonal=15, dilation_meridional=5, falloff_sigma=4.0, preserve_amplitude=True,
    smooth_sigma_zonal=2.0, smooth_sigma_meridional=0.5,
)
y_ref = mass({"y_pred": y_ref, "x": x})["y_pred"]
y_ref = water({"y_pred": y_ref, "x": x})["y_pred"]
y_ref = energy({"y_pred": y_ref, "x": x})["y_pred"]

# E2S chain
y_e2s = model._postprocess(x[:, :, 0].clone(), y[:, :, 0].clone()).unsqueeze(2)

diff = (y_ref - y_e2s).abs()
for name, sl in [("U", slice(0, 32)), ("V", slice(32, 64)), ("T", slice(64, 96)), ("Qtot", slice(96, 128)),
                 ("PS", 128), ("TREFHT", 129), ("PRECT", 130), ("diag", slice(131, 147))]:
    print(f"{name:7s} max|diff|={diff[:, sl].max().item():.3e}  ref scale={y_ref[:, sl].abs().mean().item():.3e}")
print("changed by chain (ref vs raw y):", (y_ref - y).abs().max().item())
assert torch.allclose(y_ref, y_e2s, atol=1e-5, rtol=1e-4), "MISMATCH"
print("PHYSICS CHAIN EQUIVALENT")
