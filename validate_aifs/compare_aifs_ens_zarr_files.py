"""Compare AIFS-ENS outputs from Earth2Studio and the Anemoi reference run.

This script expects two Zarr stores: one produced by the Earth2Studio workflow
(`aifs_ens_e2_example.py`) and one produced by the Anemoi runner
(`aifs_ens_anemoi_example.py`). Each comparison plot includes the two fields and
their difference for every time step they have in common.
"""

from __future__ import annotations

import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import LinearSegmentedColormap

E2_ZARR_FILE = Path(os.environ.get("E2_ENS_ZARR", "outputs/aifs_ens_e2_example.zarr"))
ANEMOI_ZARR_FILE = Path(
    os.environ.get("ANEMOI_ENS_ZARR", "outputs/aifs_ens_control_interpolated.zarr")
)
OUTPUT_DIR = Path(os.environ.get("AIFS_ENS_COMPARE_DIR", "outputs/comparison_plots"))

VARIABLE_MAPPING = {
    "t2m": "t2m",
    "d2m": "d2m",
    "u10m": "u10m",
    "v10m": "v10m",
    "tcw": "tcw",
    "t50": "t_50",
    "t100": "t_100",
    "t150": "t_150",
    "t200": "t_200",
    "t250": "t_250",
    "t300": "t_300",
    "t400": "t_400",
    "t500": "t_500",
    "t600": "t_600",
    "t700": "t_700",
    "t850": "t_850",
    "t925": "t_925",
    "t1000": "t_1000",
    "u50": "u_50",
    "u100": "u_100",
    "u150": "u_150",
    "u200": "u_200",
    "u250": "u_250",
    "u300": "u_300",
    "u400": "u_400",
    "u500": "u_500",
    "u600": "u_600",
    "u700": "u_700",
    "u850": "u_850",
    "u925": "u_925",
    "u1000": "u_1000",
    "v50": "v_50",
    "v100": "v_100",
    "v150": "v_150",
    "v200": "v_200",
    "v250": "v_250",
    "v300": "v_300",
    "v400": "v_400",
    "v500": "v_500",
    "v600": "v_600",
    "v700": "v_700",
    "v850": "v_850",
    "v925": "v_925",
    "v1000": "v_1000",
    "w50": "w_50",
    "w100": "w_100",
    "w150": "w_150",
    "w200": "w_200",
    "w250": "w_250",
    "w300": "w_300",
    "w400": "w_400",
    "w500": "w_500",
    "w600": "w_600",
    "w700": "w_700",
    "w850": "w_850",
    "w925": "w_925",
    "w1000": "w_1000",
    "q50": "q_50",
    "q100": "q_100",
    "q150": "q_150",
    "q200": "q_200",
    "q250": "q_250",
    "q300": "q_300",
    "q400": "q_400",
    "q500": "q_500",
    "q600": "q_600",
    "q700": "q_700",
    "q850": "q_850",
    "q925": "q_925",
    "q1000": "q_1000",
    "z50": "z_50",
    "z100": "z_100",
    "z150": "z_150",
    "z200": "z_200",
    "z250": "z_250",
    "z300": "z_300",
    "z400": "z_400",
    "z500": "z_500",
    "z600": "z_600",
    "z700": "z_700",
    "z850": "z_850",
    "z925": "z_925",
    "z1000": "z_1000",
    "msl": "msl",
    "skt": "skt",
    "sp": "sp",
    "stl1": "stl1",
    "stl2": "stl2",
}


def _resolve(path: Path) -> Path:
    if path.exists():
        return path
    alt = Path("validate_aifs") / path
    return alt if alt.exists() else path


def _reduce_to_lat_lon(array: np.ndarray) -> np.ndarray:
    field = array
    while field.ndim > 2:
        field = field[0]
    return field


def _list_variables(root: Path) -> list[str]:
    if (root / "zarr.json").exists():
        group = zarr.open_group(str(root), mode="r")
        return list(group.array_keys())
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _open_array(root: Path, name: str):
    if (root / "zarr.json").exists():
        return zarr.open_array(str(root), path=name, mode="r")
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return zarr.open_array(str(path), mode="r")


def _load_lat_lon(root: Path):
    lat = _open_array(root, "lat")[:]
    lon = _open_array(root, "lon")[:]
    return lat, lon


def _iter_timesteps(e2_field: np.ndarray, anemoi_field: np.ndarray):
    e2 = np.asarray(e2_field)
    an = np.asarray(anemoi_field)
    e2_steps = e2.shape[0] if e2.ndim > 2 else 1
    an_steps = an.shape[0] if an.ndim > 2 else 1
    steps = min(e2_steps, an_steps)
    if steps == 0:
        return
    if steps == 1:
        yield 0, _reduce_to_lat_lon(e2), _reduce_to_lat_lon(an)
        return
    for step in range(steps):
        yield step, _reduce_to_lat_lon(e2[step]), _reduce_to_lat_lon(an[step])


def create_plot(var_name: str, e2_field: np.ndarray, anemoi_field: np.ndarray, lat: np.ndarray, lon: np.ndarray, step: int) -> None:
    diff = e2_field - anemoi_field
    cmap_diff = LinearSegmentedColormap.from_list("diff", ["blue", "white", "red"])

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection=proj)
    ax1.coastlines(); ax1.add_feature(cfeature.BORDERS, linestyle=":")
    im1 = ax1.contourf(lon, lat, e2_field, transform=proj, cmap="viridis")
    ax1.set_title(f"E2 AIFS-ENS: {var_name} (step {step})")
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(132, projection=proj)
    ax2.coastlines(); ax2.add_feature(cfeature.BORDERS, linestyle=":")
    target_name = VARIABLE_MAPPING[var_name]
    im2 = ax2.contourf(lon, lat, anemoi_field, transform=proj, cmap="viridis")
    ax2.set_title(f"Anemoi: {target_name} (step {step})")
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(133, projection=proj)
    ax3.coastlines(); ax3.add_feature(cfeature.BORDERS, linestyle=":")
    im3 = ax3.contourf(lon, lat, diff, transform=proj, cmap=cmap_diff)
    ax3.set_title("Difference (E2 - Anemoi)")
    plt.colorbar(im3, ax=ax3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{var_name}_step{step:02d}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    e2_root = _resolve(E2_ZARR_FILE)
    anemoi_root = _resolve(ANEMOI_ZARR_FILE)
    if not e2_root.exists():
        print(f"Missing Earth2Studio file: {e2_root}")
        return
    if not anemoi_root.exists():
        print(f"Missing Anemoi file: {anemoi_root}")
        return

    lat, lon = _load_lat_lon(e2_root)

    e2_vars = _list_variables(e2_root)
    an_vars = set(_list_variables(anemoi_root))

    available_vars = [v for v in e2_vars if v in VARIABLE_MAPPING]
    print(f"Comparing {len(available_vars)} variables")

    for var in sorted(available_vars):
        mapped = VARIABLE_MAPPING[var]
        if mapped not in an_vars:
            continue

        e2_arr = _open_array(e2_root, var)[:]
        an_arr = _open_array(anemoi_root, mapped)[:]

        print(e2_arr.shape, an_arr.shape)

        for step, e2_slice, an_slice in _iter_timesteps(e2_arr, an_arr):
            if e2_slice.shape != an_slice.shape:
                print(
                    f"Skipping {var} step {step}: shape mismatch {e2_slice.shape} vs {an_slice.shape}"
                )
                continue

            diff = e2_slice - an_slice
            rms = float(np.sqrt(np.mean(diff ** 2)))
            bias = float(np.mean(diff))
            max_abs = float(np.max(np.abs(diff)))
            print(
                f"{var:<6} step {step:02d} RMS={rms:8.3e}  bias={bias:8.3e}  max|diff|={max_abs:8.3e}"
            )

            create_plot(var, e2_slice, an_slice, lat, lon, step)

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
