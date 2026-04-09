from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from earth2studio.data import UFSObsSat

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Use a date where AIRS (pre-decommission Sep 2023) is available
TIME = datetime(2021, 6, 15, 12)
TIME_TOLERANCE = timedelta(hours=3)  # ±3 h → 6-hour window centred on TIME

# Every sensor in the GSISatelliteLexicon
SENSORS = ["airs", "atms", "cris", "iasi", "mhs", "amsua"]
# Leaving amsub out because n15/n16/n17 are sparse by 2021

# Nice display names
LABELS = {
    "airs": "AIRS (Aqua)",
    "atms": "ATMS (NPP / N20)",
    "cris": "CrIS-FSR (NPP / N20)",
    "iasi": "IASI (MetOp-A/B/C)",
    "mhs": "MHS (MetOp + NOAA)",
    "amsua": "AMSU-A (MetOp + NOAA)",
}

# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
print(f"Fetching satellite obs centred on {TIME} (±{TIME_TOLERANCE}) ...")
ds = UFSObsSat(time_tolerance=TIME_TOLERANCE, cache=True, verbose=True)
frames = {}
for sensor in SENSORS:
    try:
        df = ds(TIME, [sensor])
        frames[sensor] = df
        print(f"  {sensor:8s}: {len(df):>8,} obs")
    except Exception as exc:
        print(f"  {sensor:8s}: FAILED - {exc}")

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
ncols = 3
nrows = int(np.ceil(len(frames) / ncols))
fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(7 * ncols, 4 * nrows),
    subplot_kw={"projection": None},
)
axes = np.atleast_2d(axes)

for idx, (sensor, df) in enumerate(frames.items()):
    ax = axes.flat[idx]
    lats = df["lat"].values
    lons = df["lon"].values
    obs = df["observation"].values

    # Clip colour range to 5th–95th percentile for readability
    vmin, vmax = np.nanpercentile(obs, [5, 95])
    sc = ax.scatter(
        lons,
        lats,
        c=obs,
        s=0.15,
        alpha=0.6,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{LABELS.get(sensor, sensor)}  ({len(df):,} obs)")
    fig.colorbar(sc, ax=ax, label="Brightness Temp (K)", shrink=0.7)

# Hide unused axes
for idx in range(len(frames), nrows * ncols):
    axes.flat[idx].set_visible(False)

fig.suptitle(
    f"UFS Satellite Observations – {TIME:%Y-%m-%d %H:%M}Z "
    f"(±{TIME_TOLERANCE.total_seconds() / 3600:.0f} h)",
    fontsize=16,
    y=1.01,
)
fig.tight_layout()

out_path = "ufs_satellite_sensors.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.show()
