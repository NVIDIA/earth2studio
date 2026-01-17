import matplotlib.pyplot as plt
import xarray as xr

# Load
ds1 = xr.open_dataarray("output.nc")
ds2 = xr.open_dataarray("output2.nc")

# Ensure same coords/order; will error if coords differ
try:
    ds1, ds2 = xr.align(ds1, ds2, join="exact")
except Exception as e:
    print("Alignment failed (different coords or shapes):", e)
    raise

# Quick checks
print("identical:", ds1.identical(ds2))  # includes attributes
print("equals  :", ds1.equals(ds2))  # ignores attributes, checks values/dtypes


# Numerical closeness (tolerant)
def _allclose(a, b, rtol=1e-6, atol=0.0) -> bool:
    try:
        xr.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print("Not allclose:", e)
        return False


print("allclose:", _allclose(ds1, ds2, rtol=1e-6, atol=0.0))

fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
im0 = axs[0].imshow(
    ds1.isel(sample=0).sel(variable="t2m").values, cmap="viridis", vmin=250, vmax=350
)
axs[0].set_title("t2m - output.nc")
fig.colorbar(im0, ax=axs[0], orientation="horizontal")

im1 = axs[1].imshow(
    ds2.isel(sample=0).sel(variable="t2m").values, cmap="viridis", vmin=250, vmax=350
)
axs[1].set_title("t2m - output2.nc")
fig.colorbar(im1, ax=axs[1], orientation="horizontal")

for ax in axs:
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

plt.savefig("t2m_compare.jpg", dpi=150)
print("Saved t2m_compare.jpg")
