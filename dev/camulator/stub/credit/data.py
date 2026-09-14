import xarray as xr

def get_forward_data(filename):
    return xr.open_dataset(filename)
