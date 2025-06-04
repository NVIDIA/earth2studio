import xarray as xr
import numpy as np
import argparse
from pathlib import Path
import os
import torch

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.utils.coords import CoordSystem
from earth2studio.statistics import crps

expected_scores = {
    "dlesym": {
        "t2m": torch.tensor([2.1747, 2.0343, 2.3918, 1.6183, 3.0419, 1.9484, 1.7904]),
        "z500": torch.tensor([288.6965, 474.8904, 291.3008, 469.3990, 468.8430, 210.8404, 270.3322]),
    },
    "sfno": {
        "t2m": torch.tensor([4.3031, 4.0860, 3.7687, 4.2791, 5.4535, 3.8613, 4.2332]),
        "z500": torch.tensor([143.6826, 205.8306, 263.3020, 398.4369, 504.6732, 375.4496, 235.7776]),
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    model = args.model
    path = os.path.join(script_dir, args.path)

    if model not in ["dlesym", "sfno"]:
        raise ValueError("Model must be either 'dlesym' or 'sfno'")
    
    verif_lead_times = np.arange(4, 30, 4, dtype='timedelta64[D]')
    vars = ["t2m", "z500"]
    data_source = ARCO(verbose=False)
    metric = crps(ensemble_dimension="ensemble", reduction_dimensions=["lat", "lon"], fair=True)
    passed = {var: False for var in vars}
    
    for var in vars:
        with xr.open_zarr(path) as ds:
            # Modify the inherited time/lead time coords in the io backend to be datetime64/timedelta64
            # Needed as datetime64/timedelta64 are not supported by Zarr 3.0 yet
            # https://github.com/zarr-developers/zarr-python/issues/2616
            # TODO: Remove once fixed
            ds["time"] = np.array(ds["time"], dtype="datetime64[ns]")
            ds["lead_time"] = np.array(ds["lead_time"], dtype="timedelta64[ns]")

            # Load forecast data
            fcst = ds[var].isel(time=[0]).sel(lead_time=verif_lead_times)
            fcst_coords = CoordSystem(
                ensemble=fcst.ensemble.values,
                time=fcst.time.values,
                lead_time=fcst.lead_time.values,
                lat=fcst.lat.values,
                lon=fcst.lon.values,
            )
            
            # Load verification data
            interp_coords = {
                "_lat": fcst.lat.values,
                "_lon": fcst.lon.values,
            }
            verif, verif_coords = fetch_data(
                source=data_source,
                variable=var,
                time=ds.time.values,
                lead_time=verif_lead_times,
                interp_to=interp_coords,
            )
            verif = verif[:, :, 0, :, :]
            verif_coords.pop("variable")
            verif_coords["lat"], verif_coords["lon"] = verif_coords["_lat"], verif_coords["_lon"]
            del verif_coords["_lat"], verif_coords["_lon"]

            # Check within 5% of expected scores
            scores, score_coords = metric(torch.from_numpy(fcst.values), fcst_coords, verif, verif_coords)
            if torch.allclose(scores.squeeze(), expected_scores[model][var], rtol=5e-2):
                passed[var] = True
            else:
                print(f"Expected skill not verified for {model}:")
                for lt in range(len(expected_scores[model][var])):
                    print(f"Lead time {verif_lead_times[lt]}")
                    print(f"Expected: {expected_scores[model][var][lt]}")
                    print(f"Actual: {scores.squeeze()[lt]}")

    if all(passed.values()):
        print(f"Expected skill verified for {model}")

if __name__ == "__main__":
    main()