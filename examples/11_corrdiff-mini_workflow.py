from collections import OrderedDict
from datetime import datetime

import numpy as np

from earth2studio.data import GFS
from earth2studio.io.netcdf4 import NetCDF4Backend
from earth2studio.models.auto import Package
from earth2studio.models.dx import CorrDiffMini
from earth2studio.models.px import SFNO
from earth2studio.workflows import diagnostic_downscale


def downscaled_forecast(
    corrdiff_package_path="/checkpoints/corrdiff/corrdiff-mini/earth2studio-package/",
    time=["2023-06-23T12:00:00"],
    timesteps=24,
    output_file="./corrdiffmini-output.nc",
    latlon=(37.372, -121.967)
):
    """CorrDiff-Mini downscaled forecast workflow example.
    To run this, you need to install Earth2Studio with SFNO support:
    https://nvidia.github.io/earth2studio/userguide/about/install.html#model-dependencies
    """
    data = GFS()
    fc_model = SFNO.load_model(SFNO.load_default_package())
    model = CorrDiffMini.load_model(
        Package(corrdiff_package_path, cache=False),
        center_latlon=latlon
    )
    model.number_of_samples = 2
    io = NetCDF4Backend(output_file)

    diagnostic_downscale(
        time, timesteps, fc_model, model, data, io,
        output_coords={"variable": np.array(["u10m", "v10m", "u10m_hr", "v10m_hr"])},
        output_prognostic=True
    )


if __name__ == "__main__":
    downscaled_forecast()
