import numpy as np
from earth2studio.run_orbit import run

from earth2studio.data import NCAR_ERA5

from earth2studio.io import ZarrBackend
from earth2studio.lexicon.ncar import NCAR_ERA5Lexicon
from earth2studio.models.dx import OrbitGlobalPrecip9_5M

package = OrbitGlobalPrecip9_5M.load_default_package()
orbit = OrbitGlobalPrecip9_5M.load_model(package, "global", "9.5m", "precipitation")

data = NCAR_ERA5()
file_name = "outputs/aifs_forecast.zarr"
io = ZarrBackend(file_name)

time = np.datetime64('2020-01-01T00:00:00')

data_check = True
inference_check = True
inference_check_file = '/lustre/orion/stf006/proj-shared/irl1/earth2studio/ORBIT-2-e2s/examples/0_preds.npy'
plot_inference = True

run([time], orbit, data, io, file_name, data_check, inference_check, inference_check_file, plot_inference)
