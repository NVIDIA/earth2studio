import numpy as np
from earth2studio.run_orbit import run
from earth2studio.data import NCAR_ERA5
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import OrbitGlobalPrecip

#time = [np.datetime64('2020-01-01T00:00:00')]
time = [np.datetime64('2020-01-01T00:00:00'), np.datetime64('2020-01-01T01:00:00')]

package = OrbitGlobalPrecip.load_default_package()
orbit = OrbitGlobalPrecip.load_model(package, "global", "9.5m", "precipitation")
#orbit = OrbitGlobalPrecip.load_model(package, "global", "126m", "precipitation")

data = NCAR_ERA5()
file_name = "outputs/orbit.zarr"
io = ZarrBackend(file_name)

run(time, orbit, data, io)
