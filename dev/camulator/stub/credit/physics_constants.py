"""
A collection of constants

Reference:

Harrop, B.E., Pritchard, M.S., Parishani, H., Gettelman, A., Hagos, S., Lauritzen, P.H., Leung, L.R., Lu, J., Pressel, K.G. and Sakaguchi, K., 2022. Conservation of dry air, water, and energy in CAM and its potential impact on tropical rainfall. Journal of Climate, 35(9), pp.2895-2917.
"""

from math import pi

# Pi
PI = pi

# Earth's radius
RAD_EARTH = 6371000  # m

# Earth's rate of rotation
OMEGA = 7.292e-5  # radians/s

# ideal gas constant of water vapor
RVGAS = 461.5  # J/kg/K

# ideal gas constant of dry air
RDGAS = 287.05  # J/kg/K

# ratio of Rd to Rv
EPSGAS = RDGAS / RVGAS

# gravity
GRAVITY = 9.80665  # m/s^2

# density of water
RHO_WATER = 1000.0  # kg/m^3

# ========================================================= #
# latent heat caused by the phase change of water (0 deg C)
LH_WATER = 2.501e6  # J/kg
LH_ICE = 333700  # J/kg

# ========================================================= #
# heat capacity on constant pressure
# dry air
CP_DRY = 1004.64  # J/kg K
# water vapor
CP_VAPOR = 1810.0  # J/kg K
# liquid
CP_LIQUID = 4188.0  # J/kg K
# ice
CP_ICE = 2117.27  # J/kg K
