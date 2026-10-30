# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections.abc import Callable

import numpy as np

from .base import LexiconType

LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

MDL_LEVELS = np.arange(1, 138)

# Number of trailing hourly samples summed together for each accumulated
# variable, e.g. "tp06" is the trailing 6-hour sum of hourly "tp" and
# "ttr03" is the trailing 3-hour sum of hourly "ttr".
ACCUMULATION_HOURS = {
    "cp06": 6,
    "ro06": 6,
    "sf06": 6,
    "ssrd06": 6,
    "strd06": 6,
    "tp06": 6,
    "ttr03": 3,
}


class ARCO_ERA5Lexicon(metaclass=LexiconType):
    """ARCO ERA5 Lexicon
    ARCO specified <Variable ID>::<Level>. Model-level identifiers end in ``k``.

    Note
    ----
    Covers all single-level, pressure-level, and model-level fields in the Google
    analysis-ready stores as of October 2026. Identifiers follow Earth2Studio
    conventions, otherwise ERA5 short names (dots replaced by underscores).
    Snowfall and runoff amounts use metres of water equivalent.
    """

    VOCAB = {
        "u10m": "10m_u_component_of_wind::",
        "v10m": "10m_v_component_of_wind::",
        "u100m": "100m_u_component_of_wind::",
        "v100m": "100m_v_component_of_wind::",
        "t2m": "2m_temperature::",
        "d2m": "2m_dewpoint_temperature::",
        "sp": "surface_pressure::",
        "sst": "sea_surface_temperature::",
        "msl": "mean_sea_level_pressure::",
        "tcwv": "total_column_water_vapour::",
        "tp": "total_precipitation::",
        "lsm": "land_sea_mask::",
        "z": "geopotential_at_surface::",
        "mtdwswrf": "mean_top_downward_short_wave_radiation_flux::",
        "ttr": "top_net_thermal_radiation::",
        "skt": "skin_temperature::",
        "sic": "sea_ice_cover::",
        # AIFS/AIFS ENS and AIFS2 aliases backed by ARCO_ERA5 single-level fields.
        "cdww": "coefficient_of_drag_with_waves::",
        "cp06": "convective_precipitation::",
        "cos_mwd": "mean_wave_direction::",
        "hcc": "high_cloud_cover::",
        "lcc": "low_cloud_cover::",
        "mcc": "medium_cloud_cover::",
        "mwp": "mean_wave_period::",
        "ro": "runoff::",
        "ro06": "runoff::",
        "sd": "snow_depth::",
        "sdor": "standard_deviation_of_orography::",
        "sf": "snowfall::",
        "sf06": "snowfall::",
        "sin_mwd": "mean_wave_direction::",
        "slor": "slope_of_sub_gridscale_orography::",
        "ssrd06": "surface_solar_radiation_downwards::",
        "stl1": "soil_temperature_level_1::",
        "stl2": "soil_temperature_level_2::",
        "strd06": "surface_thermal_radiation_downwards::",
        "swh": "significant_height_of_combined_wind_waves_and_swell::",
        "swvl1": "volumetric_soil_water_layer_1::",
        "swvl2": "volumetric_soil_water_layer_2::",
        "tcc": "total_cloud_cover::",
        "tcw": "total_column_water::",
        "tp06": "total_precipitation::",
        "ttr03": "top_net_thermal_radiation::",
        "wmb": "model_bathymetry::",
        # Remaining single-level fields in full_37-1h-0p25deg-chunk-1.zarr-v3.
        "u10n": "10m_u_component_of_neutral_wind::",
        "v10n": "10m_v_component_of_neutral_wind::",
        "fg10m": "10m_wind_gust_since_previous_post_processing::",
        "p140209": "air_density_over_the_oceans::",
        "anor": "angle_of_sub_gridscale_orography::",
        "isor": "anisotropy_of_sub_gridscale_orography::",
        "bfi": "benjamin_feir_index::",
        "bld": "boundary_layer_dissipation::",
        "blh": "boundary_layer_height::",
        "chnk": "charnock::",
        "cdir": "clear_sky_direct_solar_radiation_at_surface::",
        "cbh_agl": "cloud_base_height::",
        "cape": "convective_available_potential_energy::",
        "cin": "convective_inhibition::",
        "cp": "convective_precipitation::",
        "crr": "convective_rain_rate::",
        "csf": "convective_snowfall::",
        "csfr": "convective_snowfall_rate_water_equivalent::",
        "uvb": "downward_uv_radiation_at_the_surface::",
        "dctb": "duct_base_height::",
        "lgws": "eastward_gravity_wave_surface_stress::",
        "ewss": "eastward_turbulent_surface_stress::",
        "e": "evaporation::",
        "fal": "forecast_albedo::",
        "flsr": "forecast_logarithm_of_surface_roughness_for_heat::",
        "fsr": "forecast_surface_roughness::",
        "p140208": "free_convective_velocity_over_the_oceans::",
        "zust": "friction_velocity::",
        "gwd": "gravity_wave_dissipation::",
        "cvh": "high_vegetation_cover::",
        "istl1": "ice_temperature_layer_1::",
        "istl2": "ice_temperature_layer_2::",
        "istl3": "ice_temperature_layer_3::",
        "istl4": "ice_temperature_layer_4::",
        "i10fg": "instantaneous_10m_wind_gust::",
        "iews": "instantaneous_eastward_turbulent_surface_stress::",
        "ilspf": "instantaneous_large_scale_surface_precipitation_fraction::",
        "ie": "instantaneous_moisture_flux::",
        "inss": "instantaneous_northward_turbulent_surface_stress::",
        "ishf": "instantaneous_surface_sensible_heat_flux::",
        "kx": "k_index::",
        "lblt": "lake_bottom_temperature::",
        "cl": "lake_cover::",
        "dl": "lake_depth::",
        "licd": "lake_ice_depth::",
        "lict": "lake_ice_temperature::",
        "lmld": "lake_mix_layer_depth::",
        "lmlt": "lake_mix_layer_temperature::",
        "lshf": "lake_shape_factor::",
        "ltlt": "lake_total_layer_temperature::",
        "lsp": "large_scale_precipitation::",
        "lspf": "large_scale_precipitation_fraction::",
        "lsrr": "large_scale_rain_rate::",
        "lsf": "large_scale_snowfall::",
        "lssfr": "large_scale_snowfall_rate_water_equivalent::",
        "lai_hv": "leaf_area_index_high_vegetation::",
        "lai_lv": "leaf_area_index_low_vegetation::",
        "cvl": "low_vegetation_cover::",
        "mx2t": "maximum_2m_temperature_since_previous_post_processing::",
        "hmax": "maximum_individual_wave_height::",
        "mxtpr": "maximum_total_precipitation_rate_since_previous_post_processing::",
        "mbld": "mean_boundary_layer_dissipation::",
        "mcpr": "mean_convective_precipitation_rate::",
        "mcsr": "mean_convective_snowfall_rate::",
        "mdts": "mean_direction_of_total_swell::",
        "mdww": "mean_direction_of_wind_waves::",
        "megwss": "mean_eastward_gravity_wave_surface_stress::",
        "metss": "mean_eastward_turbulent_surface_stress::",
        "mer": "mean_evaporation_rate::",
        "mgwd": "mean_gravity_wave_dissipation::",
        "mlspf": "mean_large_scale_precipitation_fraction::",
        "mlspr": "mean_large_scale_precipitation_rate::",
        "mlssr": "mean_large_scale_snowfall_rate::",
        "mngwss": "mean_northward_gravity_wave_surface_stress::",
        "mntss": "mean_northward_turbulent_surface_stress::",
        "mpts": "mean_period_of_total_swell::",
        "mpww": "mean_period_of_wind_waves::",
        "mper": "mean_potential_evaporation_rate::",
        "mror": "mean_runoff_rate::",
        "mser": "mean_snow_evaporation_rate::",
        "msr": "mean_snowfall_rate::",
        "msmr": "mean_snowmelt_rate::",
        "msqs": "mean_square_slope_of_waves::",
        "mssror": "mean_sub_surface_runoff_rate::",
        "msdrswrf": "mean_surface_direct_short_wave_radiation_flux::",
        "msdrswrfcs": "mean_surface_direct_short_wave_radiation_flux_clear_sky::",
        "msdwlwrf": "mean_surface_downward_long_wave_radiation_flux::",
        "msdwlwrfcs": "mean_surface_downward_long_wave_radiation_flux_clear_sky::",
        "msdwswrf": "mean_surface_downward_short_wave_radiation_flux::",
        "msdwswrfcs": "mean_surface_downward_short_wave_radiation_flux_clear_sky::",
        "msdwuvrf": "mean_surface_downward_uv_radiation_flux::",
        "mslhf": "mean_surface_latent_heat_flux::",
        "msnlwrf": "mean_surface_net_long_wave_radiation_flux::",
        "msnlwrfcs": "mean_surface_net_long_wave_radiation_flux_clear_sky::",
        "msnswrf": "mean_surface_net_short_wave_radiation_flux::",
        "msnswrfcs": "mean_surface_net_short_wave_radiation_flux_clear_sky::",
        "msror": "mean_surface_runoff_rate::",
        "msshf": "mean_surface_sensible_heat_flux::",
        "mtnlwrf": "mean_top_net_long_wave_radiation_flux::",
        "mtnlwrfcs": "mean_top_net_long_wave_radiation_flux_clear_sky::",
        "mtnswrf": "mean_top_net_short_wave_radiation_flux::",
        "mtnswrfcs": "mean_top_net_short_wave_radiation_flux_clear_sky::",
        "mtpr": "mean_total_precipitation_rate::",
        "dndza": "mean_vertical_gradient_of_refractivity_inside_trapping_layer::",
        "mvimd": "mean_vertically_integrated_moisture_divergence::",
        "mwd": "mean_wave_direction::",
        "p140122": "mean_wave_direction_of_first_swell_partition::",
        "p140125": "mean_wave_direction_of_second_swell_partition::",
        "p140128": "mean_wave_direction_of_third_swell_partition::",
        "mp1": "mean_wave_period_based_on_first_moment::",
        "p1ps": "mean_wave_period_based_on_first_moment_for_swell::",
        "p1ww": "mean_wave_period_based_on_first_moment_for_wind_waves::",
        "p2ps": "mean_wave_period_based_on_second_moment_for_swell::",
        "p2ww": "mean_wave_period_based_on_second_moment_for_wind_waves::",
        "p140123": "mean_wave_period_of_first_swell_partition::",
        "p140126": "mean_wave_period_of_second_swell_partition::",
        "p140129": "mean_wave_period_of_third_swell_partition::",
        "mp2": "mean_zero_crossing_wave_period::",
        "mn2t": "minimum_2m_temperature_since_previous_post_processing::",
        "mntpr": "minimum_total_precipitation_rate_since_previous_post_processing::",
        "dndzn": "minimum_vertical_gradient_of_refractivity_inside_trapping_layer::",
        "alnid": "near_ir_albedo_for_diffuse_radiation::",
        "alnip": "near_ir_albedo_for_direct_radiation::",
        "phioc": "normalized_energy_flux_into_ocean::",
        "phiaw": "normalized_energy_flux_into_waves::",
        "tauoc": "normalized_stress_into_ocean::",
        "mgws": "northward_gravity_wave_surface_stress::",
        "nsss": "northward_turbulent_surface_stress::",
        "dwi": "ocean_surface_stress_equivalent_10m_neutral_wind_direction::",
        "wind": "ocean_surface_stress_equivalent_10m_neutral_wind_speed::",
        "pp1d": "peak_wave_period::",
        "tmax": "period_corresponding_to_maximum_individual_wave_height::",
        "pev": "potential_evaporation::",
        "ptype": "precipitation_type::",
        "shts": "significant_height_of_total_swell::",
        "shww": "significant_height_of_wind_waves::",
        "p140121": "significant_wave_height_of_first_swell_partition::",
        "p140124": "significant_wave_height_of_second_swell_partition::",
        "p140127": "significant_wave_height_of_third_swell_partition::",
        "src": "skin_reservoir_content::",
        "asn": "snow_albedo::",
        "rsn": "snow_density::",
        "es": "snow_evaporation::",
        "smlt": "snowmelt::",
        "stl3": "soil_temperature_level_3::",
        "stl4": "soil_temperature_level_4::",
        "slt": "soil_type::",
        "sdfor": "standard_deviation_of_filtered_subgrid_orography::",
        "ssro": "sub_surface_runoff::",
        "slhf": "surface_latent_heat_flux::",
        "ssr": "surface_net_solar_radiation::",
        "ssrc": "surface_net_solar_radiation_clear_sky::",
        "str": "surface_net_thermal_radiation::",
        "strc": "surface_net_thermal_radiation_clear_sky::",
        "sro": "surface_runoff::",
        "sshf": "surface_sensible_heat_flux::",
        "ssrdc": "surface_solar_radiation_downward_clear_sky::",
        "ssrd": "surface_solar_radiation_downwards::",
        "strdc": "surface_thermal_radiation_downward_clear_sky::",
        "strd": "surface_thermal_radiation_downwards::",
        "tsn": "temperature_of_snow_layer::",
        "tisr": "toa_incident_solar_radiation::",
        "tsr": "top_net_solar_radiation::",
        "tsrc": "top_net_solar_radiation_clear_sky::",
        "ttrc": "top_net_thermal_radiation_clear_sky::",
        "tciw": "total_column_cloud_ice_water::",
        "tclw": "total_column_cloud_liquid_water::",
        "tco3": "total_column_ozone::",
        "tcrw": "total_column_rain_water::",
        "tcsw": "total_column_snow_water::",
        "tcslw": "total_column_supercooled_liquid_water::",
        "fdir": "total_sky_direct_solar_radiation_at_surface::",
        "totalx": "total_totals_index::",
        "tplb": "trapping_layer_base_height::",
        "tplt": "trapping_layer_top_height::",
        "tvh": "type_of_high_vegetation::",
        "tvl": "type_of_low_vegetation::",
        "ust": "u_component_stokes_drift::",
        "aluvd": "uv_visible_albedo_for_diffuse_radiation::",
        "aluvp": "uv_visible_albedo_for_direct_radiation::",
        "vst": "v_component_stokes_drift::",
        "p80_162": "vertical_integral_of_divergence_of_cloud_frozen_water_flux::",
        "p79_162": "vertical_integral_of_divergence_of_cloud_liquid_water_flux::",
        "p85_162": "vertical_integral_of_divergence_of_geopotential_flux::",
        "p82_162": "vertical_integral_of_divergence_of_kinetic_energy_flux::",
        "p81_162": "vertical_integral_of_divergence_of_mass_flux::",
        "p84_162": "vertical_integral_of_divergence_of_moisture_flux::",
        "p87_162": "vertical_integral_of_divergence_of_ozone_flux::",
        "p83_162": "vertical_integral_of_divergence_of_thermal_energy_flux::",
        "p86_162": "vertical_integral_of_divergence_of_total_energy_flux::",
        "p90_162": "vertical_integral_of_eastward_cloud_frozen_water_flux::",
        "p88_162": "vertical_integral_of_eastward_cloud_liquid_water_flux::",
        "p73_162": "vertical_integral_of_eastward_geopotential_flux::",
        "p69_162": "vertical_integral_of_eastward_heat_flux::",
        "p67_162": "vertical_integral_of_eastward_kinetic_energy_flux::",
        "p65_162": "vertical_integral_of_eastward_mass_flux::",
        "p77_162": "vertical_integral_of_eastward_ozone_flux::",
        "p75_162": "vertical_integral_of_eastward_total_energy_flux::",
        "p71_162": "vertical_integral_of_eastward_water_vapour_flux::",
        "p64_162": "vertical_integral_of_energy_conversion::",
        "p59_162": "vertical_integral_of_kinetic_energy::",
        "p53_162": "vertical_integral_of_mass_of_atmosphere::",
        "p92_162": "vertical_integral_of_mass_tendency::",
        "p91_162": "vertical_integral_of_northward_cloud_frozen_water_flux::",
        "p89_162": "vertical_integral_of_northward_cloud_liquid_water_flux::",
        "p74_162": "vertical_integral_of_northward_geopotential_flux::",
        "p70_162": "vertical_integral_of_northward_heat_flux::",
        "p68_162": "vertical_integral_of_northward_kinetic_energy_flux::",
        "p66_162": "vertical_integral_of_northward_mass_flux::",
        "p78_162": "vertical_integral_of_northward_ozone_flux::",
        "p76_162": "vertical_integral_of_northward_total_energy_flux::",
        "p72_162": "vertical_integral_of_northward_water_vapour_flux::",
        "p61_162": "vertical_integral_of_potential_and_internal_energy::",
        "p62_162": "vertical_integral_of_potential_internal_and_latent_energy::",
        "p54_162": "vertical_integral_of_temperature::",
        "p60_162": "vertical_integral_of_thermal_energy::",
        "p63_162": "vertical_integral_of_total_energy::",
        "vimd": "vertically_integrated_moisture_divergence::",
        "swvl3": "volumetric_soil_water_layer_3::",
        "swvl4": "volumetric_soil_water_layer_4::",
        "wdw": "wave_spectral_directional_width::",
        "dwps": "wave_spectral_directional_width_for_swell::",
        "dwww": "wave_spectral_directional_width_for_wind_waves::",
        "wsk": "wave_spectral_kurtosis::",
        "wsp": "wave_spectral_peakedness::",
        "wss": "wave_spectral_skewness::",
        "deg0l": "zero_degree_level::",
    }
    VOCAB.update({f"u{level}": f"u_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"v{level}": f"v_component_of_wind::{level}" for level in LEVELS})
    VOCAB.update({f"w{level}": f"vertical_velocity::{level}" for level in LEVELS})
    VOCAB.update({f"z{level}": f"geopotential::{level}" for level in LEVELS})
    VOCAB.update({f"t{level}": f"temperature::{level}" for level in LEVELS})
    VOCAB.update({f"q{level}": f"specific_humidity::{level}" for level in LEVELS})
    VOCAB.update(
        {
            f"clwc{level}": f"specific_cloud_liquid_water_content::{level}"
            for level in LEVELS
        }
    )
    VOCAB.update(
        {
            f"ciwc{level}": f"specific_cloud_ice_water_content::{level}"
            for level in LEVELS
        }
    )

    VOCAB.update(
        {f"cc{level}": f"fraction_of_cloud_cover::{level}" for level in LEVELS}
    )
    VOCAB.update(
        {f"o3{level}": f"ozone_mass_mixing_ratio::{level}" for level in LEVELS}
    )
    VOCAB.update({f"pv{level}": f"potential_vorticity::{level}" for level in LEVELS})

    # Model levels are stored in a separate Zarr group, but use the same base names
    VOCAB.update(
        {f"u{level}k": f"u_component_of_wind::{level}" for level in MDL_LEVELS}
    )
    VOCAB.update(
        {f"v{level}k": f"v_component_of_wind::{level}" for level in MDL_LEVELS}
    )
    VOCAB.update({f"t{level}k": f"temperature::{level}" for level in MDL_LEVELS})
    VOCAB.update({f"q{level}k": f"specific_humidity::{level}" for level in MDL_LEVELS})
    VOCAB.update(
        {
            f"clwc{level}k": f"specific_cloud_liquid_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"ciwc{level}k": f"specific_cloud_ice_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"crwc{level}k": f"specific_rain_water_content::{level}"
            for level in MDL_LEVELS
        }
    )
    VOCAB.update(
        {
            f"cswc{level}k": f"specific_snow_water_content::{level}"
            for level in MDL_LEVELS
        }
    )

    VOCAB.update({f"d{level}k": f"divergence::{level}" for level in MDL_LEVELS})
    VOCAB.update(
        {f"cc{level}k": f"fraction_of_cloud_cover::{level}" for level in MDL_LEVELS}
    )
    VOCAB.update({f"z{level}k": f"geopotential::{level}" for level in MDL_LEVELS})
    VOCAB.update(
        {f"o3{level}k": f"ozone_mass_mixing_ratio::{level}" for level in MDL_LEVELS}
    )
    VOCAB.update({f"w{level}k": f"vertical_velocity::{level}" for level in MDL_LEVELS})
    VOCAB.update({f"vo{level}k": f"vorticity::{level}" for level in MDL_LEVELS})

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in ARCO_ERA5 vocabulary."""
        arco_key = cls.VOCAB[val]

        if val == "cos_mwd":

            def mod(x: np.array) -> np.array:
                return np.cos(np.deg2rad(x))

        elif val == "sin_mwd":

            def mod(x: np.array) -> np.array:
                return np.sin(np.deg2rad(x))

        else:

            def mod(x: np.array) -> np.array:
                """Modify name (if necessary)."""
                return x

        return arco_key, mod


def __getattr__(name: str) -> type[ARCO_ERA5Lexicon]:
    """Return deprecated lexicon aliases."""
    if name == "ARCOLexicon":
        warnings.warn(
            "ARCOLexicon has been renamed to ARCO_ERA5Lexicon and will be removed "
            "in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ARCO_ERA5Lexicon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
