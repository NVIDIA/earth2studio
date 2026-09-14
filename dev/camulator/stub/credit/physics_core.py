"""
Tools for physics-based constraints and derivations for CREDIT models
--------------------------------------------------------------------------
Content:
    - physics_pressure_level
    - physics_hybrid_sigma_level

Reference:
    - https://journals.ametsoc.org/view/journals/clim/34/10/JCLI-D-20-0676.1.xml
    - https://doi.org/10.1175/JCLI-D-13-00018.1
    - https://github.com/ai2cm/ace/tree/main/fme/fme/core
"""

import torch
import torch.nn as nn
from typing import Dict
from credit.physics_constants import RAD_EARTH, RDGAS, EPSGAS, GRAVITY


def compute_density(pressure, temperature, specific_humidity):
    """
    compute density given pressure (Pa), temperature (K), and specific humidity (kg/kg)
    """
    virtual_temperature = compute_virtual_temperature(temperature, specific_humidity)
    density = pressure / (RDGAS * virtual_temperature)
    return density


def compute_virtual_temperature(temperature, specific_humidity):
    """ref: metpy"""
    mixing_ratio = specific_humidity / (1 - specific_humidity)
    temperature_virtual = temperature * (mixing_ratio + EPSGAS) / (EPSGAS * (1 + mixing_ratio))
    return temperature_virtual


class ModelLevelPressures(nn.Module):
    """
    compute pressure levels given SP with (only compatible with torch)
    SP, a_vals with same units.
    a_vals, b_vals, with size levels at dimension plev_dim and all other dims with size 1,
    e.g. plev_dim = 1; a_vals.shape = (1, levels, 1, 1, 1)
    matching sp with size 1 at dimension plev_dim e.g. (b, 1, t, lat, lon)
    """

    def __init__(self, a_vals, b_vals, plev_dim=1):
        super().__init__()
        self.register_buffer("a_vals", a_vals, persistent=False)
        self.register_buffer("b_vals", b_vals, persistent=False)

        self.plev_dim = plev_dim
        self.is_fully_initialized = False

    def compute_p(self, sp):
        plevs = self.a_vals + self.b_vals * sp
        return plevs  # shape = sp.shape except at plev_dim which is now nlevel

    def compute_hlevs(self, plevs):
        # half levels as averages of model level pressures

        hlevs = torch.log(plevs.unfold(dimension=self.plev_dim, size=2, step=1)).mean(dim=-1)
        return torch.exp(hlevs)  # same shape a plev except plev_dim is 1 less

    def compute_mlev_thickness(self, sp):
        plevs = self.compute_p(sp)
        hlevs = self.compute_hlevs(plevs)

        if not self.is_fully_initialized:  # initialize zeros
            self.register_buffer("zeros", torch.zeros_like(sp), persistent=False)
            self.is_fully_initialized = True

        thicknesses = torch.diff(hlevs, dim=self.plev_dim, prepend=self.zeros, append=sp)
        return thicknesses  # same shape as sp but plev_dim has size levels


class physics_pressure_level:
    """
    Pressure level physics

    All inputs must be in the same torch device.

    Full order of dimensions:  (batch, time, level, latitude, longitude)
    """

    def __init__(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor,
        upper_air_pressure: torch.Tensor,
        midpoint: bool = False,
    ):
        """
        Attributes:
            upper_air_pressure (torch.Tensor): pressure levels in Pa.
            lon (torch.Tensor): longitude in degrees.
            lat (torch.Tensor): latitude in degrees.
            pressure_thickness (torch.Tensor): pressure thickness between levels.
            dx, dy (torch.Tensor): grid spacings in longitude and latitude.
            area (torch.Tensor): area of grid cells.
            integral (function): vertical integration method (midpoint or trapezoidal).
        """
        self.lon = lon
        self.lat = lat
        self.upper_air_pressure = upper_air_pressure

        # ========================================================================= #
        # compute pressure level thickness
        self.pressure_thickness = self.upper_air_pressure.diff(dim=-1)

        # # ========================================================================= #
        # # compute grid spacings
        # lat_rad = torch.deg2rad(self.lat)
        # lon_rad = torch.deg2rad(self.lon)
        # self.dy = torch.gradient(lat_rad * RAD_EARTH, dim=0)[0]
        # self.dx = torch.gradient(lon_rad * RAD_EARTH, dim=1)[0] * torch.cos(lat_rad)

        # ========================================================================= #
        # compute gtid area
        # area = R^2 * d_sin(lat) * d_lon
        lat_rad = torch.deg2rad(self.lat)
        lon_rad = torch.deg2rad(self.lon)
        sin_lat_rad = torch.sin(lat_rad)
        d_phi = torch.gradient(sin_lat_rad, dim=0, edge_order=2)[0]
        d_lambda = torch.gradient(lon_rad, dim=1, edge_order=2)[0]
        d_lambda = (d_lambda + torch.pi) % (2 * torch.pi) - torch.pi
        self.area = torch.abs(RAD_EARTH**2 * d_phi * d_lambda)

        # ========================================================================== #
        # vertical integration method
        if midpoint:
            self.integral = self.pressure_integral_midpoint
            self.integral_sliced = self.pressure_integral_midpoint_sliced
        else:
            self.integral = self.pressure_integral_trapz
            self.integral_sliced = self.pressure_integral_trapz_sliced

    def pressure_integral_midpoint(self, q_mid: torch.Tensor) -> torch.Tensor:
        """
        Compute the pressure level integral of a given quantity; assuming its mid point
        values are pre-computed

        Args:
            q_mid: the quantity with dims of (batch_size, time, level-1, latitude, longitude)

        Returns:
            Pressure level integrals of q
        """
        num_dims = len(q_mid.shape)
        delta_p = self.pressure_thickness.to(q_mid.device)

        if num_dims == 5:  # (batch_size, level, time, latitude, longitude)
            delta_p = delta_p.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            q_area = q_mid * delta_p
            q_trapz = torch.sum(q_area, dim=1)

        elif num_dims == 4:  # (batch_size, level, latitude, longitude) or (time, level, latitude, longitude)
            delta_p = delta_p.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q_area = q_mid * delta_p
            q_trapz = torch.sum(q_area, dim=1)

        elif num_dims == 3:  # (level, latitude, longitude)
            delta_p = delta_p.unsqueeze(-1).unsqueeze(-1)  # Expand for broadcasting
            q_area = q_mid * delta_p
            q_trapz = torch.sum(q_area, dim=0)

        else:
            raise ValueError(f"Unsupported tensor dimensions: {q_mid.shape}")

        return q_trapz

    def pressure_integral_midpoint_sliced(self, q_mid: torch.Tensor, ind_start: int, ind_end: int) -> torch.Tensor:
        """
        As in `pressure_integral_midpoint`, but supports pressure level indexing,
        so it can calculate integrals of a subset of levels
        """
        num_dims = len(q_mid.shape)
        delta_p = self.pressure_thickness[ind_start:ind_end].to(q_mid.device)

        if num_dims == 5:  # (batch_size, time, level, latitude, longitude)
            delta_p = delta_p.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            q_mid = q_mid[:, ind_start:ind_end, ...]
            q_area = q_mid * delta_p
            q_trapz = torch.sum(q_area, dim=1)

        elif num_dims == 4:  # (batch_size, level, latitude, longitude) or (time, level, latitude, longitude)
            delta_p = delta_p.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q_mid = q_mid[:, ind_start:ind_end, ...]
            q_area = q_mid * delta_p  # Trapezoidal rule
            q_trapz = torch.sum(q_area, dim=1)

        elif num_dims == 3:  # (level, latitude, longitude)
            delta_p = delta_p.unsqueeze(-1).unsqueeze(-1)  # Expand for broadcasting
            q_mid = q_mid[ind_start:ind_end, ...]
            q_area = q_mid * delta_p
            q_trapz = torch.sum(q_area, dim=0)

        else:
            raise ValueError(f"Unsupported tensor dimensions: {q_mid.shape}")

        return q_trapz

    def pressure_integral_trapz(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the pressure level integral of a given quantity using the trapezoidal rule.

        Args:
            q: the quantity with dims of (batch_size, time, level, latitude, longitude)

        Returns:
            Pressure level integrals of q
        """
        num_dims = len(q.shape)
        delta_p = self.pressure_thickness.to(q.device)

        if num_dims == 5:  # (batch_size, level, time, latitude, longitude)
            delta_p = delta_p.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            q_area = 0.5 * (q[:, :-1, :, :, :] + q[:, 1:, :, :, :]) * delta_p
            q_trapz = torch.sum(q_area, dim=1)

        elif num_dims == 4:  # (batch_size, level, latitude, longitude) or (time, level, latitude, longitude)
            delta_p = delta_p.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q_area = 0.5 * (q[:, :-1, :, :] + q[:, 1:, :, :]) * delta_p  # Trapezoidal rule
            q_trapz = torch.sum(q_area, dim=1)

        elif num_dims == 3:  # (level, latitude, longitude)
            delta_p = delta_p.unsqueeze(-1).unsqueeze(-1)  # Expand for broadcasting
            q_area = 0.5 * (q[:-1, :, :] + q[1:, :, :]) * delta_p
            q_trapz = torch.sum(q_area, dim=0)

        else:
            raise ValueError(f"Unsupported tensor dimensions: {q.shape}")

        return q_trapz

    def pressure_integral_trapz_sliced(self, q: torch.Tensor, ind_start: int, ind_end: int) -> torch.Tensor:
        """
        As in `pressure_integral_trapz`, but supports pressure level indexing,
        so it can calculate integrals of a subset of levels
        """
        num_dims = len(q.shape)
        delta_p = self.upper_air_pressure[ind_start:ind_end].diff(dim=-1).to(q.device)

        if num_dims == 5:  # (batch_size, level, time, latitude, longitude)
            delta_p = delta_p.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            q_slice = q[:, ind_start:ind_end, ...]
            q_area = 0.5 * (q_slice[:, :-1, :, :, :] + q_slice[:, 1:, :, :, :]) * delta_p
            q_trapz = torch.sum(q_area, dim=1)

        elif num_dims == 4:  # (batch_size, level, latitude, longitude) or (time, level, latitude, longitude)
            delta_p = delta_p.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            q_slice = q[:, ind_start:ind_end, ...]
            q_area = 0.5 * (q_slice[:, :-1, :, :] + q_slice[:, 1:, :, :]) * delta_p  # Trapezoidal rule
            q_trapz = torch.sum(q_area, dim=1)

        elif num_dims == 3:  # (level, latitude, longitude)
            delta_p = delta_p.unsqueeze(-1).unsqueeze(-1)  # Expand for broadcasting
            q_slice = q[ind_start:ind_end, ...]
            q_area = 0.5 * (q_slice[:-1, :, :] + q_slice[1:, :, :]) * delta_p
            q_trapz = torch.sum(q_area, dim=0)

        else:
            raise ValueError(f"Unsupported tensor dimensions: {q.shape}")

        return q_trapz

    def weighted_sum(self, q: torch.Tensor, axis: Dict[tuple, None] = None, keepdims: bool = False) -> torch.Tensor:
        """
        Compute the weighted sum of a given quantity for PyTorch tensors.

        Args:
            q: the quantity to be summed (PyTorch tensor)
            axis: dims to compute the sum (can be int or tuple of ints)
            keepdims: whether to keep the reduced dimensions or not

        Returns:
            Weighted sum (PyTorch tensor)
        """
        # Cast to float64 for accumulation to avoid ~0.6% rounding error from
        # summing ~55k float32 values, then cast result back to input dtype.
        in_dtype = q.dtype
        q_w = q.double() * self.area.double().to(q.device)
        q_sum = torch.sum(q_w, dim=axis, keepdim=keepdims)
        return q_sum.to(in_dtype)

    def total_dry_air_mass(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the total mass of dry air over the entire globe [kg]
        """
        mass_dry_per_area = self.integral(1 - q) / GRAVITY  # kg/m^2
        # weighted sum on latitude and longitude dimensions
        mass_dry_sum = self.weighted_sum(mass_dry_per_area, axis=(-2, -1))  # kg

        return mass_dry_sum

    def total_column_water(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute total column water (TCW) per air column [kg/m2]
        """
        TWC = self.integral(q) / GRAVITY  # kg/m^2

        return TWC


class physics_hybrid_sigma_level:
    """
    Hybrid sigma-pressure level physics

    Attributes:
        lon (torch.Tensor): Longitude in degrees.
        lat (torch.Tensor): Latitude in degrees.
        surface_pressure (torch.Tensor): Surface pressure in Pa.
        coef_a (torch.Tensor): Hybrid sigma-pressure coefficient 'a' [Pa].
        coef_b (torch.Tensor): Hybrid sigma-pressure coefficient 'b' [unitless].
        area (torch.Tensor): Area of grid cells [m^2].
        integral (function): Vertical integration method (midpoint or trapezoidal).
    """

    def __init__(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor,
        coef_a: torch.Tensor,
        coef_b: torch.Tensor,
        midpoint: bool = False,
    ):
        """
        Initialize the class with longitude, latitude, and hybrid sigma-pressure levels.
        All inputs must be on the same torch device.
        Full order of dimensions: (batch, level, time, latitude, longitude)
        Accepted dimensions: (batch, level, latitude, longitude)

        Args:
            lon (torch.Tensor): Longitude in degrees.
            lat (torch.Tensor): Latitude in degrees.
            coef_a (torch.Tensor): Hybrid sigma-pressure coefficient 'a' [Pa] (level,).
            coef_b (torch.Tensor): Hybrid sigma-pressure coefficient 'b' [unitless] (level,).
            midpoint (bool): True if vertical level quantities are midpoint values; otherwise False.

        Note:
            pressure = coef_a + coef_b * surface_pressure
        """
        self.lon = lon
        self.lat = lat
        self.coef_a = coef_a  # (level,)
        self.coef_b = coef_b  # (level,)

        # ========================================================================= #
        # Compute pressure on each hybrid sigma level
        # Reshape coef_a and coef_b for broadcasting
        self.coef_a = coef_a.view(1, -1, 1, 1)  # (1, level, 1, 1)
        self.coef_b = coef_b.view(1, -1, 1, 1)  # (1, level, 1, 1)

        # ========================================================================= #
        # compute gtid area
        # area = R^2 * d_sin(lat) * d_lon
        lat_rad = torch.deg2rad(self.lat)
        lon_rad = torch.deg2rad(self.lon)
        sin_lat_rad = torch.sin(lat_rad)
        d_phi = torch.gradient(sin_lat_rad, dim=0, edge_order=2)[0]
        d_lambda = torch.gradient(lon_rad, dim=1, edge_order=2)[0]
        d_lambda = (d_lambda + torch.pi) % (2 * torch.pi) - torch.pi
        self.area = torch.abs(RAD_EARTH**2 * d_phi * d_lambda)

        # ========================================================================== #
        # Vertical integration method
        if midpoint:
            self.integral = self.pressure_integral_midpoint
            self.integral_sliced = self.pressure_integral_midpoint_sliced
        else:
            self.integral = self.pressure_integral_trapz
            self.integral_sliced = self.pressure_integral_trapz_sliced

    def pressure_integral_midpoint(
        self,
        q_mid: torch.Tensor,
        surface_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the pressure level integral of a given quantity; assuming its mid-point
        values are pre-computed.

        Args:
            q_mid: The quantity with dims of (batch, level-1, time, latitude, longitude)
            surface_pressure: Surface pressure in Pa (batch, time, latitude, longitude).

        Returns:
            Pressure level integrals of q
        """
        # (batch, 1, lat, lon)
        surface_pressure = surface_pressure.unsqueeze(1)

        # (batch, level, lat, lon)
        pressure = self.coef_a.to(q_mid.device) + self.coef_b.to(q_mid.device) * surface_pressure

        # (batch, level-1, lat, lon)
        delta_p = pressure.diff(dim=1).to(q_mid.device)

        # Element-wise multiplication
        q_area = q_mid * delta_p

        # Sum over level dimension
        q_integral = torch.sum(q_area, dim=1)

        return q_integral

    def pressure_integral_midpoint_sliced(
        self,
        q_mid: torch.Tensor,
        surface_pressure: torch.Tensor,
        ind_start: int,
        ind_end: int,
    ) -> torch.Tensor:
        """
        As in `pressure_integral_midpoint`, but supports pressure level indexing,
        so it can calculate integrals of a subset of levels.
        """
        # (batch, 1, lat, lon)
        surface_pressure = surface_pressure.unsqueeze(1)

        # (batch, level, lat, lon)
        pressure = self.coef_a.to(q_mid.device) + self.coef_b.to(q_mid.device) * surface_pressure

        # (batch, level-1, lat, lon)
        pressure_thickness = pressure.diff(dim=1)

        delta_p = pressure_thickness[:, ind_start:ind_end, ...].to(q_mid.device)

        q_mid_sliced = q_mid[:, ind_start:ind_end, ...]
        q_area = q_mid_sliced * delta_p
        q_integral = torch.sum(q_area, dim=1)
        return q_integral

    def pressure_integral_trapz(self, q: torch.Tensor, surface_pressure: torch.Tensor) -> torch.Tensor:
        """
        Compute the pressure level integral of a given quantity using the trapezoidal rule.

        Args:
            q: The quantity with dims of (batch, level, time, latitude, longitude)

        Returns:
            Pressure level integrals of q
        """
        # (batch, 1, lat, lon)
        surface_pressure = surface_pressure.unsqueeze(1)

        # (batch, level, lat, lon)
        pressure = self.coef_a.to(q.device) + self.coef_b.to(q.device) * surface_pressure

        # (batch, level-1, lat, lon)
        delta_p = pressure.diff(dim=1).to(q.device)

        # trapz
        q1 = q[:, :-1, ...]
        q2 = q[:, 1:, ...]
        q_area = 0.5 * (q1 + q2) * delta_p
        q_trapz = torch.sum(q_area, dim=1)

        return q_trapz

    def pressure_integral_trapz_sliced(
        self,
        q: torch.Tensor,
        surface_pressure: torch.Tensor,
        ind_start: int,
        ind_end: int,
    ) -> torch.Tensor:
        """
        As in `pressure_integral_trapz`, but supports pressure level indexing,
        so it can calculate integrals of a subset of levels.
        """
        # (batch, 1, lat, lon)
        surface_pressure = surface_pressure.unsqueeze(1)

        # (batch, level, lat, lon)
        pressure = self.coef_a.to(q.device) + self.coef_b.to(q.device) * surface_pressure

        delta_p = pressure[:, ind_start:ind_end, ...].diff(dim=1).to(q.device)

        # trapz
        q_slice = q[:, ind_start:ind_end, ...]
        q1 = q_slice[:, :-1, ...]
        q2 = q_slice[:, 1:, ...]
        q_area = 0.5 * (q1 + q2) * delta_p
        q_trapz = torch.sum(q_area, dim=1)

        return q_trapz

    def weighted_sum(self, q: torch.Tensor, axis: Dict[tuple, None] = None, keepdims: bool = False) -> torch.Tensor:
        """
        Compute the weighted sum of a given quantity for PyTorch tensors.

        Args:
            data: the quantity to be summed (PyTorch tensor)
            axis: dims to compute the sum (can be int or tuple of ints)
            keepdims: whether to keep the reduced dimensions or not

        Returns:
            Weighted sum (PyTorch tensor)
        """
        # Cast to float64 for accumulation to avoid ~0.6% rounding error from
        # summing ~55k float32 values, then cast result back to input dtype.
        in_dtype = q.dtype
        q_w = q.double() * self.area.double().to(q.device)
        q_sum = torch.sum(q_w, dim=axis, keepdim=keepdims)
        return q_sum.to(in_dtype)

    def total_dry_air_mass(self, q: torch.Tensor, surface_pressure: torch.Tensor) -> torch.Tensor:
        """
        Compute the total mass of dry air over the entire globe [kg]
        """
        mass_dry_per_area = self.integral(1 - q, surface_pressure) / GRAVITY  # kg/m^2
        # weighted sum on latitude and longitude dimensions
        mass_dry_sum = self.weighted_sum(mass_dry_per_area, axis=(-2, -1))  # kg

        return mass_dry_sum

    def total_column_water(
        self,
        q: torch.Tensor,
        surface_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total column water (TCW) per air column [kg/m2]
        """
        TWC = self.integral(q, surface_pressure) / GRAVITY  # kg/m^2

        return TWC
