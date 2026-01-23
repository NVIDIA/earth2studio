# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import datetime
import math
from enum import Enum
from functools import cache, partial
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import einops
    from natten import NeighborhoodAttention2D
    from physicsnemo import Module as PhysicsNeMoModule
    from timm.models.vision_transformer import Mlp, PatchEmbed
    from torch_harmonics import InverseRealSHT
except ImportError:
    OptionalDependencyFailure("atlas")
    einops = None
    PhysicsNeMoModule = object


@cache
def get_isht(nlat, nlon, grid="equiangular", norm="ortho", device=None):
    return InverseRealSHT(nlat, nlon, grid=grid, norm=norm).to(device=device)


def spherical_white_noise(shape, scale=3.56, device=None, studentt_deg=None):
    if not isinstance(shape, list):
        shape = list(shape)

    if len(shape) < 2:
        raise ValueError(f"Shape must have at least 2 dimensions, got {shape}")

    if studentt_deg is None:
        noise = torch.randn(shape[0:-1] + [(shape[-1] // 2) + 1] + [2], device=device)
    else:
        dist = torch.distributions.studentT.StudentT(studentt_deg)
        noise = dist.sample(shape[0:-1] + [(shape[-1] // 2) + 1] + [2]).to(device)

    noise = (1.0 / math.sqrt(float(shape[-1] * shape[-2]))) * scale * noise
    noise = torch.view_as_complex(noise)

    noise = torch.tril(noise)

    isht = get_isht(shape[-2], shape[-1], device=device)

    with torch.amp.autocast("cuda", enabled=False):
        return isht(noise)


def bilinear_interpolate_torch(x, shape):
    return torch.nn.functional.interpolate(
        x, size=shape, mode="bilinear", align_corners=True, antialias=False
    )


def _obliquity_star(julian_centuries: np.ndarray) -> np.ndarray:
    """
    return obliquity of the sun
    Use 5th order equation from
    https://en.wikipedia.org/wiki/Ecliptic#Obliquity_of_the_ecliptic
    """
    return np.deg2rad(
        23.0
        + 26.0 / 60
        + 21.406 / 3600.0
        - (
            46.836769 * julian_centuries
            - 0.0001831 * (julian_centuries**2)
            + 0.00200340 * (julian_centuries**3)
            - 0.576e-6 * (julian_centuries**4)
            - 4.34e-8 * (julian_centuries**5)
        )
        / 3600.0,
        dtype=np.float32,
    )


def _sun_ecliptic_longitude(model_time: np.ndarray) -> np.ndarray:
    """
    Ecliptic longitude of the sun.
    Reference:
        http://www.geoastro.de/elevaz/basics/meeus.htm
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0

    # mean anomaly calculation
    mean_anomaly = np.deg2rad(
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries * julian_centuries
        - 0.00000048 * julian_centuries * julian_centuries * julian_centuries,
        dtype=np.float32,
    )

    # mean longitude
    mean_longitude = np.deg2rad(
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * (julian_centuries**2),
        dtype=np.float32,
    )

    d_l = np.deg2rad(
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * (julian_centuries**2))
        * np.sin(mean_anomaly)
        + (0.019993 - 0.000101 * julian_centuries) * np.sin(2 * mean_anomaly)
        + 0.000290 * np.sin(3 * mean_anomaly),
        dtype=np.float32,
    )

    # true longitude
    return mean_longitude + d_l


def _right_ascension_declination(
    model_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Right ascension and declination of the sun.
    Ref:
        http://www.geoastro.de/elevaz/basics/meeus.htm
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0
    eps = _obliquity_star(julian_centuries.astype(np.float32))

    eclon = _sun_ecliptic_longitude(model_time)
    x = np.cos(eclon)
    y = np.cos(eps) * np.sin(eclon)
    z = np.sin(eps) * np.sin(eclon)
    r = np.sqrt(1.0 - z * z)
    # sun declination
    declination = np.arctan2(z, r)
    # right ascension
    right_ascension = np.float32(2.0 * np.arctan2(y, (x + r)))
    return right_ascension, declination


def _days_from_2000(model_time: np.ndarray) -> np.ndarray:
    """Get the days since year 2000."""
    # compute total days
    time_diff = model_time - datetime.datetime(2000, 1, 1, 12, 0)
    result = np.asarray(time_diff).astype("timedelta64[us]") / np.timedelta64(1, "D")
    result = result.astype(np.float32)

    return result


def _greenwich_mean_sidereal_time(model_time: np.ndarray) -> np.ndarray:
    """
    Greenwich mean sidereal time, in radians.
    Reference:
        The AIAA 2006 implementation:
            http://www.celestrak.com/publications/AIAA/2006-6753/
    """
    jul_centuries = _days_from_2000(model_time) / 36525.0
    theta = np.float32(
        67310.54841
        + jul_centuries
        * (
            876600 * 3600
            + 8640184.812866
            + jul_centuries * (0.093104 - jul_centuries * 6.2 * 10e-6)
        )
    )

    theta_radians = np.deg2rad(theta / 240.0) % (2 * np.pi)
    theta_radians = theta_radians.astype(np.float32)

    return theta_radians


def _local_mean_sidereal_time(
    model_time: np.ndarray, longitude: np.ndarray
) -> np.ndarray:
    """
    Local mean sidereal time. requires longitude in radians.
    Ref:
        http://www.setileague.org/askdr/lmst.htm
    """
    return _greenwich_mean_sidereal_time(model_time) + longitude


def _local_hour_angle(
    model_time: np.ndarray, longitude: np.ndarray, right_ascension: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hour angle at model_time for the given longitude and right_ascension
    longitude in radians. Return shape: [t, lon]
    Ref:
        https://en.wikipedia.org/wiki/Hour_angle#Relation_with_the_right_ascension
    """
    loc_mean = _local_mean_sidereal_time(model_time, longitude)

    # take the diff
    loc_hour_angle = loc_mean - right_ascension

    return loc_hour_angle


def _star_cos_zenith(
    model_time: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """
    Return cosine of star zenith angle
    lon,lat in radians. Return shape: [t, lat, lon]
    Ref:
        Azimuth:
            https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas
        Zenith:
            https://en.wikipedia.org/wiki/Solar_zenith_angle
    """
    # right ascension, only dependent on model times
    ra, dec = _right_ascension_declination(model_time)

    # compute local hour angle
    h_angle = _local_hour_angle(model_time, lon, ra)

    # compute zenith:
    cosine_zenith = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(
        h_angle
    )

    return cosine_zenith


def cos_zenith_angle(
    time: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """
    Cosine of sun-zenith angle for lon, lat at time (UTC).
    If DataArrays are provided for the lat and lon arguments, their units will
    be assumed to be in degrees, unless they have a units attribute that
    contains "rad"; in that case they will automatically be converted to having
    units of degrees.
    Args:
        time: time in UTC
        lon: float or np.ndarray in radians (E/W)
        lat: float or np.ndarray in radians (N/S)
    Returns:
        float, np.ndarray
    """
    # reshape all inputs
    lon_rad = np.expand_dims(lon, axis=0)
    lat_rad = np.expand_dims(lat, axis=0)
    time = np.reshape(time, (-1, 1, 1))

    result = _star_cos_zenith(time, lon_rad, lat_rad)

    return result


def alpha_linear(t):
    return 1.0 - t


def alpha_linear_dot(t):
    return -1.0 * torch.ones_like(t)


def beta_linear(t):
    return t


def beta_linear_dot(t):
    return torch.ones_like(t)


def beta_quad(t):
    return t**2


def beta_quad_dot(t):
    return 2 * t


def sigma_linear(t, epsilon):
    return epsilon * (1.0 - t)


def sigma_linear_dot(t, epsilon):
    return -epsilon * torch.ones_like(t)


def uniform_time_sample():
    return torch.rand(
        1,
    ).item()


def follmer_g(t, sigma, sigma_dot, beta, beta_dot):
    sig = sigma(t)
    sigd = sigma_dot(t)
    bet = beta(t)
    betd = beta_dot(t)

    return torch.sqrt(
        torch.abs(2.0 * t * sig * ((1.0 / bet) * betd * sig - sigd) - torch.pow(sig, 2))
    )


@check_optional_dependencies()
class StochasticInterpolant(torch.nn.Module):
    def __init__(
        self,
        alpha="alpha_linear",
        beta="beta_linear",
        sigma="sigma_linear",
        g=None,
        epsilon=1.0,
        noise_sampler="white",
        time_sampler="uniform",
        sample_method="em",
        studentt_deg=None,
    ):

        super().__init__()
        if alpha == "alpha_linear":
            self.alpha = alpha_linear
            self.alpha_dot = alpha_linear_dot
        else:
            raise NotImplementedError()

        if beta == "beta_linear":
            self.beta = beta_linear
            self.beta_dot = beta_linear_dot
        elif beta == "beta_quad":
            self.beta = beta_quad
            self.beta_dot = beta_quad_dot
        else:
            raise NotImplementedError()

        if sigma == "sigma_linear":
            self.sigma = partial(sigma_linear, epsilon=epsilon)
            self.sigma_dot = partial(sigma_linear_dot, epsilon=epsilon)
        else:
            raise NotImplementedError()

        if g is not None:
            self.g = g
            self.compute_correction = True
        else:
            self.g = self.sigma
            self.compute_correction = False

        if noise_sampler == "white":
            self.noise_sampler = torch.randn
        elif noise_sampler == "spherical_white":
            self.noise_sampler = partial(
                spherical_white_noise, studentt_deg=studentt_deg
            )
        else:
            raise NotImplementedError()

        if time_sampler == "uniform":
            self.time_sampler = uniform_time_sample
        else:
            raise NotImplementedError()

        if sample_method == "em":
            self.sample_step = self.em_step
        elif sample_method == "rk_roberts":
            self.sample_step = self.rk_roberts_step
        elif sample_method == "rk_rossler_a1":
            coeff = {"c1": [1.0, 0.0], "beta1": [1.0, 0.0], "beta2": [-1.0, 1.0]}
            self.sample_step = partial(self.rk_rossler_step, coeff=coeff)
        elif sample_method == "rk_rossler_a2":
            coeff = {
                "c1": [1.0 / 3.0, 1.0],
                "beta1": [0.0, 1.0],
                "beta2": [-3.0 / 2.0, 3.0 / 2.0],
            }
            self.sample_step = partial(self.rk_rossler_step, coeff=coeff)
        else:
            raise NotImplementedError()

    def brownian_path(self, shape, t):

        expand_shape = [t.shape[0]] + [1] * len(shape)

        first_t = t[:, 0].view(expand_shape)
        out = self.noise_sampler(t.shape + shape, device=t.device)
        out[:, 0, ...] *= torch.sqrt(first_t)

        if t.shape[1] > 1:
            for j in range(1, t.shape[1]):
                out[:, j, ...] *= torch.sqrt(t[:, j] - t[:, j - 1]).view(expand_shape)
                out[:, j, ...] += out[:, j - 1, ...]

        return out

    def stochastic_path(self, x0, x1, t, return_derivative=True):

        W = self.brownian_path(x0.shape[1:], t)

        expand_shape = list(t.shape) + [1] * (len(x0.shape[1:]))

        t = t.view(expand_shape)
        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)

        I = self.alpha(t) * x0 + self.beta(t) * x1 + self.sigma(t) * W  # noqa: E741

        if return_derivative:
            R = self.alpha_dot(t) * x0 + self.beta_dot(t) * x1 + self.sigma_dot(t) * W

            return I, R

        return I

    def compute_score(self, initial, current, drift, t, return_sigma=False):
        alpha = self.alpha(t)
        alpha_dot = self.alpha_dot(t)
        beta = self.beta(t)
        beta_dot = self.beta_dot(t)
        sigma = self.sigma(t)
        sigma_dot = self.sigma_dot(t)

        A = 1.0 / (t * sigma * (beta_dot * sigma - beta * sigma_dot))
        c = beta_dot * current + (beta * alpha_dot - beta_dot * alpha) * initial
        score = A * (beta * drift - c)

        if return_sigma:
            return score, sigma

        return score

    def compute_normalization(self, t):
        alpha = self.alpha(t)
        alpha_dot = self.alpha_dot(t)
        beta = self.beta(t)
        beta_dot = self.beta_dot(t)
        sigma = self.sigma(t)
        sigma_dot = self.sigma_dot(t)

        c_S = alpha_dot
        c_O = torch.sqrt(beta_dot**2 + (sigma_dot**2) * t)
        c_I = 1.0 / torch.sqrt(alpha**2 + beta**2 + (sigma**2) * t)

        return {"c_S": c_S, "c_O": c_O, "c_I": c_I}

    def compute_drift(
        self,
        drift_velocity,
        x,
        t,
        cond=None,
        compute_normalization=False,
        compute_correction=False,
        g=None,
        expand_shape=None,
        initial=None,
    ):
        vel_inp = {"I": x, "t": t}
        if cond is not None:
            if isinstance(cond, dict):
                vel_inp = vel_inp | cond
            else:
                vel_inp["I"] = torch.cat((vel_inp["I"], cond), dim=1)

        if compute_normalization:
            vel_inp = vel_inp | self.compute_normalization(t)

        drift = drift_velocity(**vel_inp)

        if compute_correction:
            t = t.view(expand_shape)
            score, sigma = self.compute_score(initial, x, drift, t, return_sigma=True)

            drift = drift + 0.5 * (torch.pow(g, 2) - torch.pow(sigma, 2)) * score

        return drift

    def em_step(
        self,
        drift_velocity,
        x,
        t,
        step,
        expand_shape,
        cond=None,
        compute_normalization=False,
        initial=None,
    ):
        previous_t = t[:, step].view(-1, 1)
        delta_t = (t[:, step + 1] - t[:, step]).view(expand_shape)

        if step == 0:
            g = self.sigma(previous_t).view(expand_shape)
            compute_correction = False
        else:
            g = self.g(previous_t).view(expand_shape)
            compute_correction = self.compute_correction

        drift = self.compute_drift(
            drift_velocity,
            x,
            previous_t,
            cond,
            compute_normalization,
            compute_correction,
            g,
            expand_shape,
            initial,
        )

        x = x + delta_t * drift
        x = x + torch.sqrt(delta_t) * g * self.noise_sampler(x.shape, device=x.device)

        return x

    def rk_roberts_step(
        self,
        drift_velocity,
        x,
        t,
        step,
        expand_shape,
        cond=None,
        compute_normalization=False,
        initial=None,
    ):
        previous_t = t[:, step].view(-1, 1)
        current_t = t[:, step + 1].view(-1, 1)
        delta_t = (t[:, step + 1] - t[:, step]).view(expand_shape)

        if step == 0:
            g_previous = self.sigma(previous_t).view(expand_shape)
            g_current = self.sigma(current_t).view(expand_shape)
            compute_correction = False
        else:
            g_previous = self.g(previous_t).view(expand_shape)
            g_current = self.g(current_t).view(expand_shape)
            compute_correction = self.compute_correction

        S = np.random.binomial(n=1, p=0.5, size=1).item() - 0.5
        noise = torch.sqrt(delta_t) * self.noise_sampler(x.shape, device=x.device)

        drift_previous = self.compute_drift(
            drift_velocity,
            x,
            previous_t,
            cond,
            compute_normalization,
            compute_correction,
            g_previous,
            expand_shape,
            initial,
        )

        y1 = delta_t * drift_previous + g_previous * (noise - torch.sqrt(delta_t) * S)

        drift_current = self.compute_drift(
            drift_velocity,
            x + y1,
            current_t,
            cond,
            compute_normalization,
            compute_correction,
            g_current,
            expand_shape,
            initial,
        )

        y2 = delta_t * drift_current + g_current * (noise + torch.sqrt(delta_t) * S)

        x = x + 0.5 * (y1 + y2)

        return x

    def rk_rossler_step(
        self,
        drift_velocity,
        x,
        t,
        step,
        expand_shape,
        cond=None,
        compute_normalization=False,
        initial=None,
        coeff=None,
    ):
        t_n = t[:, step].view(-1, 1)
        delta_t = (t[:, step + 1] - t[:, step]).view(-1, 1)
        t_n_1 = t_n + coeff["c1"][0] * delta_t
        t_n_2 = t_n + coeff["c1"][1] * delta_t
        t_n_3 = t_n + (3.0 / 4.0) * delta_t
        delta_t = delta_t.view(expand_shape)

        if step == 0:
            g_n = self.sigma(t_n).view(expand_shape)
            g_n_1 = self.sigma(t_n_1).view(expand_shape)
            g_n_2 = self.sigma(t_n_2).view(expand_shape)
            g_n_3 = self.sigma(t_n_3).view(expand_shape)
            compute_correction = False
        else:
            g_n = self.g(t_n).view(expand_shape)
            g_n_1 = self.g(t_n_1).view(expand_shape)
            g_n_2 = self.g(t_n_2).view(expand_shape)
            g_n_3 = self.g(t_n_3).view(expand_shape)
            compute_correction = self.compute_correction

        r_n = torch.sqrt(delta_t) * self.noise_sampler(x.shape, device=x.device)
        r_n_tilde = 0.5 * (
            r_n
            + (1.0 / math.sqrt(3.0))
            * torch.sqrt(delta_t)
            * self.noise_sampler(x.shape, device=x.device)
        )

        drift_1 = self.compute_drift(
            drift_velocity,
            x,
            t_n,
            cond,
            compute_normalization,
            compute_correction,
            g_n,
            expand_shape,
            initial,
        )

        h_n = x + (3.0 / 4.0) * delta_t * drift_1 + (3.0 / 2.0) * g_n_1 * r_n_tilde

        drift_2 = self.compute_drift(
            drift_velocity,
            h_n,
            t_n_3,
            cond,
            compute_normalization,
            compute_correction,
            g_n_3,
            expand_shape,
            initial,
        )

        x = (
            x
            + (1.0 / 3.0) * delta_t * drift_1
            + (2.0 / 3.0) * delta_t * drift_2
            + (coeff["beta1"][0] * r_n + coeff["beta2"][0] * r_n_tilde) * g_n_1
            + (coeff["beta1"][1] * r_n + coeff["beta2"][1] * r_n_tilde) * g_n_2
        )

        return x

    def sample(
        self,
        drift_velocity,
        x,
        t=None,
        steps=200,
        cond=None,
        compute_normalization=False,
        keep_every=None,
        verbose=False,
    ):

        initial = x.clone()

        if t is None:
            t = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
            t = t.view(1, -1)
            t = torch.repeat_interleave(t, x.shape[0], dim=0)
        else:
            steps = t.shape[1] - 1

        expand_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)

        x = self.sample_step(
            drift_velocity, x, t, 0, expand_shape, cond, compute_normalization, initial
        )

        kept_samples = 0
        if keep_every is not None:
            out = torch.zeros(
                [x.shape[0], steps // keep_every] + list(x.shape[1:]), device=x.device
            )

            if keep_every == 1:
                out[:, 0, ...] = x
                kept_samples += 1

        for j in range(1, t.shape[1] - 1):
            physical_time = default_timer()

            x = self.sample_step(
                drift_velocity,
                x,
                t,
                j,
                expand_shape,
                cond,
                compute_normalization,
                initial,
            )

            if keep_every is not None:
                if (j + 1) % keep_every == 0:
                    out[:, kept_samples, ...] = x
                    kept_samples += 1

            if verbose:
                print(f"Sampling step: {j+1}. Time: {default_timer() - physical_time}")

        if keep_every is not None:
            return out

        return x


class EquiangularInterpolator(nn.Module):
    def __init__(self, method="bilinear_torch", **kwargs):
        super().__init__()

        self.interp_func = None
        self.method = None

        self.set_method(method)

    def set_method(self, method):
        if method == "conservative":
            self.interp_func = None
        elif method == "bilinear":
            self.interp_func = None
        elif method == "bilinear_torch":
            self.interp_func = bilinear_interpolate_torch
        else:
            raise NotImplementedError(
                f"Interpolation method {method} is not supported."
            )

        self.method = method

    def forward(self, x, shape):
        return self.interp_func(x, shape)


class GaussianNormalizer(nn.Module):
    def __init__(self, stats_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.empty(stats_shape))
        self.register_buffer("std", torch.empty(stats_shape))

    def set_stats(self, mean, std):
        if not torch.is_tensor(mean):
            mean = torch.from_numpy(mean).to(dtype=torch.float32)
            std = torch.from_numpy(std).to(dtype=torch.float32)

        self.mean.data = mean
        self.std.data = std

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, x, multiplier=1):
        return (x - self.mean * multiplier) / (self.std + self.eps)

    def unnormalize(self, x, multiplier=1):
        return (x * (self.std + self.eps)) + self.mean * multiplier


class NattenAttention(nn.Module):
    """A neighborhood attention module with circular padding and natten."""

    def __init__(
        self,
        kernel_size: tuple[int, int],
        dim: int,
        num_heads: int,
        # The following are all standard
        qkv_bias=True,
        # if 'True', circularly pad the input
        # gets sufficient context in the East/West
        # direction.
        # this is probably useful if you are performing
        # prediction with this block, but likely less
        # useful for autoencoding
        circular_pad_width: bool = False,
    ):
        """Neighborhood attention."""
        super().__init__()
        self.kernel_size = kernel_size

        kernel_size_h, kernel_size_w = kernel_size

        self.circular_pad_width = circular_pad_width

        if circular_pad_width:
            pad_size = (kernel_size_w - 1) // 2
        else:
            pad_size = 0
        self.attn = NeighborhoodAttention2D(
            dim,
            num_heads=num_heads,
            kernel_size=[kernel_size_h, kernel_size_w],
            qkv_bias=qkv_bias,
        )

        self.circ_pad_reverse_dim_order = (
            0,
            0,  # channel dim
            pad_size,
            pad_size,  # width dim
            0,
            0,  # height dim
        )

        self.crop_pad = (
            0,
            0,  # channel dim
            -pad_size,
            -pad_size,  # width dim
            0,
            0,  # height dim
        )

    def forward(self, x: torch.Tensor):
        """x is of shape b h w c, from tokenizer"""

        b, h, w, c = x.shape
        x = F.pad(x, self.circ_pad_reverse_dim_order, mode="circular")

        x = self.attn(x)
        # center_crop
        x = F.pad(x, self.crop_pad)

        return x


class NattenDiTBlock(nn.Module):
    """A DiT Block with Natten Attention."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        hidden_size: int,
        num_heads: int,
        kernel_size: tuple[int, int] = (7, 7),
        # This following are all standard
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        # for traditional DiT Block drop params are 0.0
        act_layer=partial(nn.GELU, approximate="tanh"),
        norm_layer=partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6),
        attn_drop_rate=0.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        attn_mask=None,
        use_swiglu=False,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.attn = NattenAttention(
            kernel_size,
            hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.norm1 = norm_layer(hidden_size)
        self.norm2 = norm_layer(hidden_size)
        self.drop_path = nn.Identity()

        mlp_hidden_size = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_size,
            act_layer=act_layer,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        b, t, c = x.shape
        h, w = self.grid_size

        # TODO(sumanr): check if this is correct,
        # especially with drop_path
        res = self.adaLN_modulation(cond).chunk(6, dim=1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = res

        y = modulate(self.norm1(x), shift_attn, scale_attn)
        z = gate_attn.unsqueeze(1) * self.attn(y.view(b, h, w, c)).view(b, h * w, c)
        x = x.view(b, h * w, c) + self.drop_path(z)

        y = modulate(self.norm2(x), shift_mlp, scale_mlp)
        z = gate_mlp.unsqueeze(1) * self.mlp(y)
        x = x + self.drop_path(z)

        return x


class AttentionType(Enum):
    ORIGINAL = "original"
    TEAttention = "te_attention"
    NATTEN = "natten"
    MULTI_AXIS = "multi_axis"
    TENSORIZED = "tensorized"


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FourierEmbedder(nn.Module):
    """
    Embeds scalar or vector into a cos-sin representation.
    """

    def __init__(
        self,
        out_dim,
        frequency_embedding_dim=256,
        input_multiplier=1000,
        max_period=10000,
    ):
        super().__init__()

        self.frequency_embedding_dim = frequency_embedding_dim
        self.input_multiplier = input_multiplier
        self.max_period = max_period

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_dim, out_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(out_dim, out_dim, bias=True),
        )

    def cos_sin_embedding(self, x):
        """
        Create cos-sin embeddings of a vector.
        Conventional time-step embedding on each
        dimension of the vector. Adapted from:
        https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        half = (self.frequency_embedding_dim // x.shape[-1]) // 2
        freqs = (
            torch.exp(
                -math.log(self.max_period)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / half
            )
            .to(device=x.device)
            .unsqueeze(1)
        )

        args = x[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-2)
        embedding = einops.rearrange(embedding, "... a b -> ... (a b)")

        emb_diff = self.frequency_embedding_dim - embedding.shape[-1]
        if emb_diff != 0:
            embedding = torch.cat(
                (embedding, torch.zeros(embedding.shape[0], emb_diff, device=x.device)),
                dim=-1,
            )

        return embedding

    def forward(self, x):
        x = self.input_multiplier * x
        x = self.cos_sin_embedding(x)
        x = self.mlp(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[torch.nn.Module] = torch.nn.LayerNorm,
        bfloat_cast: bool = True,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.bfloat_cast = bfloat_cast

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else torch.nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else torch.nn.Identity()
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.bfloat_cast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                x = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
        else:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                x = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )

        x = x.to(torch.float32)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_dim, num_heads=None, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_dim, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = partial(torch.nn.GELU, approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm_final = torch.nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.linear = torch.nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_dit_block(natten=False, kernel_size=None, grid_size=None, bfloat_cast=False):
    if natten:
        block = partial(NattenDiTBlock, grid_size=grid_size, kernel_size=kernel_size)
    else:
        block = partial(DiTBlock, bfloat_cast=bfloat_cast)

    return block


@check_optional_dependencies()
class SInterpolantLatentDiT(PhysicsNeMoModule):
    def __init__(
        self,
        input_shape_1=[181, 360],
        input_shape_2=[721, 1440],
        input_channels_1=75,
        input_channels_2=79,
        embed_dim_1=1152,
        embed_dim_2=1152,
        num_patches=[46, 90],
        natten_kernel=None,
        output_channels=75,
        depth=16,
        num_heads=None,
        mlp_ratio=4.0,
        checkpoint=None,
        num_classes=None,
        bfloat_cast=True,
        **kwargs,
    ):
        super().__init__()

        self.input_shape_1 = input_shape_1
        self.input_shape_2 = input_shape_2
        self.input_shape_1_dit = input_shape_1.copy()
        self.input_shape_2_dit = input_shape_2.copy()
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.checkpoint = depth + 1 if checkpoint is None else checkpoint

        # Padding for input 1
        if input_shape_1[0] % num_patches[0] != 0:
            pad_rows_1 = num_patches[0] - (input_shape_1[0] % num_patches[0])
            self.input_shape_1_dit[0] += pad_rows_1
            self.pad_rows_1 = torch.nn.ReflectionPad2d((0, 0, 0, pad_rows_1))
        else:
            self.pad_rows_1 = None

        if input_shape_1[1] % num_patches[1] != 0:
            pad_cols_1 = num_patches[1] - (input_shape_1[1] % num_patches[1])
            self.input_shape_1_dit[1] += pad_cols_1
            self.pad_cols_1 = torch.nn.CircularPad2d((0, pad_cols_1, 0, 0))
        else:
            self.pad_cols_1 = None

        # Padding for input 2
        if input_shape_2[0] % num_patches[0] != 0:
            pad_rows_2 = num_patches[0] - (input_shape_2[0] % num_patches[0])
            self.input_shape_2_dit[0] += pad_rows_2
            self.pad_rows_2 = torch.nn.ReflectionPad2d((0, 0, 0, pad_rows_2))
        else:
            self.pad_rows_2 = None

        if input_shape_2[1] % num_patches[1] != 0:
            pad_cols_2 = num_patches[1] - (input_shape_2[1] % num_patches[1])
            self.input_shape_2_dit[1] += pad_cols_2
            self.pad_cols_2 = torch.nn.CircularPad2d((0, pad_cols_2, 0, 0))
        else:
            self.pad_cols_2 = None

        # Patch embeddings
        self.patch_size_1 = (
            self.input_shape_1_dit[0] // num_patches[0],
            self.input_shape_1_dit[1] // num_patches[1],
        )
        self.patch_size_2 = (
            self.input_shape_2_dit[0] // num_patches[0],
            self.input_shape_2_dit[1] // num_patches[1],
        )
        embed_dim = embed_dim_1 + embed_dim_2

        self.x_embedder_1 = PatchEmbed(
            self.input_shape_1_dit,
            self.patch_size_1,
            input_channels_1,
            embed_dim_1,
            bias=True,
        )
        self.x_embedder_2 = PatchEmbed(
            self.input_shape_2_dit,
            self.patch_size_2,
            input_channels_2,
            embed_dim_2,
            bias=True,
        )
        self.t_embedder = FourierEmbedder(embed_dim)

        if num_classes is not None:
            self.embedding_table = torch.nn.Embedding(num_classes, embed_dim)
        else:
            self.embedding_table = None

        # Positional embeddings
        self.pos_embed_1 = torch.nn.Parameter(
            torch.zeros(1, self.x_embedder_1.num_patches, embed_dim_1),
            requires_grad=False,
        )
        self.pos_embed_2 = torch.nn.Parameter(
            torch.zeros(1, self.x_embedder_2.num_patches, embed_dim_2),
            requires_grad=False,
        )

        # DiT blocks
        if natten_kernel is not None:
            block = get_dit_block(True, natten_kernel, self.x_embedder_1.grid_size)
        else:
            block = get_dit_block(natten=False, bfloat_cast=bfloat_cast)

        self.blocks = torch.nn.ModuleList(
            [
                block(hidden_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(
            embed_dim, self.patch_size_1[0] * self.patch_size_1[1] * output_channels
        )

    def unpatchify(self, x):
        c = self.output_channels
        p1 = self.patch_size_1[0]
        p2 = self.patch_size_1[1]
        h = self.input_shape_1_dit[0]
        w = self.input_shape_1_dit[1]

        x = x.reshape(shape=(x.shape[0], h // p1, w // p2, p1, p2, c))
        return einops.rearrange(x, "a b c d e f -> a f (b d) (c e)")

    def forward(
        self,
        I,  # noqa: E741
        t,
        x_1,
        x_2,
        c_S=0.0,
        c_O=1.0,
        c_I=1.0,
        label=None,
        **kwargs,
    ):  # noqa: E741

        if isinstance(c_I, float):
            I = torch.cat((c_I * I, x_1.clone()), dim=1)  # noqa: E741
        else:
            I = torch.cat(  # noqa: E741
                (
                    c_I.view([c_I.shape[0]] + [1] * (len(x_1.shape) - 1))
                    * I,  # noqa: E741
                    x_1.clone(),
                ),
                dim=1,
            )  # noqa: E741

        # Padding
        if self.pad_rows_1 is not None:
            I = self.pad_rows_1(I)  # noqa: E741
        if self.pad_cols_1 is not None:
            I = self.pad_cols_1(I)  # noqa: E741

        if self.pad_rows_2 is not None:
            x_2 = self.pad_rows_2(x_2)
        if self.pad_cols_2 is not None:
            x_2 = self.pad_cols_2(x_2)

        # Embeddings
        I = self.x_embedder_1(I) + self.pos_embed_1  # noqa: E741
        x_2 = self.x_embedder_2(x_2) + self.pos_embed_2
        x = torch.cat((I, x_2), dim=-1)

        # Conditioning
        t = self.t_embedder(t)

        # print(t.shape)

        if self.embedding_table is not None and label is not None:
            t = t + self.embedding_table(label)

        # print(t.shape)
        # Apply DiT blocks
        for j, block in enumerate(self.blocks):
            if j >= self.checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, t, use_reentrant=False)
            else:
                x = block(x, t)

        x = self.final_layer(x, t)
        x = self.unpatchify(x)

        # Removing padding
        x = x[..., : self.input_shape_1[0], : self.input_shape_1[1]]

        # Skip
        if isinstance(c_S, float):
            x_1 = c_S * x_1
        else:
            x_1 = c_S.view([c_S.shape[0]] + [1] * (len(x_1.shape) - 1)) * x_1

        if isinstance(c_O, float):
            x = c_O * x
        else:
            x = c_O.view([c_O.shape[0]] + [1] * (len(x.shape) - 1)) * x

        x = x_1 + x

        return x


def _ensure_int_tuple(tup, n_elements=2):
    if isinstance(tup, int):
        tup = (tup,) * n_elements
    return tup


def validate_patch_size(input_size, patch_size=None, n_patch=None):
    """Returns valid sizes, raises error if parameters incompatible"""
    if patch_size is None and n_patch is None:
        raise ValueError(
            "You passed both patch_size=None and n_patch=None. One should be defined as a tuple of ints."
        )

    if patch_size is not None and n_patch is not None:
        raise ValueError(
            f"You passed both {patch_size=} and {n_patch=}. Only one be defined, as a tuple of ints."
        )

    if patch_size is None:
        patch_size = _ensure_int_tuple(patch_size, 2)

        patch_size = (
            input_size[0] // n_patch[0] + bool(input_size[0] % n_patch[0]),
            input_size[1] // n_patch[1] + bool(input_size[1] % n_patch[1]),
        )
    else:
        n_patch = _ensure_int_tuple(n_patch, 2)

        n_patch = (
            input_size[0] // patch_size[0] + bool(input_size[0] % patch_size[0]),
            input_size[1] // patch_size[1] + bool(input_size[1] % patch_size[1]),
        )

    target_size = (patch_size[0] * n_patch[0], patch_size[1] * n_patch[1])

    return patch_size, n_patch, target_size


class PatchPad(nn.Module):
    def __init__(self, input_size, target_size):
        super().__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.lat_pad, self.lon_pad = abs(target_size[0] - input_size[0]), abs(
            target_size[1] - input_size[1]
        )

    def forward(self, x):
        if self.lat_pad:
            x = F.pad(x, pad=(0, 0, 0, self.lat_pad), mode="reflect")
        if self.lon_pad:
            x = F.pad(x, pad=(0, self.lon_pad, 0, 0), mode="circular")
        return x


class PatchUnpad(nn.Module):
    def __init__(self, input_size, target_size):
        super().__init__()
        self.input_size = input_size
        self.target_size = target_size
        if input_size[0] < target_size[0]:
            raise ValueError(
                f"Input_size[0] {input_size} is smaller than target_size[0] {target_size}, cannot unpad."
            )
        if input_size[1] < target_size[1]:
            raise ValueError(
                f"Input_size[1] {input_size} is smaller than target_size[1] {target_size}, cannot unpad."
            )

    def forward(self, x):
        return x[..., : self.target_size[0], : self.target_size[1]]


@check_optional_dependencies()
class NattenCombineDiT(PhysicsNeMoModule):
    """
    Diffusion model with a Transformer backbone.
    """

    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        input_size1=(90, 181),
        input_size2=(90, 181),
        input_channels1=3,
        input_channels2=3,
        hidden_channels1=1152,
        hidden_channels2=1152,
        patch_size=None,  # Either specify patch_size or n_patch, int or (int, int)
        n_patch=None,
        output_channels=None,  # If None, = input_channels
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        patch_processing="pad",  # resample
        patch_processing_add_conv=True,  # If resample only
        date_condition=False,
        combination_mode="token_addition",  # 'token_adition', 'token_multiplication'
        use_natten=True,
        # natten_kernel_size=(3, 3),
        kernel_size=3,
        checkpoint=None,  # it int > 0, checkpoint every n blocks
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads

        output_channels = (
            output_channels if output_channels is not None else input_channels1
        )
        self.output_channels = output_channels

        self.use_padding = False
        self.use_resampling = False

        natten_kernel_size = (kernel_size, kernel_size)

        # Embedding for first input
        input_size1 = _ensure_int_tuple(input_size1, 2)
        self.input_size1 = input_size1

        patch_size, n_patch, latent_size = validate_patch_size(
            input_size=input_size1, patch_size=patch_size, n_patch=n_patch
        )
        self.patch_size = patch_size
        self.n_patch = n_patch
        self.latent_size = latent_size

        self.combination_mode = combination_mode
        self.hidden_channels1 = hidden_channels1
        self.hidden_channels2 = hidden_channels2

        if patch_processing == "pad":
            self.preprocess1 = PatchPad(input_size1, target_size=latent_size)
            self.postprocess = PatchUnpad(latent_size, target_size=input_size1)
        elif patch_processing == "resample":
            self.preprocess1 = nn.Identity()
            self.postprocess = nn.Identity()
        self.x_embedder1 = PatchEmbed(
            latent_size, patch_size, input_channels1, hidden_channels1, bias=True
        )
        self.pos_embed1 = nn.Parameter(
            torch.zeros(1, self.x_embedder1.num_patches, hidden_channels1),
            requires_grad=False,
        )

        # Embedding for second input: NOTE that we use the same postprocessing
        input_size2 = _ensure_int_tuple(input_size2, 2)
        # Should have the same number of patches as the first one
        patch_size2, _, latent_size = validate_patch_size(
            input_size=input_size2, patch_size=None, n_patch=n_patch
        )
        self.input_size2 = input_size2
        if patch_processing == "pad":
            self.preprocess2 = PatchPad(input_size2, target_size=latent_size)
        elif patch_processing == "resample":
            self.preprocess2 = nn.Identity()  # removed as unused
        self.x_embedder2 = PatchEmbed(
            latent_size, patch_size2, input_channels2, hidden_channels2, bias=True
        )
        self.pos_embed2 = nn.Parameter(
            torch.zeros(1, self.x_embedder2.num_patches, hidden_channels2),
            requires_grad=False,
        )

        # Combine the two embeddings: results in hidden-channels sum of the two
        if (
            self.combination_mode == "token_concatenation"
            or self.combination_mode == "token_addition"
            or self.combination_mode == "token_multiplication"
        ):
            hidden_channels = hidden_channels1
        elif self.combination_mode == "channel_concatenation":
            hidden_channels = hidden_channels1 + hidden_channels2
        self.hidden_channels = hidden_channels

        # We always have a time embedding
        self.t_embedder = FourierEmbedder(hidden_channels)
        if date_condition:
            self.date_embedder = nn.Identity()  # removed as unused
        else:
            self.date_embedder = None

        block = self._get_dit_block(use_natten, natten_kernel_size)
        self.blocks = nn.ModuleList(
            [
                block(
                    hidden_size=hidden_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_channels, patch_size[0] * patch_size[1] * output_channels
        )
        self.checkpoint = checkpoint

    def _get_dit_block(
        self, use_natten: bool, natten_kernel_size: tuple[int, int] | None = None
    ):
        self.use_natten = use_natten
        if use_natten:
            self.natten_kernel_size = natten_kernel_size
            block = partial(
                NattenDiTBlock,
                grid_size=self.x_embedder1.grid_size,
                kernel_size=natten_kernel_size,
            )
        else:
            block = DiTBlock

        return block

    def upsample(self, x, resample_shape):
        if self.upsampler is not None:
            x = self.upsampler(x, resample_shape)
            x = self.up_conv(x)
        return x

    def unpatchify(self, x):
        c = self.output_channels
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        n1 = self.n_patch[0]
        n2 = self.n_patch[1]

        x = x.reshape(shape=(x.shape[0], n1, n2, p1, p2, c))
        return einops.rearrange(x, "a b c d e f -> a f (b d) (c e)")

    def forward(self, x_1, x_2, t=None, date=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, seq_length, seq_dim) tensor input
        t: (N,) tensor of diffusion timesteps
        """
        # Pad or resample if needed
        x_1 = self.preprocess1(x_1)
        x_2 = self.preprocess2(x_2)

        # First, embed the patches + add fixed positional embedding:
        x_1 = (
            self.x_embedder1(x_1) + self.pos_embed1
        )  # (N, seq_length, seq_dim) -> (N, seq_length, hidden_dim)
        x_2 = (
            self.x_embedder2(x_2) + self.pos_embed2
        )  # (N, seq_length, seq_dim) -> (N, seq_length, hidden_dim)

        # x = torch.cat((x_1, x_2), dim=-1)

        #################################################
        # Recombination with x_t
        if (
            self.combination_mode == "token_addition"
            and self.hidden_channels1 == self.hidden_channels2
        ):
            x = x_1 + x_2
        elif (
            self.combination_mode == "token_multiplication"
            and self.hidden_channels1 == self.hidden_channels2
        ):
            x = x_1 * x_2
        elif self.combination_mode == "token_concatenation":
            x = torch.cat((x_1, x_2), dim=1)
        elif (
            self.combination_mode == "channel_concatenation"
            and x_1.shape[1] == x_2.shape[1]
        ):
            x = torch.cat((x_1, x_2), dim=-1)

        # Add optional time and date embeddings
        if t is None:
            t = torch.ones(x.shape[0], 1, device=x.device).view(-1)
        # time conditioning
        conditioning = self.t_embedder(t)  # (N, hidden_dim)
        # date conditioning
        if self.date_embedder is not None and date is not None:
            conditioning = conditioning + self.date_embedder(date)

        for j, block in enumerate(self.blocks):
            if self.checkpoint is not None and (j + 1) % self.checkpoint == 0:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, conditioning, use_reentrant=False
                )
            else:
                x = block(x, conditioning)

        x = self.final_layer(x, conditioning)  # (N, seq_length, out_dim)

        x = self.unpatchify(x)

        # Unpad or resample
        x = self.postprocess(x)

        return x


class BaseProcessor(PhysicsNeMoModule):
    def __init__(
        self,
        normalization="gaussian",
        normalizer_stats_shape=None,
        normalizer_eps=1e-6,
        normalizer_range=[-1, 1],
        positional_encoding=None,
        static_channels_shape=None,
        cosine_zenith_angle=False,
        input_grid_shape=(721, 1440),
        regrid_to_healpix: bool = False,
        healpix_level: int = 8,
        device=None,
    ):
        super().__init__()

        self.positional_encoding = positional_encoding
        self.cosine_zenith_angle = cosine_zenith_angle

        if normalization is None:
            self.normalizer_in = None
            self.normalizer_out = None
        elif normalization.lower() == "gaussian":
            self.normalizer_in = GaussianNormalizer(
                stats_shape=normalizer_stats_shape, eps=normalizer_eps
            )
            self.normalizer_out = GaussianNormalizer(
                stats_shape=normalizer_stats_shape, eps=normalizer_eps
            )
        elif normalization.lower() == "range":
            # removed as unused
            self.normalizer_in = nn.Identity()
            self.normalizer_out = nn.Identity()
        else:
            raise ValueError(
                f'normalization has to be one of ["gaussian", "range"], but got {normalization=}'
            )

        if cosine_zenith_angle:
            lon = np.linspace(0, 2 * np.pi, input_grid_shape[1] + 1, dtype=np.float32)[
                :-1
            ]
            lat = np.linspace(
                -np.pi / 2, np.pi / 2, input_grid_shape[0], dtype=np.float32
            )
            self.lon_grid, self.lat_grid = np.meshgrid(lon, lat)

        if regrid_to_healpix:
            pass
        else:
            self.healpix_regridder = None
            self.equiangular_regridder = None

        if static_channels_shape is not None:
            self.register_buffer("static_channels", torch.empty(static_channels_shape))
        else:
            self.register_buffer("static_channels", None)

        self.to(device)
        self.register_buffer("device_buffer", torch.empty([0]))

    @property
    def device(self):
        return self.device_buffer.device

    def add_cosine_zenith_angle(self, x, x_dates):
        if not isinstance(x_dates, np.ndarray):
            if x.shape[0] == 1:
                x_dates = [x_dates]
            else:
                raise ValueError(
                    f"Got a single date for a batch of {x.shape[0]} samples"
                )
        cos_zenith_angles = [
            torch.from_numpy(cos_zenith_angle(date, self.lon_grid, self.lat_grid))
            for date in x_dates
        ]
        cos_zenith_angles = torch.stack(cos_zenith_angles).to(x.device)
        return torch.cat((x, cos_zenith_angles), dim=1)

    def set_static_channels(self, static_channels):
        self.static_channels.data = static_channels

    def set_normalizer_in_stats(self, **kwargs):
        self.normalizer_in.set_stats(**kwargs)

    def set_normalizer_out_stats(self, **kwargs):
        self.normalizer_out.set_stats(**kwargs)

    def add_static_channels(self, x):
        if self.static_channels is not None:
            return torch.cat(
                (x, self.static_channels.expand(x.shape[0], -1, -1, -1)), dim=1
            )
        return x

    def preprocess_input(self, input, dates, regrid=True, normalize=True, multiplier=1):
        input = input.to(self.device)

        if normalize:
            input = self.normalizer_in.normalize(input, multiplier=multiplier)

        if self.cosine_zenith_angle:
            input = self.add_cosine_zenith_angle(input, dates)

        if self.static_channels is not None:
            input = self.add_static_channels(input)

        if self.positional_encoding is not None:
            input = self.positional_encoding(input)

        if self.healpix_regridder is not None and regrid:
            input = self.to_healpix(input)

        return input

    def preprocess_target(self, target, regrid=True, multiplier=1):
        target = target.to(self.device)

        if self.normalizer_out is not None:
            target = self.normalizer_out.normalize(target, multiplier=multiplier)

        if self.healpix_regridder is not None and regrid:
            target = self.to_healpix(target)

        return target

    def preprocess(self, sample):
        """Takes in sample and return prepared input"""
        input = sample["x"].to(self.device)
        target = sample["y"].to(self.device)
        input_dates = sample["x_dates"]

        input = self.preprocess_input(input, dates=input_dates)
        target = self.preprocess_target(target)

        return input, target

    def postprocess(self, target, **kwargs):
        """Takes in output and unnormalizes it"""
        if self.equiangular_regridder is not None:
            target = self.to_equiangular(target)

        if self.normalizer_out:
            target = self.normalizer_out.unnormalize(target)

        return target


class SinterpolantDifferenceProcessor(BaseProcessor):
    def __init__(
        self,
        normalizer_stats_shape=(1, 75, 1, 1),
        normalizer_eps=1e-6,
        static_channels_shape=None,
        cosine_zenith_angle=False,
        input_grid_shape=(181, 360),
    ):
        super().__init__(
            normalizer_stats_shape,
            normalizer_eps,
            static_channels_shape,
            cosine_zenith_angle,
            input_grid_shape,
        )

        self.downsample = partial(bilinear_interpolate_torch, shape=input_grid_shape)

    def preprocess_input(self, input, normalize=True):
        if normalize:
            input = self.normalizer_in.normalize(input)

        input = self.downsample(input)

        return input

    def postprocess(self, target, input):
        if self.normalizer_out is not None:
            target = self.normalizer_out.unnormalize(target)

        return target + input


class CombineDifferencesAutoencoder(BaseProcessor):
    def __init__(
        self,
        normalization="gaussian",
        normalizer_stats_shape=None,
        normalizer_eps=1e-6,
        normalizer_range=[-1, 1],
        positional_encoding=None,
        static_channels_shape=None,
        cosine_zenith_angle=False,
        input_grid_shape=(721, 1440),
        regrid_to_healpix: bool = False,
        healpix_level: int = 8,
        device=None,
        **kwargs,
    ):
        super().__init__(
            normalization,
            normalizer_stats_shape,
            normalizer_eps,
            normalizer_range,
            positional_encoding,
            static_channels_shape,
            cosine_zenith_angle,
            input_grid_shape,
            regrid_to_healpix,
            healpix_level,
            device,
        )

        self.intep = EquiangularInterpolator().to(self.device)

    def preprocess_target(self, target, input, regrid=True):
        target = target - input
        target = target.to(self.device)

        if self.normalizer_out is not None:
            target = self.normalizer_out.normalize(target)

        if self.healpix_regridder is not None and regrid:
            target = self.to_healpix(target)

        return target

    def preprocess(self, sample):
        """Takes in sample and return prepared input"""
        input = sample["x"].to(self.device)
        target = sample["y"].to(self.device)
        input_dates = sample["x_dates"]

        target = self.preprocess_target(target, input, regrid=False)
        with torch.no_grad():
            target_low_res = self.intep(target, (181, 360))

        input = self.preprocess_input(input, dates=input_dates)

        return input, target_low_res, target

    def postprocess(self, target, input, **kwargs):
        """Takes in output and unnormalizes it"""
        if self.equiangular_regridder is not None:
            target = self.to_equiangular(target)

        if self.normalizer_out:
            target = self.normalizer_out.unnormalize(target)

        target = target + input

        return target


class SInterpolantDownsampleProcessor(BaseProcessor):
    def __init__(
        self,
        normalization="gaussian",
        normalizer_stats_shape=None,
        normalizer_eps=1e-6,
        normalizer_range=[-1, 1],
        positional_encoding=None,
        static_channels_shape=None,
        cosine_zenith_angle=False,
        input_grid_shape=(721, 1440),
        downsample_grid_shape=(181, 360),
        regrid_to_healpix: bool = False,
        healpix_level: int = 8,
        device=None,
        **kwargs,
    ):
        super().__init__(
            normalization,
            normalizer_stats_shape,
            normalizer_eps,
            normalizer_range,
            positional_encoding,
            static_channels_shape,
            cosine_zenith_angle,
            input_grid_shape,
            regrid_to_healpix,
            healpix_level,
            device,
        )

        self.intep = EquiangularInterpolator().to(self.device)
        self.downsample_grid_shape = downsample_grid_shape

    def preprocess_input(self, x, dates, normalize=True):
        high_res = x
        low_res = x.clone()

        if normalize:
            high_res = self.normalizer_in.normalize(high_res)
            low_res = self.normalizer_in.normalize(low_res)

        low_res = self.intep(low_res, self.downsample_grid_shape)

        if self.cosine_zenith_angle:
            high_res = self.add_cosine_zenith_angle(high_res, dates)

        if self.static_channels is not None:
            high_res = self.add_static_channels(high_res)

        return high_res, low_res

    def preprocess_target(self, target):
        target = self.normalizer_out.normalize(target)
        target = self.intep(target, self.downsample_grid_shape)

        return target

    def preprocess(self, sample):
        """Takes in sample and return prepared input"""
        x = sample["x"].to(self.device)
        target = sample["y"].to(self.device)

        high_res, low_res = self.preprocess_input(x, sample["x_dates"])
        target = self.preprocess_target(target)

        return high_res, low_res, target

    def postprocess(self, target):
        target = self.normalizer_out.unnormalize(target)

        return target


# Below is needed to define the class target for instantiating from the checkpoint
class SinterpolantDifferenceProcessorHistory(SInterpolantDownsampleProcessor):
    pass
