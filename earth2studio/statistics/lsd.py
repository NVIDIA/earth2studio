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

from __future__ import annotations

import torch
import xarray as xr

from earth2studio.statistics.moments import mean
from earth2studio.statistics.utils import (
    _align_xy,
    _array_module,
    _as_host_or_cupy,
    _output_signature,
    _spatial_dims_to_end,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    from physicsnemo.metrics.general.power_spectrum import power_spectrum
except ImportError:
    OptionalDependencyFailure("statistics")
    power_spectrum = None


@check_optional_dependencies()
class log_spectral_distance:
    """
    Statistic for calculating the radially averaged 2D log spectral distance (LSD)
    of one array with respect to another over a set of given dimensions. This is given
    in decibel (dB) as 10 * sqrt( mean( log10( psd(x) / psd(y) )**2 ) ).

    Parameters
    ----------
    reduction_dimensions: List[str] = []
        A list of names corresponding to additional dimensions (besides the wavenumber)
        to perform the statistical reduction over.
    spatial_dimensions: Tuple[str, str] | None = None
        Indicates the spatial dimensions. If None, it is assumed that these are the last
        two dimensions.
    ensemble_dimension: str | None = None
        Indicates the ensemble dimension, if not None. The LSD is computed between each
        ensemble member in the prediction and the corresponding observation.
    wavenumber_cutoff: int | None = None
        If a positive integer, use only the first `wavenumber_cutoff` modes to compute LSD.
        If a negative integer, use all except the last `-wavenumber_cutoff` modes.
        If None (default), use all modes.
    batch_update: bool = False
        Whether to apply batch updates to the LSD with each invocation of __call__.
        This is particularly useful when data is recieved in a stream of batches. Each
        invocation of __call__ will return the running LSD.
    """

    def __init__(
        self,
        reduction_dimensions: list[str] = [],
        spatial_dimensions: tuple[str, str] | None = None,
        ensemble_dimension: str | None = None,
        wavenumber_cutoff: int | None = None,
        batch_update: bool = False,
    ):
        self.mean = mean(
            reduction_dimensions + ["wavenumber"], batch_update=batch_update
        )
        self.spatial_dimensions = spatial_dimensions
        self.ensemble_dimension = ensemble_dimension
        self.wavenumber_cutoff = wavenumber_cutoff
        self._reduction_dimensions = reduction_dimensions

    def __str__(self) -> str:
        return "_".join(self._reduction_dimensions + ["log_spectral_distance"])

    @property
    def reduction_dimensions(self) -> list[str]:
        return self._reduction_dimensions

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the computed statistic, corresponding to the given input coordinates

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordinateSystem
            Coordinate system of the statistic output
        """
        removed_dims = list(self._reduction_dimensions)
        spatial_dimensions = (
            list(input_coords.dims)[-2:]
            if self.spatial_dimensions is None
            else self.spatial_dimensions
        )
        removed_dims.extend(spatial_dimensions)
        return _output_signature(input_coords, removed_dims)

    def _validate_coords(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        return _align_xy(x, y, self.ensemble_dimension)

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample RMSE over all seen batches.

        Parameters
        ----------
        x : xr.DataArray
            Input array, typically the forecast or prediction array.
            `reduction_dimensions` must be in x, as must `ensemble_dimension` and
            `spatial_dimensions` if provided in constructor.
        y : xr.DataArray
            Input array #2 intended to be used as validation data. Must not contain
            the ensemble dimension.

        Returns
        -------
        xr.DataArray
            Returns log spectral distance array with appropriate reduced coordinates.
        """
        y = self._validate_coords(x, y)

        if self.spatial_dimensions is not None:
            x = _spatial_dims_to_end(x, self.spatial_dimensions)
            y = _spatial_dims_to_end(y, self.spatial_dimensions)

        x_t, _ = _as_host_or_cupy(x).e2s.to_torch()
        y_t, _ = _as_host_or_cupy(y).e2s.to_torch()

        if self.ensemble_dimension is not None:
            ensemble_dim = list(x.dims).index(self.ensemble_dimension)
            y_t = y_t.unsqueeze(ensemble_dim)

        # Power spectrum computed with torch (physicsnemo)
        k, spectrum_x = power_spectrum(x_t)
        spectrum_y = power_spectrum(y_t)[1]

        if self.wavenumber_cutoff is not None:
            k = k[: self.wavenumber_cutoff]
            spectrum_x = spectrum_x[..., : self.wavenumber_cutoff]
            spectrum_y = spectrum_y[..., : self.wavenumber_cutoff]

        spectrum_dims = list(x.dims)[:-2]
        spectrum = from_torch(
            torch.log10(spectrum_x / spectrum_y) ** 2,
            _output_signature(
                x,
                list(x.dims)[-2:],
                add={"wavenumber": k.detach().cpu().numpy()},
            ),
        )
        assert list(spectrum.dims) == spectrum_dims + ["wavenumber"]

        lsd = self.mean(spectrum)
        xp = _array_module(lsd)
        return lsd.copy(data=10 * xp.sqrt(lsd.data))  # to dB
