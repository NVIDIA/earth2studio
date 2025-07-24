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

from collections import OrderedDict

import torch

from earth2studio.utils import handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

from .moments import mean
from .utils import _spatial_dims_to_end

try:
    from physicsnemo.metrics.general.power_spectrum import power_spectrum
except ImportError:
    OptionalDependencyFailure("statistics")
    power_spectrum = None


@check_optional_dependencies()
class log_spectral_distance:
    """
    Statistic for calculating the radially averaged 2D log spectral distance (LSD)
    of one tensor with respect to another over a set of given dimensions. This is given
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

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the computed statistic, corresponding to the given input coordinates

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        removed_dims = list(self._reduction_dimensions)
        spatial_dimensions = (
            list(input_coords)[-2:]
            if self.spatial_dimensions is None
            else self.spatial_dimensions
        )
        removed_dims.extend(spatial_dimensions)

        output_coords = input_coords.copy()
        for dimension in removed_dims:
            handshake_dim(input_coords, dimension)
            output_coords.pop(dimension)

        return output_coords

    def _validate_coords(self, x_coords: CoordSystem, y_coords: CoordSystem) -> None:
        x_coords = x_coords.copy()
        if self.ensemble_dimension is not None:
            if self.ensemble_dimension not in x_coords:
                raise ValueError(
                    f"Ensemble dimension '{self.ensemble_dimension}' set but not present in x_coords."
                )
            if self.ensemble_dimension in y_coords:
                raise ValueError(
                    f"Ensemble dimension '{self.ensemble_dimension}' present in y_coords."
                )
            x_coords.pop("ensemble", None)
        for (x_dim, x_coord), (y_dim, y_coord) in zip(
            x_coords.items(), y_coords.items()
        ):
            if (x_dim != y_dim) or (x_coord != y_coord).any():
                raise ValueError("Coordinates are incompatible.")

    def __call__(
        self,
        x: torch.Tensor,
        x_coords: CoordSystem,
        y: torch.Tensor,
        y_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Apply metric to data `x` and `y`, checking that their coordinates
        are broadcastable. While reducing over `reduction_dims`.

        If batch_update was passed True upon metric initialization then this method
        returns the running sample RMSE over all seen batches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, typically the forecast or prediction tensor.
        x_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `x` tensor.
            `reduction_dimensions` must be in x_coords, as do `ensemble_dimension` and
            `spatial_dimensions` if provided in constructor.
        y : torch.Tensor
            Input tensor #2 intended to be used as validation data.
        y_coords : CoordSystem
            Ordered dict representing coordinate system that describes the `y` tensor.
            `reduction_dimensions` must be in y_coords, do `spatial_dimensions` if
            provided in constructor.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Returns root mean squared error tensor with appropriate reduced coordinates.
        """
        self._validate_coords(x_coords, y_coords)

        if self.spatial_dimensions is not None:
            (x, x_coords) = _spatial_dims_to_end(x, x_coords, self.spatial_dimensions)
            (y, y_coords) = _spatial_dims_to_end(y, y_coords, self.spatial_dimensions)

        if self.ensemble_dimension is not None:
            ensemble_dim = list(x_coords).index(self.ensemble_dimension)
            y = y.unsqueeze(ensemble_dim)

        (k, spectrum_x) = power_spectrum(x)
        spectrum_y = power_spectrum(y)[1]

        if self.wavenumber_cutoff is not None:
            k = k[: self.wavenumber_cutoff]
            spectrum_x = spectrum_x[..., : self.wavenumber_cutoff]
            spectrum_y = spectrum_y[..., : self.wavenumber_cutoff]

        spectrum_coords = OrderedDict(list(x_coords.items())[:-2])
        spectrum_coords["wavenumber"] = k

        (lsd, out_coords) = self.mean(
            torch.log10(spectrum_x / spectrum_y) ** 2, spectrum_coords
        )
        lsd = 10 * lsd.sqrt()  # to dB

        return (lsd, out_coords)
