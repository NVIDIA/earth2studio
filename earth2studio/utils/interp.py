import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.interpolate import LinearNDInterpolator
from torch import Tensor, nn


def latlon_interpolation_regular(
    values: torch.Tensor,
    lat0: torch.Tensor,
    lon0: torch.Tensor,
    lat1: torch.Tensor,
    lon1: torch.Tensor,
) -> torch.Tensor:
    """Specialized form of bilinear interpolation intended for optimal use on GPU.

    In particular, the mapped values must be defined on a regular rectangular grid,
    (lat0, lon0). Both lat0 and lon0 are vectors with equal spacing.

    lat1, lon1 are assumed to be 2-dimensional meshgrids with possibly unequal spacing.

    Parameters
    ----------
    values : torch.Tensor [..., H_in, W_in]
        Input values defined over (lat0, lon0) that will be interpolated onto
        (lat1, lon1) grid.
    lat0 : torch.Tensor [W_in, ]
        Vector of input latitude coordinates, assumed to be increasing with
        equal spacing.
    lon0 : torch.Tensor [W_in, ]
        Vector of input longitude coordinates, assumed to be increasing with
        equal spacing.
    lat1 : torch.Tensor [H_out, W_out]
        Tensor of output latitude coordinates
    lon1 : torch.Tensor [H_out, W_out]
        Tensor of output longitude coordinates

    Returns
    -------
    result : torch.Tensor [..., H_out, W_out]
        Tensor of interpolated values onto lat1, lon1 grid.
    """

    # Get input grid shape and flatten
    latshape, lonshape = lat1.shape
    lat1 = lat1.flatten()
    lon1 = lon1.flatten()

    # Get indices of nearest points
    latinds = torch.searchsorted(lat0, lat1) - 1
    loninds = torch.searchsorted(lon0, lon1) - 1

    # Get original grid spacing
    dlat = lat0[1] - lat0[0]
    dlon = lon0[1] - lon0[0]

    # Get unit distances
    normed_lat_distance = (lat1 - lat0[latinds]) / dlat
    normed_lon_distance = (lon1 - lon0[loninds]) / dlon

    # Apply bilinear mapping
    result = (
        values[..., latinds, loninds]
        * (1 - normed_lat_distance)
        * (1 - normed_lon_distance)
    )
    result += (
        values[..., latinds, loninds + 1]
        * (1 - normed_lat_distance)
        * (normed_lon_distance)
    )
    result += (
        values[..., latinds + 1, loninds]
        * (normed_lat_distance)
        * (1 - normed_lon_distance)
    )
    result += (
        values[..., latinds + 1, loninds + 1]
        * (normed_lat_distance)
        * (normed_lon_distance)
    )
    return result.reshape(*values.shape[:-2], latshape, lonshape)


class LatLonInterpolation(nn.Module):
    """Bilinear interpolation between arbitrary grids.

    The mapped values can be on arbitrary grid, but the output grid should be
    contained within the input grid.

    Initializing the interpolation object can be somewhat slow, but interpolation is
    fast and runs on the GPU once initialized. Therefore, prefer to reuse the
    interpolation object when possible.

    Parameters
    ----------
    lat0 : torch.Tensor [H_in, W_in]
        Tensor of input latitude coordinates
    lon0 : torch.Tensor [H_in, W_in]
        Tensor of input longitude coordinates
    lat1 : torch.Tensor [W_out, H_out]
        Tensor of output latitude coordinates
    lon1 : torch.Tensor [W_out, H_out]
        Tensor of output longitude coordinates
    """

    def __init__(
        self,
        lat_in: Tensor | ArrayLike,
        lon_in: Tensor | ArrayLike,
        lat_out: Tensor | ArrayLike,
        lon_out: Tensor | ArrayLike,
    ):
        super().__init__()

        lat_in = (
            lat_in.cpu().numpy() if isinstance(lat_in, Tensor) else np.array(lat_in)
        )
        lon_in = (
            lon_in.cpu().numpy() if isinstance(lon_in, Tensor) else np.array(lon_in)
        )
        lat_out = (
            lat_out.cpu().numpy() if isinstance(lat_out, Tensor) else np.array(lat_out)
        )
        lon_out = (
            lon_out.cpu().numpy() if isinstance(lon_out, Tensor) else np.array(lon_out)
        )

        (i_in, j_in) = np.mgrid[: lat_in.shape[0], : lat_in.shape[1]]

        in_points = np.stack((lat_in.ravel(), lon_in.ravel()), axis=-1)
        i_interp = LinearNDInterpolator(in_points, i_in.ravel())
        j_interp = LinearNDInterpolator(in_points, j_in.ravel())

        out_points = np.stack((lat_out.ravel(), lon_out.ravel()), axis=-1)
        i_map = i_interp(out_points).reshape(lat_out.shape)
        j_map = j_interp(out_points).reshape(lat_out.shape)

        i_map = torch.Tensor(i_map.clip(min=0, max=lat_in.shape[0] - 2))
        j_map = torch.Tensor(j_map.clip(min=0, max=lat_in.shape[1] - 2))

        self.register_buffer("i_map", i_map)
        self.register_buffer("j_map", j_map)

    @torch.inference_mode()
    def forward(self, values: Tensor) -> Tensor:
        """Perform bilinear interpolation for values.

        Parameters
        ----------
        values : torch.Tensor [..., H_in, W_in]
            Input values defined over (lat0, lon0) that will be interpolated onto
            (lat1, lon1) grid.
        """
        i = self.i_map
        i0 = i.floor().to(dtype=torch.int64)
        i1 = i0 + 1
        j = self.j_map
        j0 = j.floor().to(dtype=torch.int64)
        j1 = j0 + 1

        f00 = values[..., i0, j0]
        f01 = values[..., i0, j1]
        f10 = values[..., i1, j0]
        f11 = values[..., i1, j1]

        dj = j - j0
        f0 = torch.lerp(f00, f01, dj)
        f1 = torch.lerp(f10, f11, dj)
        return torch.lerp(f0, f1, i - i0)
