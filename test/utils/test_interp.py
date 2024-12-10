import numpy as np
import pytest
import torch

from earth2studio.utils.interp import LatLonInterpolation


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("input_type", ["zeros", "random", "gradient"])
def test_interpolation(device, input_type):
    (lat_in, lon_in) = np.meshgrid(
        np.arange(35.0, 38.0, 0.25), np.arange(5.0, 8.0, 0.25), indexing="ij"
    )
    (lat_out, lon_out) = np.meshgrid(
        np.arange(36.0, 37.0, 0.1), np.arange(6.0, 7.0, 0.1), indexing="ij"
    )

    interp = LatLonInterpolation(lat_in, lon_in, lat_out, lon_out)
    interp.to(device=device)
    if input_type == "zeros":
        x = torch.zeros(lat_in.shape, device=device)
    elif input_type == "random":
        x = torch.rand(*lat_in.shape, device=device)
    elif input_type == "gradient":
        x = (
            torch.linspace(0, 1, lat_in.shape[1], device=device)
            .unsqueeze(0)
            .repeat(lat_in.shape[0], 1)
        )

    y = interp(x)

    if input_type == "zeros":
        assert (y == 0).all()
    elif input_type == "random":
        assert ((y >= 0) & (y <= 1)).all()
    elif input_type == "gradient":
        assert (y[:, 1:] > y[:, :-1]).all()
