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


def test_interpolation_analytical(device):
    lat_in = np.array([[0.0, 0.0], [1.0, 1.0]])
    lon_in = np.array([[0.0, 1.0], [0.0, 1.0]])

    (lat_out, lon_out) = np.mgrid[:1.01:0.25, :1.01:0.25]

    interp = LatLonInterpolation(lat_in, lon_in, lat_out, lon_out)
    interp.to(device=device)

    x = torch.tensor([[0.0, 1.0], [1.0, 2.0]], device=device)
    y = interp(x)

    y_correct = torch.tensor(
        [
            [0.00, 0.25, 0.50, 0.75, 1.00],
            [0.25, 0.50, 0.75, 1.00, 1.25],
            [0.50, 0.75, 1.00, 1.25, 1.50],
            [0.75, 1.00, 1.25, 1.50, 1.75],
            [1.00, 1.25, 1.50, 1.75, 2.00],
        ],
        device=device,
    )

    epsilon = 1e-6  # allow for some FP roundoff
    assert (abs(y - y_correct) < epsilon).all()
