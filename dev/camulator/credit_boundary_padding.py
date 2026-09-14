import torch
import torch.nn.functional as F


class TensorPadding:
    def __init__(self, mode="earth", pad_lat=(40, 40), pad_lon=(40, 40), **kwargs):
        """
        Initialize the TensorPadding class with the specified mode and padding sizes.

        Args:
            mode (str): The padding mode, either 'mirror' or 'earth'.
            pad_lat (list[int]): Padding sizes for the North-South (latitude) dimension [top, bottom].
            pad_lon (list[int]): Padding sizes for the West-East (longitude) dimension [left, right].
        """

        self.mode = mode
        self.pad_NS = pad_lat
        self.pad_WE = pad_lon

    def pad(self, x):
        """
        Apply padding to the tensor based on the specified mode.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, var, time, lat, lon).

        Returns:
            torch.Tensor: The padded tensor.
        """
        if self.mode == "mirror":
            return self._mirror_padding(x)
        elif self.mode == "earth":
            return self._earth_padding(x)

    def unpad(self, x):
        """
        Remove padding from the tensor based on the specified mode.

        Args:
            x (torch.Tensor): Padded tensor of shape (batch, var, time, lat, lon).

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        if self.mode == "mirror":
            return self._mirror_unpad(x)
        elif self.mode == "earth":
            return self._earth_unpad(x)

    def _earth_padding(self, x):
        """
        Apply earth padding to the tensor (poles and circular padding).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        """
        if any(p > 0 for p in self.pad_NS):
            # 180-degree shift using half the longitude size
            shift_size = int(x.shape[-1] // 2)
            xroll = torch.roll(x, shifts=shift_size, dims=-1)
            # pad poles
            xroll_flip_top = torch.flip(xroll[..., : self.pad_NS[0], :], (-2,))
            xroll_flip_bot = torch.flip(xroll[..., -self.pad_NS[1] :, :], (-2,))
            x = torch.cat([xroll_flip_top, x, xroll_flip_bot], dim=-2)

        if any(p > 0 for p in self.pad_WE):
            x = F.pad(x, (self.pad_WE[0], self.pad_WE[1], 0, 0, 0, 0), mode="circular")

        return x

    def _earth_unpad(self, x):
        """
        Remove earth padding to restore the original tensor size.

        Args:
            x (torch.Tensor): Padded tensor.

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        # unpad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            start_NS = self.pad_NS[0]
            end_NS = -self.pad_NS[1] if self.pad_NS[1] > 0 else None
            x = x[..., start_NS:end_NS, :]

        # unpad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            start_WE = self.pad_WE[0]
            end_WE = -self.pad_WE[1] if self.pad_WE[1] > 0 else None
            x = x[..., :, start_WE:end_WE]

        return x

    def _mirror_padding(self, x):
        """
        Apply mirror padding to the tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The padded tensor.
        """
        # pad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            pad_lon_left, pad_lon_right = self.pad_WE
            x = F.pad(x, pad=(self.pad_WE[0], self.pad_WE[1], 0, 0, 0, 0), mode="circular")

        # pad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            x = F.pad(x, pad=(0, 0, self.pad_NS[0], self.pad_NS[1], 0, 0), mode="reflect")

        return x

    def _mirror_unpad(self, x):
        """
        Remove mirror padding to restore the original tensor size.

        Args:
            x (torch.Tensor): Padded tensor.

        Returns:
            torch.Tensor: The unpadded tensor.
        """
        # unpad along latitude (north-south)
        if any(p > 0 for p in self.pad_NS):
            x = x[..., self.pad_NS[0] : -self.pad_NS[1], :]

        # unpad along longitude (west-east)
        if any(p > 0 for p in self.pad_WE):
            x = x[..., :, self.pad_WE[0] : -self.pad_WE[1]]

        return x
