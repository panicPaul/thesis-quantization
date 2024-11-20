""" Screen space denoising CNN. """

import torch
import torch.nn as nn
from jaxtyping import Float

# TODO: instance normalization maybe?


class ScreenSpaceDenoising(nn.Module):
    """ Screen space CNN that denoises LPIPS overfitting artifacts. """

    def __init__(self) -> None:
        super().__init__()
        hidden_channels = 32
        self.conv1 = nn.Conv2d(4, hidden_channels, 3, padding=1)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)

        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)

        self.conv6 = nn.Conv2d(hidden_channels, 3, 3, padding=1)

    def inverse_sigmoid(
        self,
        x: Float[torch.Tensor, "cam height width 3"],
    ) -> Float[torch.Tensor, "cam height width 3"]:
        x = torch.clamp(x, 1e-6, 1 - 1e-6)  # small clipping to avoid log(0)
        return torch.log(x / (1-x))

    def forward(
        self,
        image: Float[torch.Tensor, "cam height width 3"],
        alphas: Float[torch.Tensor, "cam height width 1"],
    ) -> Float[torch.Tensor, "cam height width 3"]:
        """
        Args:
            image (torch.Tensor): The input image, shape (batch, height, width, 3).
            alphas (torch.Tensor): The alpha values, shape (batch, height, width, 1).

        Returns:
            torch.Tensor: The denoised image, shape (batch, height, width, 3).
        """
        x = torch.cat([image, alphas], dim=-1)
        x = torch.permute(x, (0, 3, 1, 2))  # (cam, 4, height, width)
        image = self.inverse_sigmoid(image)
        image = torch.permute(image, (0, 3, 1, 2))

        x = nn.functional.silu(self.conv1(x))

        residual = x
        x = nn.functional.silu(self.conv2(x))
        x = nn.functional.silu(self.conv3(x)) + residual

        residual = x
        x = nn.functional.silu(self.conv4(x))
        x = nn.functional.silu(self.conv5(x)) + residual

        x = nn.functional.sigmoid(self.conv6(x) + image)
        x = torch.permute(x, (0, 2, 3, 1))
        return x
