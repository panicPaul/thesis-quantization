""" Learns the camera color correction matrix. """

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int


class LearnableColorCorrection(nn.Module):
    """ Learnable color correction matrix for each camera. """

    def __init__(self, num_cameras: int) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.color_correction = nn.Parameter(torch.eye(3, 3).repeat(num_cameras, 1, 1))

    def forward(
        self,
        camera_indices: Int[torch.Tensor, "cam"],
        image: Float[torch.Tensor, "cam h w 3"],
    ) -> Float[torch.Tensor, "cam h w 3"]:
        """
        Apply the color correction matrix to the image.

        Args:
            camera_indices: Camera indices. Has shape (cam,).
            image: Image tensor. Has shape (cam, height, width, 3).
        """

        correction_matrices = self.color_correction[camera_indices]
        cam, height, width, _ = image.shape
        reshaped_image = rearrange(image, "cam h w c -> cam (h w) c")
        # Apply color correction: (cam, h*w, 3) @ (cam, 3, 3) -> (cam, h*w, 3)
        corrected_image = torch.bmm(reshaped_image, correction_matrices)
        corrected_image = rearrange(
            corrected_image,
            "cam (h w) c -> cam h w c",
            cam=cam,
            h=height,
            w=width,
        )

        return corrected_image
