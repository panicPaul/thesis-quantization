""" Post-processing functions for Gaussian splatting. """

import math

import torch
import torch.nn as nn
from einops import repeat
from jaxtyping import Float, Int
from omegaconf import DictConfig
from torch.optim import AdamW

from thesis.config import DynamicGaussianSplattingSettings
from thesis.constants import TRAIN_CAMS
from thesis.gaussian_splatting.view_dependent_coloring import LearnableColorCorrection


class PostProcessor(nn.Module):
    """ Any function that happens after the rasterization. """

    def __init__(
        self,
        gaussian_splatting_settings: DynamicGaussianSplattingSettings,
        learning_rates: DictConfig,
    ) -> None:
        """
        Args:
            gaussian_splatting_settings (GaussianSplattingSettings): Settings for the Gaussian
                splatting.
        """
        super().__init__()
        self.gaussian_splatting_settings = gaussian_splatting_settings
        self.learning_rates = learning_rates

        # screen space denoiser
        # TODO: port me over from the legacy code

        # learnable color correction
        if gaussian_splatting_settings.learnable_color_correction:
            self.learnable_color_correction = LearnableColorCorrection(len(TRAIN_CAMS))

    def setup_optimizer(self) -> AdamW | None:
        """
        Returns:
            AdamW | None: Optimizer for the post-processor or None if there are no learnable
            parameters.
        """
        batch_size = self.gaussian_splatting_settings.camera_batch_size
        batch_scaling = math.sqrt(batch_size)
        params = []
        if hasattr(self, "learnable_color_correction"):
            params.append({
                "params": self.learnable_color_correction.parameters(),
                "lr": self.learning_rates.color_correction_lr * batch_scaling,
                'weight_decay': 1e-6,
            })
        if len(params) == 0:
            return None
        return AdamW(params)

    def forward(
        self,
        infos: dict,
        render_images: Float[torch.Tensor, "cam H W 3"],
        render_alphas: Float[torch.Tensor, "cam H W 1"],
        background: Float[torch.Tensor, "3"],
        color_correction: Float[torch.Tensor, "cam 3 3"] | None = None,
        camera_indices: Int[torch.Tensor, "cam"] | None = None,
    ) -> tuple[Float[torch.Tensor, "cam H W 3"], Float[torch.Tensor, "cam H W 1"], dict]:
        """
        Post-processing step for the rasterization.

        Args:
            infos (dict): Dictionary to store additional information.
            render_images (torch.Tensor): Rendered images, shape: `(cam, H, W, 3)`.
            inside_mouth_render_images (torch.Tensor): Inside mouth rendered images, shape:
                `(cam, H, W, 3)`.
            background (torch.Tensor): Background color, shape: `(3,)`.
            color_correction (torch.Tensor): Color correction matrix, shape: `(cam, 3, 3)`.
            camera_indices (torch.Tensor): Camera indices. Used to choose the correct camera
                color correction matrix. Shape: `(cam,)`.

        Returns:
            tuple: A tuple containing
                - (*torch.Tensor*): RGB images, shape: `(cam, H, W, 3)`.
                - (*torch.Tensor*): Alphas, shape: `(cam, H, W, 1)`.
                - (*dict*): Infos, a dictionary containing additional information.
        """

        # Apply color correction
        if (color_correction is not None and
                self.gaussian_splatting_settings.camera_color_correction):
            # NOTE: the nersamble color corrections seem off, so its probably better to use
            #       the learnable color correction
            # Reshape color_correction to (batch * height * width, 3, 3)
            batch_size, height, width, _ = render_images.shape
            color_correction = color_correction.expand(batch_size * height * width, -1, -1)

            # Reshape images to (batch * height * width, 3, 1)
            render_images = render_images.view(batch_size, height, width, 3, 1)
            render_images = render_images.permute(0, 1, 2, 4, 3).contiguous()
            render_images = render_images.view(-1, 3, 1)

            # Perform batch matrix multiplication
            corrected_colors = torch.bmm(color_correction, render_images)

            # Reshape the result back to (batch, height, width, 3)
            corrected_colors = corrected_colors.view(batch_size, height, width, 1, 3)
            corrected_colors = corrected_colors.permute(0, 1, 2, 4, 3).contiguous()
            corrected_colors = corrected_colors.view(batch_size, height, width, 3)
            render_images = corrected_colors

        # Apply learnable color correction
        if hasattr(self, "learnable_color_correction") and camera_indices is not None:
            render_images = self.learnable_color_correction.forward(camera_indices, render_images)

        # Apply background
        image_height, image_width = render_images.shape[1:3]
        background = repeat(
            background,
            "f -> cam H W f",
            cam=render_images.shape[0],
            H=image_height,
            W=image_width,
        )
        render_images = render_images*render_alphas + (1-render_alphas) * background
        infos['raw_rendered_images'] = render_images
        infos['raw_rendered_alphas'] = render_alphas

        # Apply screen-space denoising
        if hasattr(self, "screen_space_denoiser"):

            raise NotImplementedError("Screen space denoiser not ported over.")
            render_images = self.screen_space_denoiser.forward(render_images, render_alphas)

        return render_images, render_alphas, infos
