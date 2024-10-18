""" Gaussian Splatting on a single time step. Mostly for debugging. """

import math

import lightning as pl
import nerfview
import numpy as np
import torch
from gsplat import DefaultStrategy, MCMCStrategy, rasterization, rasterization_2dgs
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Adam

from thesis.config import GaussianSplattingSettings
from thesis.data_management.data_classes import SingleFrameData
from thesis.gaussian_splatting.initialize_splats import random_initialization
from thesis.gaussian_splatting.view_dependent_color import ViewDependentColorMLP
from thesis.utils import apply_se3


class GaussianSplattingSingleFrame(pl.LightningModule):
    """Gaussian splatting on a single time step."""

    def __init__(
        self,
        gaussian_splatting_settings: GaussianSplattingSettings | DictConfig,
        initialization_settings: OmegaConf,
        densification_settings: OmegaConf,
        learning_rates: OmegaConf,
        training_settings: OmegaConf,
        rasterization_settings: OmegaConf,
    ) -> None:
        """Initializes the Gaussian splatting model."""
        super().__init__()

        self.save_hyperparameters()
        self.learning_rates = learning_rates
        self.training_settings = training_settings
        self.gaussian_splatting_settings = gaussian_splatting_settings
        self.rasterization_settings = rasterization_settings
        self.automatic_optimization = False
        if isinstance(gaussian_splatting_settings, DictConfig):
            gaussian_splatting_settings = GaussianSplattingSettings(
                **gaussian_splatting_settings
            )
        self.max_sh_degree = gaussian_splatting_settings.sh_degree

        # Initialize splats
        match gaussian_splatting_settings.initialization_mode:
            case "random":
                self.splats = nn.ParameterDict(
                    random_initialization(**initialization_settings)
                )

            case _:
                raise ValueError(
                    "Unknown initialization mode: "
                    f"{gaussian_splatting_settings.initialization_mode}"
                )

        # Initialize the densification strategy
        match gaussian_splatting_settings.densification_mode:
            case "default":
                self.strategy = DefaultStrategy(**densification_settings)

            case "monte_carlo_markov_chain":
                self.strategy = MCMCStrategy(**densification_settings)

            case _:
                raise ValueError(
                    "Unknown densification mode: "
                    f"{gaussian_splatting_settings.densification_mode}"
                )

        # View-dependent color module
        if gaussian_splatting_settings.use_view_dependent_color_mlp:
            self.view_dependent_color_mlp = ViewDependentColorMLP(
                feature_dim=gaussian_splatting_settings.feature_dim,
                sh_degree=gaussian_splatting_settings.sh_degree,
            )

        # Get the rasterization function
        match gaussian_splatting_settings.rasterization_mode:
            case "default" | "3dgs":
                self.rasterize = rasterization

            case "2dgs":
                self.rasterize = rasterization_2dgs

            case _:
                raise ValueError(
                    "Unknown rasterization mode: "
                    f"{gaussian_splatting_settings.rasterization_mode}"
                )

    def configure_optimizers(self) -> tuple[set[Adam], dict[Adam]]:
        """
        Configures the optimizer.

        Returns:
            A tuple containing the splat optimizers and other optimizers.
        """
        batch_size = self.training_settings.batch_size
        scaling = math.sqrt(batch_size)
        splat_optimizers = {
            Adam(
                [
                    {
                        "params": self.splats[name],
                        "lr": self.learning_rates[f"{name}_lr"] * math.sqrt(batch_size),
                        "name": name,
                    }
                ],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name in ["means", "scales", "quats", "opacities"]
        }
        other_optimizers = {}
        if hasattr(self, "view_dependent_color_mlp"):
            other_optimizers["view_dependent_color_mlp"] = Adam(
                self.view_dependent_color_mlp.parameters(),
                lr=self.learning_rates.color_mlp_lr * scaling,
            )
        return splat_optimizers, other_optimizers

    # ================================================================================ #
    #                                 Train / Val Steps                                #
    # ================================================================================ #

    def get_cur_sh_degree(self, step: int) -> int:
        """Returns the current spherical harmonic degree."""
        return min(
            step // self.gaussian_splatting_settings.sh_increase_interval,
            self.max_sh_degree,
        )

    def forward(
        self, batch: SingleFrameData, cur_sh_degree: int | None
    ) -> Float[torch.Tensor, "cam H W 3"]:
        """Forward pass."""
        # Set up
        if cur_sh_degree is None:
            cur_sh_degree = self.max_sh_degree
        cur_sh_degree = self.get_cur_sh_degree(self.global_step)
        means = self.splats["means"]
        quats = self.splats["quats"]
        features = self.splats["features"]
        means, quats = apply_se3(batch.se3_transform, means, quats)
        if hasattr(self, "view_dependent_color_mlp"):
            colors = self.view_dependent_color_mlp.forward(
                features, means, batch.world_2_cam, cur_sh_degree
            )
        else:
            colors = self.splats["colors"]
        # Rasterization
        images = self.rasterize(
            means=means,
            quats=quats,
            scales=self.splats["scales"],
            opacities=self.splats["opacities"],
            colors=colors,
            render_mode="RGB",
            viewmats=None,  # TODO: fix me
            Ks=None,
            width=None,
            height=None,
        )
        return images

    def train_step(self, batch: SingleFrameData, batch_idx: int) -> torch.Tensor:
        """Training step."""

        splat_optimizers, other_optimizers = self.optimizers()
        for optimizer in splat_optimizers.values():
            optimizer.zero_grad()
        for optimizer in other_optimizers.values():
            optimizer.zero_grad()

    def val_step(self, batch: SingleFrameData, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Forward pass
        pass

    def render(
        self,
        camera_state: nerfview.CameraState,
        img_wh: tuple[int, int],
    ) -> Float[np.ndarray, "H W 3"]:
        """Renders the RGB image."""
        pass

    def render_depth(
        self,
        camera_state: nerfview.CameraState,
        img_wh: tuple[int, int],
    ) -> Float[np.ndarray, "H W"]:
        """Renders the depth image."""
        pass


# ==================================================================================== #
#                                 Train / Val Loops                                    #
# ==================================================================================== #


def train(config_path: str) -> None:
    pass
    # TODO: sanity check
    # self.strategy.check_sanity(self.splats, self.optimizers)
