""" Vector Quantized Variational Autoencoder (VQ-VAE) implementation. """

import lightning as pl
from jaxtyping import Float
from torch import nn
import torch
from torch.optim.adamw import AdamW
from vector_quantize_pytorch import FSQ
from einops import rearrange
from typing import Literal
from omegaconf import DictConfig

# ==================================================================================== #
#                                   Layers                                             #
# ==================================================================================== #


class ResBlock(nn.Module):
    """Residual Block from MAGVIT"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3) -> None:
        """Residual Block from MAGVIT"""
        self.residual_connection = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(
                in_channels, out_channels, kernel_size=1))
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding="reflect",
            padding=kernel_size // 2,
        )
        self.conv_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding="reflect",
            padding=kernel_size // 2,
        )
        self.group_norm_1 = nn.GroupNorm(32, out_channels)  # as per the paper
        self.group_norm_2 = nn.GroupNorm(32, out_channels)

    def forward(
        self, x: Float[torch.Tensor, "batch in_channels time"]
    ) -> Float[torch.Tensor, "batch out_channels time"]:
        """Forward pass of the Residual Block"""
        residual = self.residual_connection(x)
        x = self.group_norm_1(x)
        x = nn.functional.silu(x)
        x = self.conv_1(x)
        x = self.group_norm_2(x)
        x = nn.functional.silu(x)
        x = self.conv_2(x)
        return x + residual


class ResBlockDown(nn.Module):
    """Residual Block from MAGVIT with downsampling"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3) -> None:
        """Residual Block from MAGVIT with downsampling"""
        self.residual_connection = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(
                in_channels, out_channels, kernel_size=1))
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding_mode="reflect",
            padding=kernel_size // 2,
        )
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="reflect")
        self.group_norm_1 = nn.GroupNorm(32, out_channels)  # as per the paper
        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.average_pool = nn.AvgPool1d(2)

    def forward(
        self, x: Float[torch.Tensor, "batch in_channels time"]
    ) -> Float[torch.Tensor, "batch out_channels new_time"]:
        """Forward pass of the Residual Block with downsampling"""
        residual = self.residual_connection(self.average_pool(x))
        x = self.group_norm_1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        # thats what they say in the paper but as far as I can tell they just
        # use ReLU in their code
        x = self.conv_1(x)
        x = self.group_norm_2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.conv_2(x)
        x = self.average_pool(x)
        return x + residual


class UpSampling(nn.Module):
    """UpSampling module from MAGVIT"""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        """UpSampling module from MAGVIT"""
        # nearest neighbor upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding="reflect",
            padding=kernel_size // 2,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch channels time"]
    ) -> Float[torch.Tensor, "batch channels new_time"]:
        """Forward pass of the UpSampling module"""
        x = self.upsample(x)
        return self.conv(x)


# ==================================================================================== #
#                                   VQ-VAE                                             #
# ==================================================================================== #


class VQ_VAE(pl.LightningModule):
    """Vector Quantized Variational Autoencoder from MAGVIT."""

    def __init__(
        self,
        feature_dim: int,
        channel_multiplier: float = 1.0,
        latent_dim: int = 256,
        levels: list[int] = [8, 6, 5],
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        training_steps: int = 100_000,
        loss_fn: Literal["mse", "mae", "flame_vertex"] = "mse",
    ) -> None:
        """
        Args:
            feature_dim: The number of features in the input data.
            channel_multiplier: The multiplier for the number of channels in the model.
            latent_dim: The dimension of the latent space.
            levels: The number of levels for each codebook. Defaults to a (8, 6, 5),
                i.e. 240 unique codes.
            lr: The learning rate for the optimizer.
            weight_decay: The weight decay for the optimizer.
            training_steps: The number of training steps.
            flame_vertex_loss: Whether to use compute the loss via the FLAME vertices
                instead of MSE.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_steps = training_steps

        input_layer = nn.Conv1d(
            feature_dim, int(64 * channel_multiplier), kernel_size=3, padding=1)
        self.encoder = nn.Sequential(
            input_layer,
            # batch x 64c x time
            ResBlock(int(64 * channel_multiplier), int(64 * channel_multiplier)),
            # batch x 64c x time
            ResBlock(int(64 * channel_multiplier), int(64 * channel_multiplier)),
            # batch x 64c x time
            nn.AvgPool1d(kernel_size=2),
            # batch x 64c x time / 2
            ResBlock(int(64 * channel_multiplier), int(128 * channel_multiplier)),
            # batch x 128c x time / 2
            ResBlock(int(128 * channel_multiplier), int(128 * channel_multiplier)),
            # batch x 128c x time / 2
            nn.AvgPool1d(kernel_size=2),
            # batch x 128c x time / 4
            ResBlock(int(128 * channel_multiplier), int(128 * channel_multiplier)),
            # batch x 128c x time / 4
            ResBlock(int(128 * channel_multiplier), int(128 * channel_multiplier)),
            # batch x 128c x time / 4
            nn.GroupNorm(32, int(128 * channel_multiplier)),
            nn.SiLU(),
            nn.Conv1d(int(128 * channel_multiplier), latent_dim, 1),
        )
        self.quantization = FSQ(levels=levels, dim=latent_dim)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, int(128 * channel_multiplier), 1),
            # batch x 128c x time / 4
            ResBlock(int(128 * channel_multiplier), int(128 * channel_multiplier)),
            # batch x 128c x time / 4
            ResBlock(int(128 * channel_multiplier), int(128 * channel_multiplier)),
            # batch x 128c x time / 4
            UpSampling(int(128 * channel_multiplier)),
            # batch x 128c x time / 2
            ResBlock(int(128 * channel_multiplier), int(128 * channel_multiplier)),
            # batch x 128c x time / 2
            ResBlock(int(128 * channel_multiplier), int(128 * channel_multiplier)),
            # batch x 128c x time / 2
            UpSampling(int(128 * channel_multiplier)),
            # batch x 128c x time
            ResBlock(int(128 * channel_multiplier), int(64 * channel_multiplier)),
            # batch x 64c x time
            ResBlock(int(64 * channel_multiplier), int(64 * channel_multiplier)),
            # batch x 64c x time
            nn.GroupNorm(32, int(64 * channel_multiplier)),
            nn.SiLU(),
            nn.Conv1d(int(64 * channel_multiplier), feature_dim, 3, padding=1),
        )

        match loss_fn:
            case "mse":
                self.loss_fn = nn.functional.mse_loss
            case "mae":
                self.loss_fn = nn.functional.l1_loss
            case "flame_vertex":
                raise NotImplementedError("FLAME vertex loss not implemented yet")

    def configure_optimizers(self) -> AdamW:
        """Configure the optimizer for the VQ-VAE"""
        # TODO: batch size scaling
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    # ================================================================================ #

    def forward(
        self, x: Float[torch.Tensor, "batch feature_dim time"]
    ) -> Float[torch.Tensor, "batch feature_dim time"]:
        """Forward pass of the VQ-VAE"""
        x = self.encoder(x)
        x = rearrange(x, "batch latent_dim time -> batch time latent_dim").unsqueeze(-1)
        x, _ = self.quantization(x)
        x = rearrange(x.squeeze(-1), "batch time latent_dim -> batch latent_dim time")
        x = self.decoder(x)
        return x

    # ================================================================================ #

    def training_step(self, x: Float[torch.Tensor, "batch feature_dim time"]) -> dict:
        """Training step for the VQ-VAE"""
        # TODO: doesn't work for flame yet but we can adjust it later
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return {"loss": loss}

    def val_step(self, x: Float[torch.Tensor, "batch feature_dim time"]) -> dict:
        """Validation step for the VQ-VAE"""
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("val_loss", loss)
        return {"loss": loss}

    # ================================================================================ #


# ==================================================================================== #
#                               Train loop                                             #
# ==================================================================================== #


def train_vq_vae(config: DictConfig) -> None:
    """Train the VQ-VAE model."""
    pass
