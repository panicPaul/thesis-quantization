""" Vector Quantized Variational Autoencoder (VQ-VAE) implementation. """

import argparse
from typing import Literal

import lightning as pl
import torch
from einops import rearrange
from jaxtyping import Float
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from vector_quantize_pytorch import LFQ

from thesis.constants import TEST_SEQUENCES, TRAIN_SEQUENCES
from thesis.data_management import QuantizationDataset

# ==================================================================================== #
#                                   Layers                                             #
# ==================================================================================== #


class ResBlock(nn.Module):
    """Residual Block from MAGVIT"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3) -> None:
        """Residual Block from MAGVIT"""
        super().__init__()
        self.residual_connection = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(
                in_channels, out_channels, kernel_size=1))
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding_mode="reflect",
            padding=kernel_size // 2,
        )
        self.conv_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding_mode="reflect",
            padding=kernel_size // 2,
        )
        self.group_norm_1 = nn.GroupNorm(32, in_channels)  # as per the paper
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
        super().__init__()
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
        super().__init__()
        # nearest neighbor upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding_mode="reflect",
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
        latent_dim: int = 64,
        levels: list[int] = [8, 6, 5],
        lr: float = 3e-4,
        weight_decay: float = 0.01,
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
        self.levels = torch.tensor(levels)
        # codebook_size = np.prod(levels)
        # histogram = torch.zeros(codebook_size, dtype=torch.int64)

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
        self.quantization_projection_in = nn.Linear(latent_dim, 8)
        self.quantization = LFQ(
            codebook_size=256,
            scale_trick=True,
            entropy_loss_weight=0.1,
            commitment_loss_weight=0.25,
        )
        self.quantization_projection_out = nn.Linear(8, latent_dim)
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

    def forward(self, x: Float[torch.Tensor, "batch time feature_dim"]):
        #  -> tuple[Float[torch.Tensor, "batch time feature_dim"], Int[torch.Tensor, "batch time"],
        #            Any]:
        """Forward pass of the VQ-VAE"""
        x = rearrange(x, "batch time feature_dim -> batch feature_dim time")
        x = self.encoder(x)
        x = rearrange(x, "batch latent_dim time -> batch time latent_dim")
        x = self.quantization_projection_in(x)
        x, indices, aux_loss = self.quantization.forward(x)
        x = self.quantization_projection_out(x)
        # TODO: get all fine grained losses
        x = rearrange(x, "batch time latent_dim -> batch latent_dim time")
        x = self.decoder(x)
        x = rearrange(x, "batch feature_dim time -> batch time feature_dim")
        return x, indices, aux_loss

    # ================================================================================ #

    def training_step(self, batch) -> Float[torch.Tensor, ""]:
        """Training step for the VQ-VAE"""
        # TODO: doesn't work for flame yet but we can adjust it later
        _, _, x = batch
        x_hat, indices, aux_loss = self.forward(x)
        reconstruction_loss = self.loss_fn(x_hat, x)
        loss = reconstruction_loss + 50*aux_loss
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/reconstruction_loss", reconstruction_loss, prog_bar=False)
        self.log("train/aux_loss", aux_loss, prog_bar=False)
        # update histogram

        return loss

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    def validation_step(self, batch) -> Float[torch.Tensor, ""]:
        """Validation step for the VQ-VAE"""
        _, _, x = batch
        x_hat, indices, aux_loss = self.forward(x)
        reconstruction_loss = self.loss_fn(x_hat, x)
        loss = reconstruction_loss + aux_loss
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/reconstruction_loss", reconstruction_loss, prog_bar=False)
        self.log("val/aux_loss", aux_loss, prog_bar=False)
        return loss

    # ================================================================================ #


# ==================================================================================== #
#                               Train loop                                             #
# ==================================================================================== #


def train_vq_vae(config_path: str) -> None:
    """Train the VQ-VAE model."""
    config = OmegaConf.load(config_path)

    # setup model
    model = VQ_VAE(
        lr=config.training.lr,
        loss_fn=config.training.loss_fn,
        **config.model,
    )

    # setup data
    train_set = QuantizationDataset(
        sequences=TRAIN_SEQUENCES, window_size=config.training.window_size)
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_train_workers)
    val_set = QuantizationDataset(
        sequences=TEST_SEQUENCES, window_size=config.training.window_size)
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_val_workers)

    # setup trainer
    logger = TensorBoardLogger("tb_logs/quantization", name="my_model")
    trainer = pl.Trainer(
        logger=logger,
        max_steps=config.training.training_steps,
    )

    # train
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train quantization.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/quantization.yml",
        help="Path to the configuration file.")
    args = parser.parse_args()

    train_vq_vae(args.config)
