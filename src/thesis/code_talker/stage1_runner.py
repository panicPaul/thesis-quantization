""" Running and evaluation of models. """

import argparse

import lightning as pl
import torch
import torch.nn as nn
from jaxtyping import Float
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from thesis.code_talker.models.code_talker_config import (
    QuantizationTrainingConfig,
    QuantizerConfig,
)
from thesis.code_talker.models.stage1_nersemble import VQAutoEncoder
from thesis.constants import CANONICAL_FLAME_PARAMS, TEST_SEQUENCES, TRAIN_SEQUENCES
from thesis.data_management import (
    FlameParams,
    QuantizationDataset,
    UnbatchedFlameParams,
)
from thesis.flame import FlameHead


class Stage1Runner(pl.LightningModule):
    """ Training and Evaluation of Stage 1 models. """

    def __init__(
        self,
        config: QuantizerConfig,
        training_config: QuantizationTrainingConfig,
    ) -> None:
        """ Descriptions in QuantizerConfig. """

        super().__init__()
        self.save_hyperparameters()
        self.model = VQAutoEncoder(*config)
        self.flame_head = FlameHead()
        self.config = config
        self.training_config = training_config
        canonical_flame_params = UnbatchedFlameParams(*CANONICAL_FLAME_PARAMS)
        canonical_flame_vertices = self.flame_head.forward(canonical_flame_params)
        canonical_flame_vertices = canonical_flame_vertices.flatten().unsqueeze(0)  # (1, v*3)
        self.register_buffer("canonical_flame_vertices", canonical_flame_vertices)

    def configure_optimizers(self):
        """ Configures the optimizers and schedulers. """
        #NOTE: even tho we do have weight decay as a config option they do not use it in their
        #      code base for whatever reason.
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.base_lr,
        )
        scheduler = StepLR(
            optimizer, step_size=self.training_config.step_size, gamma=self.training_config.gamma)
        return [optimizer], [scheduler]

    def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
        """ function that computes the various components of the VQ loss """
        rec_loss = nn.L1Loss()(pred, target)
        ## loss is VQ reconstruction + weighted pre-computed quantization loss
        quant_loss = quant_loss.mean()
        return quant_loss*quant_loss_weight + rec_loss, [rec_loss, quant_loss]

    def training_step(self, batch, batch_idx):
        """ Training step for the model. """
        # set up
        flame_params, _, _ = batch
        flame_params = FlameParams(*flame_params)
        flame_vertices = self.flame_head.forward(flame_params)  # (b, t, v, 3)
        batch_size, timesteps, _, _ = flame_vertices.shape
        flame_vertices = flame_vertices.view(batch_size, timesteps, -1)
        template = self.canonical_flame_vertices.repeat(batch_size, 1)

        # forward pass
        rec, quant_loss, info = self.model.forward(flame_vertices, template)
        loss, losses = self.calc_vq_loss(rec, flame_vertices, quant_loss)

        # logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/rec_loss', losses[0], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/quant_loss', losses[1], on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """ Validation step for the model. """
        # set up
        flame_params, _, _ = batch
        flame_params = FlameParams(*flame_params)
        flame_vertices = self.flame_head.forward(flame_params)
        batch_size, timesteps, num_vertices, _ = flame_vertices.shape
        flame_vertices = flame_vertices.view(batch_size, timesteps, -1)
        template = self.canonical_flame_vertices.repeat(batch_size, 1)

        # forward pass
        rec, quant_loss, info = self.model.forward(flame_vertices, template)
        loss, losses = self.calc_vq_loss(rec, flame_vertices, quant_loss)

        # logging
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/rec_loss', losses[0], on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/quant_loss', losses[1], on_step=True, on_epoch=True, prog_bar=True)

        return loss


# ==================================================================================== #
#                                   Training Loop                                      #
# ==================================================================================== #


def train_vq_vae(config_path: str) -> None:
    """ Training loop for the VQ-VAE model. """
    # load config
    config = OmegaConf.load(config_path)
    training_config = QuantizationTrainingConfig(**config.training)
    config = QuantizerConfig(**config.model)

    # set up
    model = Stage1Runner(config, training_config)
    logger = TensorBoardLogger("tb_logs/audio2flame", name="my_model")
    trainer = pl.Trainer(max_epochs=training_config.epochs, logger=logger)

    # setup data
    train_set = QuantizationDataset(sequences=TRAIN_SEQUENCES, window_size=1)
    train_loader = DataLoader(
        train_set,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_train_workers)
    val_set = QuantizationDataset(sequences=TEST_SEQUENCES, window_size=1)
    val_loader = DataLoader(
        val_set,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_val_workers,
    )

    # train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio to flame.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/code_talker_vq_vae.yml",
        help="Path to the configuration file.")
    args = parser.parse_args()

    train_vq_vae(args.config)
