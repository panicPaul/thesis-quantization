""" Running and evaluation of stage 1 model. """

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
from thesis.flame import FlameHead, FlameHeadWithInnerMouth
from thesis.video_utils import render_mesh_image


# TODO: adapt time downsampling as well!
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
        if config.n_vertices == 5443:
            self.flame_head = FlameHeadWithInnerMouth()
        else:
            self.flame_head = FlameHead()
        self.model = VQAutoEncoder(*config)
        self.config = config
        self.training_config = training_config
        canonical_flame_params = UnbatchedFlameParams(*CANONICAL_FLAME_PARAMS)
        if config.disable_neck:
            canonical_flame_params = UnbatchedFlameParams(
                shape=canonical_flame_params.shape,
                expr=canonical_flame_params.expr,
                neck=torch.zeros_like(canonical_flame_params.neck),
                jaw=canonical_flame_params.jaw,
                eye=canonical_flame_params.eye,
                scale=canonical_flame_params.scale,
            )
        canonical_flame_vertices = self.flame_head.forward(canonical_flame_params)  # (1, v, 3)
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

    # TODO: experiment with predicting flame vertices directly
    def calc_vq_loss(self, pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
        """ function that computes the various components of the VQ loss """
        rec_loss = nn.functional.l1_loss(pred, target)
        ## loss is VQ reconstruction + weighted pre-computed quantization loss
        quant_loss = quant_loss.mean()
        return quant_loss*quant_loss_weight + rec_loss, [rec_loss, quant_loss]

    def training_step(self, batch, batch_idx):
        """ Training step for the model. """
        # set up
        flame_params, _, _ = batch
        flame_params = FlameParams(*flame_params)
        if self.config.disable_neck:
            flame_params = FlameParams(
                shape=flame_params.shape,
                expr=flame_params.expr,
                neck=torch.zeros_like(flame_params.neck),
                jaw=flame_params.jaw,
                eye=flame_params.eye,
                scale=flame_params.scale,
            )
        flame_vertices = self.flame_head.forward(flame_params)  # (b, t, v, 3)
        batch_size = flame_vertices.shape[0]
        template = self.canonical_flame_vertices.repeat(batch_size, 1, 1)  # (b, v, 3)

        # forward pass
        rec, quant_loss, info = self.model.forward(flame_vertices, template)
        loss, losses = self.calc_vq_loss(rec, flame_vertices, quant_loss)

        # logging
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/rec_loss', losses[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/quant_loss', losses[1], on_step=False, on_epoch=True, prog_bar=False)

        # log reconstructed picture
        if self.global_step % 250 == 0:
            img = render_mesh_image(
                vertex_positions=rec[0][0],  # (v, 3)
                faces=self.flame_head.faces,
            )
            self.logger.experiment.add_image(
                "train/reconstructed", img, self.global_step, dataformats="HWC")

        return loss

    def validation_step(self, batch, batch_idx):
        """ Validation step for the model. """
        # set up
        flame_params, _, _ = batch
        flame_params = FlameParams(*flame_params)
        if self.config.disable_neck:
            flame_params = FlameParams(
                shape=flame_params.shape,
                expr=flame_params.expr,
                neck=torch.zeros_like(flame_params.neck),
                jaw=flame_params.jaw,
                eye=flame_params.eye,
                scale=flame_params.scale,
            )
        flame_vertices = self.flame_head.forward(flame_params)  # (b, t, v, 3)
        batch_size = flame_vertices.shape[0]
        template = self.canonical_flame_vertices.repeat(batch_size, 1, 1)  # (b, v, 3)

        # forward pass
        rec, quant_loss, info = self.model.forward(flame_vertices, template)
        loss, losses = self.calc_vq_loss(rec, flame_vertices, quant_loss)

        # logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/rec_loss', losses[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/quant_loss', losses[1], on_step=False, on_epoch=True, prog_bar=False)

        return loss

    @torch.no_grad()
    def predict(
        self,
        flame_params: UnbatchedFlameParams,
    ) -> Float[torch.Tensor, "time n_vertices 3"]:
        """
        Returns the reconstructed flame vertices.

        Args:
            flame_params: The flame parameters to predict the vertices for.

        Returns:
            The reconstructed flame vertices.
        """
        self.model.eval()
        flame_params = FlameParams(
            shape=flame_params.shape.unsqueeze(0),
            expr=flame_params.expr.unsqueeze(0),
            neck=torch.zeros_like(flame_params.neck).unsqueeze(0)
            if self.config.disable_neck else flame_params.neck.unsqueeze(0),
            jaw=flame_params.jaw.unsqueeze(0),
            eye=flame_params.eye.unsqueeze(0),
            scale=flame_params.scale.unsqueeze(0),
        )
        flame_vertices = self.flame_head.forward(flame_params)
        template = self.canonical_flame_vertices
        rec, _, _ = self.model.forward(flame_vertices, template)
        return rec.squeeze(0)


# ==================================================================================== #
#                                   Training Loop                                      #
# ==================================================================================== #


def train_vq_vae(config_path: str) -> None:
    """ Training loop for the VQ-VAE model. """
    torch.set_float32_matmul_precision('high')
    # load config
    config = OmegaConf.load(config_path)
    training_config = QuantizationTrainingConfig(**config.training)
    model_config = QuantizerConfig(**config.model)
    window_size = None  # NOTE: this returns the entire sequence

    # set up
    model = Stage1Runner(model_config, training_config)
    if config.compile:
        model.compile()
    logger = TensorBoardLogger("tb_logs/vector_quantization", name=config.name)
    trainer = pl.Trainer(
        max_epochs=training_config.epochs,
        logger=logger,
        check_val_every_n_epoch=5,
    )

    # setup data
    train_set = QuantizationDataset(
        sequences=TRAIN_SEQUENCES,
        window_size=window_size,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_train_workers,
        shuffle=True,
        persistent_workers=True,
    )
    val_set = QuantizationDataset(
        sequences=TEST_SEQUENCES,
        window_size=window_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_val_workers,
        shuffle=False,
        persistent_workers=True,
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
