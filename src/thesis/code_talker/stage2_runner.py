""" Runner for the audio to vertex prediction."""

import argparse

import lightning as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from thesis.code_talker.models.code_talker_config import (
    CodeTalkerConfig,
    CodeTalkerTrainingConfig,
)
from thesis.code_talker.models.stage2 import CodeTalker
from thesis.code_talker.utils import flame_params_to_code
from thesis.constants import CANONICAL_FLAME_PARAMS, TEST_SEQUENCES, TRAIN_SEQUENCES
from thesis.data_management import (
    FlameParams,
    QuantizationDataset,
    UnbatchedFlameParams,
)
from thesis.flame import FlameHead, FlameHeadWithInnerMouth


# TODO: adapt time downsampling as well!
class Stage2Runner(pl.LightningModule):
    """ Training and Evaluation of Stage 2 models. """

    def __init__(self, config: CodeTalkerConfig,
                 training_config: CodeTalkerTrainingConfig) -> None:
        """ Descriptions in QuantizerConfig. """

        super().__init__()
        self.save_hyperparameters()
        self.model = CodeTalker(
            *config,
            reg_weight=training_config.reg_weight,
            motion_weight=training_config.motion_weight,
        )
        self.disable_neck = self.model.disable_neck
        n_vertices = self.model.autoencoder.n_vertices
        if n_vertices == 5443:
            self.flame_head = FlameHeadWithInnerMouth()
        else:
            self.flame_head = FlameHead()
        self.config = config
        self.training_config = training_config

        canonical_flame_params = UnbatchedFlameParams(*CANONICAL_FLAME_PARAMS)
        if self.model.disable_neck:
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
        self.register_buffer("canonical_flame_code", flame_params_to_code(canonical_flame_params))

    def configure_optimizers(self):
        """ Configures the optimizers and schedulers. """
        # NOTE: even tho we do have weight decay as a config option they do not use it in their
        #       code base for whatever reason.
        optimizer = torch.optim.AdamW(
            [{
                'params': self.model.audio_feature_map.parameters(),
            }, {
                'params': self.model.vertices_map.parameters(),
            }, {
                'params': self.model.PPE.parameters(),
            }, {
                'params': self.model.transformer_decoder.parameters(),
            }, {
                'params': self.model.feat_map.parameters(),
            }],
            lr=self.training_config.base_lr,
            weight_decay=0.002,
        )
        # scheduler = StepLR(
        #    optimizer, step_size=self.training_config.step_size, gamma=self.training_config.gamma)
        return optimizer

    def training_step(self, batch, batch_idx):
        """ Training step for the model. """
        # set up
        flame_params, _, audio_features = batch
        flame_params = FlameParams(*flame_params)
        if self.disable_neck:
            flame_params = FlameParams(
                shape=flame_params.shape,
                expr=flame_params.expr,
                neck=torch.zeros_like(flame_params.neck),
                jaw=flame_params.jaw,
                eye=flame_params.eye,
                scale=flame_params.scale,
            )

        if self.model.use_flame_code:
            x = flame_params_to_code(flame_params)
            batch_size = x.shape[0]
            template = self.canonical_flame_code.repeat(batch_size, 1)
        else:
            x = self.flame_head.forward(flame_params)  # (b, t, v, 3)
            batch_size = x.shape[0]
            template = self.canonical_flame_vertices.repeat(batch_size, 1, 1)  # (b, v, 3)

        # forward pass
        loss, motion_loss, reg_loss = self.model.forward(
            audio_features=audio_features,
            template=template,
            gt=x,
        )

        # logging
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/motion_loss', motion_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/reg_loss', reg_loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """ Validation step for the model. """
        # set up
        flame_params, _, audio_features = batch
        flame_params = FlameParams(*flame_params)
        if self.disable_neck:
            flame_params = FlameParams(
                shape=flame_params.shape,
                expr=flame_params.expr,
                neck=torch.zeros_like(flame_params.neck),
                jaw=flame_params.jaw,
                eye=flame_params.eye,
                scale=flame_params.scale,
            )

        if self.model.use_flame_code:
            x = flame_params_to_code(flame_params)
            batch_size = x.shape[0]
            template = self.canonical_flame_code.repeat(batch_size, 1)
        else:
            x = self.flame_head.forward(flame_params)  # (b, t, v, 3)
            batch_size = x.shape[0]
            template = self.canonical_flame_vertices.repeat(batch_size, 1, 1)  # (b, v, 3)

        # forward pass
        loss, motion_loss, reg_loss = self.model.forward(
            audio_features=audio_features,
            template=template,
            gt=x,
        )

        # logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/motion_loss', motion_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/reg_loss', reg_loss, on_step=False, on_epoch=True, prog_bar=False)

    def predict(self, audio_features):
        """ Predicts the flame vertices from the audio features and template. """
        self.eval()
        if self.model.use_flame_code:
            template = self.canonical_flame_code
        else:
            template = self.canonical_flame_vertices
        flame_vertices = self.model.predict(audio_features, template)

        if self.model.flame_code_head:
            flame_vertices = flame_vertices.unsqueeze(0)
            flame_params = self.model.autoencoder.flame_head_forward(flame_vertices)
            flame_params = UnbatchedFlameParams(
                shape=flame_params.shape.squeeze(0),
                expr=flame_params.expr.squeeze(0),
                neck=flame_params.neck.squeeze(0),
                jaw=flame_params.jaw.squeeze(0),
                eye=flame_params.eye.squeeze(0),
                scale=flame_params.scale.squeeze(0),
            )
            return flame_vertices.squeeze(0), flame_params
        else:
            return flame_vertices


# ==================================================================================== #
#                                   Training Loop                                      #
# ==================================================================================== #


def train_vq_vae(config_path: str) -> None:
    """ Training loop for the VQ-VAE model. """
    torch.set_float32_matmul_precision('high')
    # load config
    config = OmegaConf.load(config_path)
    model_config = CodeTalkerConfig(**config.model)
    training_config = CodeTalkerTrainingConfig(**config.training)
    window_size = None  # NOTE: this returns the entire sequence

    # set up
    model = Stage2Runner(model_config, training_config)
    if config.compile:
        model.compile()
    logger = TensorBoardLogger("tb_logs/audio_prediction", name=config.name)
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
        batch_size=1,
        num_workers=training_config.train_workers,
        shuffle=True,
        persistent_workers=True,
    )
    val_set = QuantizationDataset(
        sequences=TEST_SEQUENCES,
        window_size=window_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        num_workers=training_config.val_workers,
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
        default="configs/code_talker_prediction.yml",
        help="Path to the configuration file.")
    args = parser.parse_args()

    train_vq_vae(args.config)
