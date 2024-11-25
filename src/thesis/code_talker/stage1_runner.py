""" Running and evaluation of stage 1 model. """

import argparse

import lightning as pl
import torch
import torch.nn as nn
from einops import repeat
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
from thesis.constants import (
    CANONICAL_FLAME_PARAMS,
    DATA_DIR_NERSEMBLE,
    OTHER_GUY_DATA_DIR,
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
)
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
        self.canonical_flame_params = UnbatchedFlameParams(
            shape=canonical_flame_params.shape.cuda(),
            expr=canonical_flame_params.expr.cuda(),
            neck=canonical_flame_params.neck.cuda(),
            jaw=canonical_flame_params.jaw.cuda(),
            eye=canonical_flame_params.eye.cuda(),
            scale=canonical_flame_params.scale.cuda(),
        )
        canonical_flame_vertices = self.flame_head.forward(canonical_flame_params)  # (1, v, 3)
        self.register_buffer("canonical_flame_vertices", canonical_flame_vertices)

    def configure_optimizers(self):
        """ Configures the optimizers and schedulers. """
        # NOTE: even tho we do have weight decay as a config option they do not use it in their
        #       code base for whatever reason.
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.base_lr,
        )
        scheduler = StepLR(
            optimizer, step_size=self.training_config.step_size, gamma=self.training_config.gamma)
        return [optimizer], [scheduler]

    # TODO: experiment with predicting flame vertices directly
    def calc_vq_loss(self,
                     pred_vertices,
                     pred_flame_code,
                     pred_audio,
                     target_vertices,
                     target_flame_params,
                     target_audio,
                     quant_loss,
                     quant_loss_weight=1.0,
                     audio_loss_weight=1e-3,
                     alpha=1.0):
        """ function that computes the various components of the VQ loss """
        loss = torch.tensor(0.0, device=target_vertices.device)
        losses = {}
        match self.config.flame_mode:
            case 'flame':
                target_flame_code = torch.cat([target_flame_params.expr, target_flame_params.jaw],
                                              dim=-1)
                flame_code_loss = nn.functional.l1_loss(pred_flame_code, target_flame_code)
                loss = loss + flame_code_loss*1e-3
                losses['flame_code_loss'] = flame_code_loss
                flame_params = FlameParams(
                    shape=target_flame_params.shape,
                    expr=pred_flame_code[:, :, :100],
                    neck=target_flame_params.neck,
                    jaw=pred_flame_code[:, :, 100:103],
                    eye=target_flame_params.eye,
                    scale=target_flame_params.scale,
                )
                pred_vertices = self.flame_head.forward(flame_params)
                vertex_loss = nn.functional.l1_loss(pred_vertices, target_vertices)
                loss = loss + vertex_loss
                losses['vertex_loss'] = vertex_loss
            case 'vertex':
                vertex_loss = nn.functional.l1_loss(pred_vertices, target_vertices)
                loss = loss + vertex_loss
                losses['vertex_loss'] = vertex_loss
            case _:
                raise ValueError(f"Invalid flame mode: {self.config.flame_mode}")
        if self.config.use_audio:
            audio_rec_loss = nn.functional.l1_loss(pred_audio, target_audio)
            loss = loss + audio_loss_weight*audio_rec_loss
            losses['audio_loss'] = audio_rec_loss

        loss = quant_loss*quant_loss_weight + loss
        losses['quant_loss'] = quant_loss
        return loss, losses

    def training_step(self, batch, batch_idx):
        """ Training step for the model. """
        # set up
        flame_params, _, audio_features = batch
        flame_params = FlameParams(*flame_params)
        if self.config.disable_neck:
            flame_params = FlameParams(
                shape=flame_params.shape.cuda(),
                expr=flame_params.expr.cuda(),
                neck=torch.zeros_like(flame_params.neck).cuda(),
                jaw=flame_params.jaw.cuda(),
                eye=flame_params.eye.cuda(),
                scale=flame_params.scale.cuda(),
            )
        flame_code = torch.cat([flame_params.expr, flame_params.jaw], dim=-1)
        flame_vertices = self.flame_head.forward(flame_params)  # (b, t, v, 3)
        batch_size = flame_vertices.shape[0]
        template = self.canonical_flame_vertices.repeat(batch_size, 1, 1)  # (b, v, 3)

        # sub sampling
        min_size = 45
        max_size = flame_vertices.shape[1]
        size = torch.randint(min_size, max_size, (1,)).item()
        start_idx = torch.randint(0, max_size - size, (1,)).item()
        flame_vertices = flame_vertices[:, start_idx:start_idx + size]
        audio_features = audio_features[:, start_idx:start_idx + size]
        flame_code = flame_code[:, start_idx:start_idx + size]
        flame_params = FlameParams(
            shape=flame_params.shape[:, start_idx:start_idx + size],
            expr=flame_params.expr[:, start_idx:start_idx + size],
            neck=flame_params.neck[:, start_idx:start_idx + size],
            jaw=flame_params.jaw[:, start_idx:start_idx + size],
            eye=flame_params.eye[:, start_idx:start_idx + size],
            scale=flame_params.scale[:, start_idx:start_idx + size],
        )

        # audio masking
        audio_features_in = nn.functional.dropout(audio_features, p=0.5)

        # forward pass
        rec_vertices, rec_flame_code, rec_audio, quant_loss, info = self.model.forward(
            template=template,
            vertices=flame_vertices,
            flame_code=flame_code,
            audio_features=audio_features_in,
        )
        loss, losses = self.calc_vq_loss(
            pred_vertices=rec_vertices,
            pred_flame_code=rec_flame_code,
            pred_audio=rec_audio,
            target_vertices=flame_vertices,
            target_flame_params=flame_params,
            target_audio=audio_features,
            quant_loss=quant_loss,
        )

        # logging
        for k, v in losses.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=True)

        # log reconstructed picture
        if self.global_step % 250 == 0 and self.config.flame_mode == 'vertex':
            img = render_mesh_image(
                vertex_positions=rec_vertices[0][0],  # (v, 3)
                faces=self.flame_head.faces,
            )
            self.logger.experiment.add_image(
                "train/reconstructed", img, self.global_step, dataformats="HWC")

        return loss

    def validation_step(self, batch, batch_idx):
        """ Validation step for the model. """
        # set up
        flame_params, _, audio_features = batch
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
        flame_code = torch.cat([flame_params.expr, flame_params.jaw], dim=-1)
        batch_size = flame_vertices.shape[0]
        template = self.canonical_flame_vertices.repeat(batch_size, 1, 1)  # (b, v, 3)

        # forward pass
        rec_vertices, rec_flame_code, rec_audio, quant_loss, info = self.model.forward(
            template=template,
            vertices=flame_vertices,
            flame_code=flame_code,
            audio_features=audio_features,
        )

        loss, losses = self.calc_vq_loss(
            pred_vertices=rec_vertices,
            pred_flame_code=rec_flame_code,
            pred_audio=rec_audio,
            target_vertices=flame_vertices,
            target_flame_params=flame_params,
            target_audio=audio_features,
            quant_loss=quant_loss,
        )
        # logging
        for k, v in losses.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def predict(
        self,
        flame_params: UnbatchedFlameParams,
        audio_features: Float[torch.Tensor, "time feature_dim"] | None = None,
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
        if audio_features is not None:
            audio_features = audio_features.unsqueeze(0)
        flame_vertices = self.flame_head.forward(flame_params)
        flame_code = torch.cat([flame_params.expr, flame_params.jaw], dim=-1)
        template = self.canonical_flame_vertices
        rec_vertices, rec_flame_code, _, _, _ = self.model.forward(
            template=template,
            vertices=flame_vertices,
            flame_code=flame_code,
            audio_features=audio_features,
        )
        if self.config.flame_mode == 'flame':
            flame_params = FlameParams(
                shape=flame_params.shape,
                expr=rec_flame_code[:, :, :100],
                neck=flame_params.neck,
                jaw=rec_flame_code[:, :, 100:103],
                eye=flame_params.eye,
                scale=flame_params.scale,
            )
            rec_vertices = self.flame_head.forward(flame_params)

        return rec_vertices.squeeze(0)


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
        data_dir=DATA_DIR_NERSEMBLE if not training_config.use_other_guy else OTHER_GUY_DATA_DIR,
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
        data_dir=DATA_DIR_NERSEMBLE if not training_config.use_other_guy else OTHER_GUY_DATA_DIR,
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
