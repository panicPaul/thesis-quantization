""" Inverse flame model using MLP. """

import lightning as pl
import torch
import torch.nn as nn
from jaxtyping import Float
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from thesis.constants import TEST_SEQUENCES, TRAIN_SEQUENCES
from thesis.data_management import FlameParams, QuantizationDataset, SequenceManager
from thesis.flame import FlameHeadWithInnerMouth
from thesis.render_vertex_video import render_vertex_video


class InverseFlameMLP(pl.LightningModule):
    """ Compute the inverse flame model using an MLP. """

    def __init__(
        self,
        n_vertices: int = 5443,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.flame_head = FlameHeadWithInnerMouth()

        self.neck_embedding = nn.Sequential(nn.Linear(3, 512 * 3))

        self.mlp = nn.Sequential(
            nn.Linear(n_vertices*3 + 512*3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 112),
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """ Configure the optimizer. """
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def forward(
        self,
        shape_params: Float[torch.Tensor, "batch time 300"],
        scale_params: Float[torch.Tensor, "batch time 1"],
        neck_params: Float[torch.Tensor, "batch time 3"],
        vertices: Float[torch.Tensor, "batch time n_vertices 3"],
    ) -> FlameParams:
        """ Forward pass. """
        neck_embedding = self.neck_embedding(neck_params)
        x = vertices.reshape(vertices.shape[0], vertices.shape[1], -1)
        x = torch.cat([x, neck_embedding], dim=-1)
        x = self.mlp(x)

        return FlameParams(
            shape=shape_params,
            expr=x[..., :100],
            neck=x[..., 100:103],
            jaw=x[..., 103:106],
            eye=x[..., 106:112],
            scale=scale_params,
        )

    def compute_loss(
        self,
        prediction: FlameParams,
        target: FlameParams,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """ Compute the loss. """
        pred_vertices = self.flame_head(prediction)
        target_vertices = self.flame_head(target)
        vertex_loss = torch.nn.functional.mse_loss(pred_vertices, target_vertices)
        expr_loss = torch.nn.functional.mse_loss(prediction.expr, target.expr)
        neck_loss = torch.nn.functional.mse_loss(prediction.neck, target.neck)
        jaw_loss = torch.nn.functional.mse_loss(prediction.jaw, target.jaw)
        eye_loss = torch.nn.functional.mse_loss(prediction.eye, target.eye)
        loss = expr_loss + neck_loss + jaw_loss + eye_loss
        # loss = expr_loss
        return {
            "loss": loss,
            "vertex_loss": vertex_loss,
            "expr_loss": expr_loss,
            "neck_loss": neck_loss,
            "jaw_loss": jaw_loss,
            "eye_loss": eye_loss,
        }

    def training_step(
        self,
        batch: QuantizationDataset,
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """ Training step. """

        flame_params, _, _ = batch
        flame_params = FlameParams(*flame_params)
        vertices = self.flame_head(flame_params)
        prediction = self.forward(
            shape_params=flame_params.shape,
            scale_params=flame_params.scale,
            neck_params=flame_params.neck,
            vertices=vertices,
        )
        losses = self.compute_loss(prediction, flame_params)
        for key, value in losses.items():
            self.log(f"train/{key}", value)
        return losses['loss']

    def validation_step(
        self,
        batch: QuantizationDataset,
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """ Validation step. """

        flame_params, _, _ = batch
        flame_params = FlameParams(*flame_params)
        vertices = self.flame_head(flame_params)
        prediction = self.forward(
            shape_params=flame_params.shape,
            scale_params=flame_params.scale,
            neck_params=flame_params.neck,
            vertices=vertices,
        )
        losses = self.compute_loss(prediction, flame_params)
        for key, value in losses.items():
            self.log(f"val/{key}", value)
        return losses['loss']

    def render_reconstruction(
        self,
        sequence: int,
        device: torch.device | str = 'cuda',
    ) -> None:
        """ Renders a reconstruction"""
        sm = SequenceManager(sequence=sequence)
        flame_params = sm.flame_params[:]
        flame_params = FlameParams(
            shape=flame_params.shape.unsqueeze(0).to(device),
            expr=flame_params.expr.unsqueeze(0).to(device),
            neck=flame_params.neck.unsqueeze(0).to(device),
            jaw=flame_params.jaw.unsqueeze(0).to(device),
            eye=flame_params.eye.unsqueeze(0).to(device),
            scale=flame_params.scale.unsqueeze(0).to(device),
        )
        vertices = self.flame_head(flame_params)
        predicted_params = self.forward(
            shape_params=flame_params.shape,
            scale_params=flame_params.scale,
            vertices=vertices,
        )
        predicted_vertices = self.flame_head(predicted_params).squeeze(0)
        audio_path = '../new_master_thesis/data/nersemble/Paul-audio-856/856/sequences/' \
            f'sequence_{sequence:04d}/audio/audio_recording.ogg'
        render_vertex_video(
            vertices=predicted_vertices,
            faces=self.flame_head.faces,
            output_path=f"tmp/vertex2flame_reconstruction_{sequence}.mp4",
            audio_path=audio_path,
        )


def main() -> None:
    """ Train the model. """

    model = InverseFlameMLP()
    train_set = QuantizationDataset(TRAIN_SEQUENCES, window_size=1)
    val_set = QuantizationDataset(TEST_SEQUENCES, window_size=1)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
    logger = TensorBoardLogger("tb_logs/vertex_to_flame")

    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
    )
    model.compile()
    torch.set_float32_matmul_precision("high")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
