""" Audio to flame network."""

import argparse

import librosa
import lightning as pl
import pydub
import soundfile as sf
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from thesis.audio_feature_processing.audio_cleaning import pydub_to_np
from thesis.audio_feature_processing.wav2vec import (
    get_processor_and_model,
    process_sequence_interpolation,
)
from thesis.constants import TEST_SEQUENCES, TRAIN_SEQUENCES
from thesis.data_management import (
    FlameParams,
    QuantizationDataset,
    UnbatchedFlameParams,
)
from thesis.flame import FlameHead


class AudioToFlame(pl.LightningModule):

    def __init__(
        self,
        initial_projection: int = 128,
        window_size: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        lr: float = 3e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dropout = dropout
        self.initial_projection = nn.Conv1d(1024, initial_projection, 1)

        self.window_size = window_size
        input_layer = nn.Linear(initial_projection * window_size, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [input_layer] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.output_layer = nn.Linear(hidden_dim, 103)
        self.flame_head = FlameHead()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(
        self,
        audio_features: Float[torch.Tensor, 'batch window 1024'],
    ) -> tuple[Float[torch.Tensor, 'batch 100'], Float[torch.Tensor, 'batch 3']]:
        """ Returns the expression and jaw code for the middle frame of the window."""
        x = rearrange(audio_features, 'batch window d -> batch d window')
        x = self.initial_projection(x)
        x = nn.functional.silu(x)
        x = rearrange(x, 'batch d window -> batch (d window)')
        for layer in self.hidden_layers:
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = nn.functional.silu(layer(x))
        # x = nn.functional.sigmoid(self.output_layer(x))
        # maybe it's nn.tanh??

        x = self.output_layer(x)
        return x[:, :100], x[:, 100:]

    def to_flame(
        self,
        expression: Float[torch.Tensor, 'batch 100'],
        jaw_code: Float[torch.Tensor, 'batch 3'],
    ) -> Float[torch.Tensor, 'batch num_vertices 3']:
        """ Returns the flame vertices for the middle frame of the window."""
        batch_size = expression.shape[0]
        shape = torch.zeros(batch_size, 300, device=self.device)
        neck = torch.zeros(batch_size, 3, device=self.device)
        eye = torch.zeros(batch_size, 6, device=self.device)
        scale = torch.ones(batch_size, 1, device=self.device)
        flame_params = UnbatchedFlameParams(shape, expression, neck, jaw_code, eye, scale)
        return self.flame_head.forward(flame_params)

    def compute_loss(
        self,
        audio_features: Float[torch.Tensor, 'batch window 1024'],
        flame_params: FlameParams,
    ) -> dict[str, Float[torch.Tensor, '']]:
        """ Returns the L2 loss between the predicted flame vertices and the ground truth."""
        pred_expr, pred_jaw = self.forward(audio_features)
        pred_flame_vertices = self.to_flame(pred_expr, pred_jaw)
        batch_size = audio_features.shape[0]
        with torch.no_grad():
            target_params = UnbatchedFlameParams(
                shape=torch.zeros(batch_size, 300, device=self.device),
                expr=flame_params.expr[:, self.window_size // 2],
                neck=torch.zeros(batch_size, 3, device=self.device),
                jaw=flame_params.jaw[:, self.window_size // 2],
                eye=torch.zeros(batch_size, 6, device=self.device),
                scale=torch.ones(batch_size, 1, device=self.device),
            )
            target_vertices = self.flame_head.forward(target_params)
        loss = nn.functional.l1_loss(pred_flame_vertices, target_vertices)
        expr_loss = nn.functional.l1_loss(pred_expr, flame_params.expr[:, self.window_size // 2])
        jaw_loss = nn.functional.l1_loss(pred_jaw, flame_params.jaw[:, self.window_size // 2])
        return {'loss': loss, 'expr_loss': expr_loss, 'jaw_loss': jaw_loss}

    def training_step(self, batch, batch_idx: int) -> dict[str, Float[torch.Tensor, '']]:
        flame_params, _, audio_features = batch
        flame_params = FlameParams(*flame_params)
        loss = self.compute_loss(audio_features, flame_params)
        for key, value in loss.items():
            self.log(f'train/{key}', value, prog_bar=key == 'loss')
        return loss

    def validation_step(self, batch, batch_idx: int) -> dict[str, Float[torch.Tensor, '']]:
        flame_params, _, audio_features = batch
        flame_params = FlameParams(*flame_params)
        loss = self.compute_loss(audio_features, flame_params)
        for key, value in loss.items():
            self.log(f'val/{key}', value, prog_bar=key == 'loss')
        return loss


# ==================================================================================== #
#                               Prediction loop                                        #
# ==================================================================================== #


def prediction_loop(
    model_path: str,
    input_path: str | None = None,
    audio_features: Float[torch.Tensor, 'time 1024'] | None = None,
    batch_size: int = 64,
    fps: int = 30,
    device: torch.device | str = 'cuda',
    padding: bool = True,
) -> UnbatchedFlameParams:
    """
    Takes an audio file and returns the flame parameters for the sequence.

    Args:
        model: The trained model.
        input_path: The path to the audio file. Mutually exclusive with audio_features.
        audio_features: The audio features to process. Mutually exclusive with input_path.
        batch_size: Number of windows to process at once.
        device: Device to run inference on.
        padding: Whether to pad the flame parameters to match the length of the audio.

    Returns:
        The flame parameters for the sequence, padded to fit the length of the audio.
    """
    assert (input_path
            is None) != (audio_features
                         is None), "Either input_path or audio_features must be provided."
    # Load model
    print("Loading model... ", end="")
    model = AudioToFlame.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)
    print("\t\tDone!")

    if input_path is not None:
        # Get audio model
        print("Loading audio model...", end="")
        processor, wav2vec_model = get_processor_and_model()
        wav2vec_model = wav2vec_model.to(device)
        print("\t\tDone!")

        # Process audio features
        print("Processing audio features...", end="")
        file_format = input_path.split(".")[-1]
        if file_format == "m4a":
            # for macbook recordings
            audio = pydub.AudioSegment.from_file(input_path)
            audio, sr = pydub_to_np(audio)
        else:
            audio, sr = sf.read(input_path)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        sr = 16_000
        audio_length = len(audio) / sr  # in seconds
        n_frames = int(audio_length * fps)
        audio_features = process_sequence_interpolation(
            audio=audio,
            sampling_rate=sr,
            n_frames=n_frames,
            processor=processor,
            model=wav2vec_model,
            device=device,
        )
        print("\tDone!")
    else:
        assert isinstance(audio_features, torch.Tensor)
        n_frames = audio_features.shape[0]

    # Setup sliding windows
    window_size = model.window_size
    n_windows = n_frames - window_size + 1

    # Initialize output tensors for predictions
    all_expressions = torch.zeros((n_windows, 100), device=device)
    all_jaw_codes = torch.zeros((n_windows, 3), device=device)

    # Process in batches
    print("Predicting flame parameters...", end="")
    with torch.no_grad():
        for start_idx in range(0, n_windows, batch_size):
            end_idx = min(start_idx + batch_size, n_windows)

            # Create batch of windows
            batch_windows = torch.stack(
                [audio_features[i:i + window_size] for i in range(start_idx, end_idx)])

            # Forward pass
            expressions, jaw_codes = model(batch_windows)

            # Store results
            all_expressions[start_idx:end_idx] = expressions
            all_jaw_codes[start_idx:end_idx] = jaw_codes
    print("\tDone!")

    # Calculate padding sizes
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left

    # Apply reflection padding to both tensors
    if padding:
        reflection_mode = 'constant'
        padded_expressions = torch.nn.functional.pad(
            all_expressions.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            pad=(0, 0, pad_left, pad_right),  # (left, right, top, bottom)
            mode=reflection_mode)[0, 0]  # Remove batch and channel dims

        padded_jaw_codes = torch.nn.functional.pad(
            all_jaw_codes.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            pad=(0, 0, pad_left, pad_right),  # (left, right, top, bottom)
            mode=reflection_mode)[0, 0]  # Remove batch and channel dims

        assert padded_expressions.shape[
            0] == n_frames, f"Expected {n_frames} frames, got {padded_expressions.shape[0]}"
        assert padded_jaw_codes.shape[
            0] == n_frames, f"Expected {n_frames} frames, got {padded_jaw_codes.shape[0]}"
    else:
        padded_expressions = all_expressions
        padded_jaw_codes = all_jaw_codes

    # Convert to flame parameters
    shape = torch.zeros(n_frames, 300, device=device)
    neck = torch.zeros(n_frames, 3, device=device)
    eye = torch.zeros(n_frames, 6, device=device)
    scale = torch.ones(n_frames, 1, device=device)
    flame_params = UnbatchedFlameParams(
        shape=shape,
        expr=padded_expressions,
        neck=neck,
        jaw=padded_jaw_codes,
        eye=eye,
        scale=scale,
    )
    return flame_params


# ==================================================================================== #
#                               Train loop                                             #
# ==================================================================================== #


def train_audio_to_flame(config_path: str) -> None:
    """Train the audio to flame model."""
    config = OmegaConf.load(config_path)

    # setup model
    model = AudioToFlame(
        lr=config.training.learning_rate,
        **config.model,
    )

    # setup data
    train_set = QuantizationDataset(
        sequences=TRAIN_SEQUENCES, window_size=config.model.window_size)
    train_loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_train_workers)
    val_set = QuantizationDataset(sequences=TEST_SEQUENCES, window_size=config.model.window_size)
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_val_workers)

    # setup trainer
    logger = TensorBoardLogger("tb_logs/audio2flame", name="my_model")
    trainer = pl.Trainer(
        logger=logger,
        max_steps=config.training.training_steps,
    )

    # train
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train audio to flame.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/windowed_audio2flame.yml",
        help="Path to the configuration file.")
    args = parser.parse_args()

    train_audio_to_flame(args.config)
