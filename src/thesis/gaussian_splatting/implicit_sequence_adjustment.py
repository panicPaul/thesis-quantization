""" Implicit representation of sequence adjustments."""

import torch
import torch.nn as nn
from jaxtyping import Float, Int


class ImplicitSequenceAdjustment(nn.Module):
    """ Adds a bias to the vertices of the input sequence."""

    def __init__(self, num_vertices: int = 5443, num_sequences: int = 80) -> None:
        super().__init__()
        self.num_vertices = num_vertices
        self.num_sequences = num_sequences

        self.sequence_codebook = nn.Embedding(num_sequences, 8)

        self.mlp = nn.Sequential(
            nn.Linear(8 + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, num_vertices * 3),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        sequence: Int[torch.Tensor, ""],
        time_step: Int[torch.Tensor, ""],
    ) -> Float[torch.Tensor, 'n_vertices 3']:
        """
        Args:
            sequence: The sequence index.
            time_step: The time step index.

        Returns:
            The bias for the vertices of the input sequence at the given time step.
        """

        sequence = sequence.unsqueeze(0)
        time_step = time_step.unsqueeze(0).unsqueeze(0) / 1_000
        time_step_sin = torch.sin(time_step)
        time_step_cos = torch.cos(time_step)
        x = torch.cat([self.sequence_codebook(sequence), time_step_sin, time_step_cos], dim=-1)
        # x = self.mlp(x).squeeze(0) * 1e-4
        # x = x.clamp(-3e-2, 3e-2)  # hard clamp to avoid total collapse
        x = self.mlp(x).squeeze(0) * 1e-2  # encourages small adjustments
        x = nn.functional.tanh(x) * 5e-2  # hard clamp to avoid total collapse
        x = x.view(-1, 3)
        # replace NaNs with zeros
        # x[torch.isnan(x)] = 0.0
        if torch.isnan(x).any():
            print("NaNs found in the output of the implicit sequence adjustment.")
        return x
