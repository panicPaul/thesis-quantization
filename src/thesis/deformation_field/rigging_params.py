""" Codebook of the motion rigging parameters. """

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from thesis.constants import TRAIN_SEQUENCES
from thesis.data_management import MultiSequenceManager, UnbatchedFlameParams
from thesis.flame import FlameHeadWithInnerMouth


class RiggingParams(nn.Module):
    """
    Saves the rigging parameters for the motion rigging.

    NOTE: Should be used with sparse optimizers!
    """

    def __init__(
        self,
        sequences: list[int] = TRAIN_SEQUENCES,
        num_flame_vertices: int = 5143,
        num_inner_mouth_vertices: int = 300,
    ) -> None:
        """
        Args:
            sequences (list[int]): list of sequences to be used for the rigging parameters.
            num_flame_vertices (int): number of flame vertices.
            num_inner_mouth_vertices (int): number of inner mouth vertices.
        """
        super().__init__()
        self.msm = MultiSequenceManager(sequences)
        flame_code_books = []
        inner_mouth_code_books = []
        # this is a bit hacky, but this way it can stay on the device
        idx_to_sequence = torch.ones(max(sequences) + 1, dtype=torch.int) * -1
        for i, sequence in enumerate(self.msm.sequences):
            idx_to_sequence[sequence] = i
            sequence_len = len(self.msm[i])
            flame_code_books.append(nn.Embedding(sequence_len, num_flame_vertices * 3))
            inner_mouth_code_books.append(nn.Embedding(sequence_len, num_inner_mouth_vertices * 3))

        self.flame_code_books = nn.ModuleList(flame_code_books)
        self.inner_mouth_code_books = nn.ModuleList(inner_mouth_code_books)

        self.register_buffer('idx_to_sequence', idx_to_sequence)
        self.register_buffer('device_check', torch.tensor(0))
        self.initialize_params()

    @property
    def device(self) -> torch.device:
        return self.device_check.device

    def initialize_params(self) -> None:
        """ Initialize the rigging parameters with the flame vertices. """
        device = self.device
        self.to('cuda')
        flame_head = FlameHeadWithInnerMouth().cuda()
        for sequence in self.msm.sequences:
            i = int(self.idx_to_sequence[sequence])
            flame_params = self.msm[i].flame_params[:]
            flame_params = UnbatchedFlameParams(
                shape=flame_params.shape.cuda(),
                expr=flame_params.expr.cuda(),
                neck=flame_params.neck.cuda(),
                jaw=flame_params.jaw.cuda(),
                eye=flame_params.eye.cuda(),
                scale=flame_params.scale.cuda(),
            )
            vertices = flame_head.forward(flame_params)  # (n_frames, n_vertices, 3)
            vertices = vertices.reshape(vertices.shape[0], -1)  # (n_frames, n_vertices * 3)
            flame_vertices = vertices[:, :5143 * 3]
            inner_mouth_vertices = vertices[:, 5143 * 3:]
            self.flame_code_books[i].weight.data = flame_vertices
            self.inner_mouth_code_books[i].weight.data = inner_mouth_vertices
        self.to(device)

    def forward(
        self,
        sequence: Int[torch.Tensor, ''] | int,
        frame: Int[torch.Tensor, ''] | int,
    ) -> Float[torch.Tensor, 'num_vertices 3']:
        """
        Args:
            sequence (torch.Tensor | int): sequence index.
            frame (torch.Tensor | int): frame index.
        Returns:
            rigging_params: rigging parameters for the given sequence and frame.
        """
        if isinstance(frame, int):
            frame = torch.tensor(frame, dtype=torch.int, device=self.device)
        idx = self.idx_to_sequence[sequence]
        flame_vertices = self.flame_code_books[idx](frame)
        flame_vertices = flame_vertices.reshape(-1, 3)
        inner_mouth_vertices = self.inner_mouth_code_books[idx](frame)
        inner_mouth_vertices = inner_mouth_vertices.reshape(-1, 3)
        vertices = torch.cat([flame_vertices, inner_mouth_vertices], dim=0)
        return vertices
