"""
Hacky way to add more vertices to the inside of the mouth, which should enable the flame to be more
expressive.
"""

import torch
import torch.nn as nn
from jaxtyping import Float

from thesis.data_management import FlameParams, UnbatchedFlameParams
from thesis.flame.flame_head import FlameHead


class FlameHeadWithInnerMouth(FlameHead):
    """ Adds more vertices to the inside of the mouth. """

    def __init__(self) -> None:
        super().__init__()
        self.rows_to_add = 5
        teeth_upper_upper_front = self.mask.v.teeth_upper[:15]
        teeth_upper_lower_front = self.mask.v.teeth_upper[15:30]
        teeth_upper_upper_back = self.mask.v.teeth_upper[30:45]
        teeth_upper_lower_back = self.mask.v.teeth_upper[45:60]

        teeth_lower_upper_front = self.mask.v.teeth_lower[:15]
        teeth_lower_lower_front = self.mask.v.teeth_lower[15:30]
        teeth_lower_upper_back = self.mask.v.teeth_lower[30:45]
        teeth_lower_lower_back = self.mask.v.teeth_lower[45:60]

        # register as buffer
        self.register_buffer("teeth_upper_upper_front", teeth_upper_upper_front)
        self.register_buffer("teeth_upper_lower_front", teeth_upper_lower_front)
        self.register_buffer("teeth_upper_upper_back", teeth_upper_upper_back)
        self.register_buffer("teeth_upper_lower_back", teeth_upper_lower_back)

        self.register_buffer("teeth_lower_upper_front", teeth_lower_upper_front)
        self.register_buffer("teeth_lower_lower_front", teeth_lower_lower_front)
        self.register_buffer("teeth_lower_upper_back", teeth_lower_upper_back)
        self.register_buffer("teeth_lower_lower_back", teeth_lower_lower_back)

        # NOTE: there are 5143 vertices in total, 15 vertices, per row and level, so 60 vertices
        #       per additional row for all 4 levels.
        self.register_buffer(
            'inner_mouth_indices',
            torch.arange(5143, 5143 + 60 * self.rows_to_add),
        )

    def _add_inner_mouth_vertices(
        self,
        vertices: Float[torch.Tensor, "num_vertices 3"],
    ) -> Float[torch.Tensor, "num_total_vertices 3"]:
        """
        Adds more vertices to the inside of the mouth. We just use the teeth planes and offset them
        by the same amount to get the new vertices. We also leave one row blank between the teeth
        planes and the new vertices.
        """

        # Get the vertices
        teeth_upper_upper_front = vertices[self.teeth_upper_upper_front]
        teeth_upper_lower_front = vertices[self.teeth_upper_lower_front]
        teeth_upper_upper_back = vertices[self.teeth_upper_upper_back]
        teeth_upper_lower_back = vertices[self.teeth_upper_lower_back]

        teeth_lower_upper_front = vertices[self.teeth_lower_upper_front]
        teeth_lower_lower_front = vertices[self.teeth_lower_lower_front]
        teeth_lower_upper_back = vertices[self.teeth_lower_upper_back]
        teeth_lower_lower_back = vertices[self.teeth_lower_lower_back]

        # Get the offsets
        upper_upper_displacement = teeth_upper_upper_back - teeth_upper_upper_front
        upper_lower_displacement = teeth_upper_lower_back - teeth_upper_lower_front
        lower_upper_displacement = teeth_lower_upper_back - teeth_lower_upper_front
        lower_lower_displacement = teeth_lower_lower_back - teeth_lower_lower_front

        # Add the new vertices
        new_vertices = []
        for i in range(1, self.rows_to_add + 1):
            new_vertices.append(teeth_upper_upper_back + upper_upper_displacement * (i+1))
            new_vertices.append(teeth_upper_lower_back + upper_lower_displacement * (i+1))
            new_vertices.append(teeth_lower_upper_back + lower_upper_displacement * (i+1))
            new_vertices.append(teeth_lower_lower_back + lower_lower_displacement * (i+1))
        new_vertices = torch.concatenate(new_vertices, dim=0)

        # Concatenate the new vertices
        vertices = torch.cat([vertices, new_vertices], dim=0)

        return vertices

    def forward(
        self, params: FlameParams | UnbatchedFlameParams
    ) -> (Float[torch.Tensor, "batch time num_vertices 3"]
          | Float[torch.Tensor, "time num_vertices 3"]):
        """
        Args:
            params (FlameParams): The flame parameters.

        Returns:
            Float[torch.Tensor, "batch time num_vertices 3"]: The vertices.
        """

        # Get the vertices from the base class and add the inner mouth vertices
        vertices = super().forward(params)
        if vertices.ndim == 3:
            vertices = torch.vmap(self._add_inner_mouth_vertices)(vertices)
        else:
            batch, time, n_vertices = vertices.shape[:3]
            # collapse the batch and time dimensions
            vertices = vertices.reshape(batch * time, n_vertices, 3)
            vertices = torch.vmap(self._add_inner_mouth_vertices)(vertices)
            vertices = vertices.reshape(batch, time, -1, 3)
        return vertices
