""" Utilities."""

import torch
from jaxtyping import Float


def datum_to_device(datum, device: torch.device | str):
    """Move the datum to the specified device."""

    l = []
    for d in datum:
        if isinstance(d, torch.Tensor):
            l.append(d.to(device))
        else:
            l.append(datum_to_device(d, device))
    return type(datum)(*l)


def apply_se3(
    rotation: Float[torch.Tensor, "... 3 3"],
    translation: Float[torch.Tensor, "... 3"],
    points: Float[torch.Tensor, "... 3"],
) -> Float[torch.Tensor, "... 3"]:
    """Apply the SE3 transform to the points.

    Args:
        rotation: Rotation matrix. Shape: `(..., 3, 3)`.
        translation: Translation vector. Shape: `(..., 3)`.
        points: Points to transform. Shape: `(..., 3)`.
    """
    return torch.einsum("...ij,...j->...i", rotation, points) + translation
