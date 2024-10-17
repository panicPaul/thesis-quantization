""" Some data classes for the thesis project. """

import torch
from jaxtyping import Float, Int
from typing import NamedTuple


# ==================================================================================== #
#                                 Data Classes                                         #
# ==================================================================================== #


class FlameParams(NamedTuple):
    """
    Args:
        shape (torch.Tensor): Shape parameters. Shape: `(batch, time, 300)`.
        expr (torch.Tensor): Expression parameters. Shape: `(batch, time, 100)`.
        neck (torch.Tensor): Neck pose parameters. Shape: `(batch, time, 3)`.
        jaw (torch.Tensor): Jaw pose parameters. Shape: `(batch, time, 3)`.
        eye (torch.Tensor): Eye pose parameters. Shape: `(batch, time, 6)`.
        scale (torch.Tensor): Scale parameters. Shape: `(batch, time, 1)`.
    """

    shape: Float[torch.Tensor, "batch time 300"]
    expr: Float[torch.Tensor, "batch time 100"]
    neck: Float[torch.Tensor, "batch time 3"]
    jaw: Float[torch.Tensor, "batch time 3"]
    eye: Float[torch.Tensor, "batch time 6"]
    scale: Float[torch.Tensor, "batch time 1"]


class SE3Transform(NamedTuple):
    """
    Args:
        rotation (torch.Tensor): Rotation matrix. Shape: `(3, 3)` or `(time, 3, 3)`.
        translation (torch.Tensor): Translation vector. Shape: `(3,)` or `(time, 3)`.
    """

    rotation: Float[torch.Tensor, "batch time 3 3"]
    translation: Float[torch.Tensor, "batch time 3"]


# ==================================================================================== #
#                              Unbatched versions                                      #
# ==================================================================================== #


class UnbatchedFlameParams(NamedTuple):
    """
    Args:
        shape (torch.Tensor): Shape parameters. Shape: `(time, 300)`.
        expr (torch.Tensor): Expression parameters. Shape: `(time, 100)`.
        neck (torch.Tensor): Neck pose parameters. Shape: `(time, 3)`.
        jaw (torch.Tensor): Jaw pose parameters. Shape: `(time, 3)`.
        eye (torch.Tensor): Eye pose parameters. Shape: `(time, 6)`.
        scale (torch.Tensor): Scale parameters. Shape: `(time, 1)`.
    """

    shape: Float[torch.Tensor, "time 300"]
    expr: Float[torch.Tensor, "time 100"]
    neck: Float[torch.Tensor, "time 3"]
    jaw: Float[torch.Tensor, "time 3"]
    eye: Float[torch.Tensor, "time 6"]
    scale: Float[torch.Tensor, "time 1"]


class UnbatchedSE3Transform(NamedTuple):
    """
    Args:
        rotation (torch.Tensor): Rotation matrix. Shape: `(time, 3, 3)` or `(3, 3)`.
        translation (torch.Tensor): Translation vector. Shape: `(time, 3)` or `(3,)`.
    """

    rotation: Float[torch.Tensor, "time 3 3"] | Float[torch.Tensor, "3 3"]
    translation: Float[torch.Tensor, "time 3"] | Float[torch.Tensor, "3"]


# ==================================================================================== #
#                              Data Batches                                            #
# ==================================================================================== #


class SingleFrameData(NamedTuple):
    """
    Args:
        image (torch.Tensor): Image tensor. Shape: `(cam, H, W, 3)`.
        mask (torch.Tensor): Mask tensor. Shape: `(cam, H, W)`.
        intrinsics (torch.Tensor): Intrinsics tensor. Shape: `(cam, 3, 3)`.
        extrinsics (torch.Tensor): Extrinsics tensor. Shape: `(cam, 4, 4)`.
        color_correction (torch.Tensor): Color correction tensor. Shape: `(cam, 3, 3)`.
        se3_transform (SE3Transform): SE3 transform object.
        sequence_id (torch.Tensor): Sequence ID tensor. Shape: `(cam,)`.
        time_step (torch.Tensor): Time step tensor. Shape: `(cam,)`.
    """

    image: Float[torch.Tensor, "cam H W 3"]
    mask: Float[torch.Tensor, "cam H W"]
    intrinsics: Float[torch.Tensor, "3 3"]
    extrinsics: Float[torch.Tensor, "cam 4 4"]
    color_correction: Float[torch.Tensor, "cam 3 3"]
    se3_transform: UnbatchedSE3Transform
    sequence_id: Int[torch.Tensor, ""]
    time_step: Int[torch.Tensor, ""]


class UnbatchedQuantizationData(NamedTuple):
    """
    Args:
        flame_params (FlameParams): Flame parameters object.
        se3_transform (SE3Transform): SE3 transform object.
        audio_features (torch.Tensor): Audio features tensor. Shape: `(time, 1024)`.
    """

    flame_params: UnbatchedFlameParams
    se3_transforms: UnbatchedSE3Transform
    audio_features: Float[torch.Tensor, "time 1024"]


class QuantizationData(NamedTuple):
    """
    Args:
        flame_params (FlameParams): Flame parameters object.
        se3_transform (SE3Transform): SE3 transform object.
        audio_features (torch.Tensor): Audio features tensor.
    """

    flame_params: FlameParams
    se3_transforms: SE3Transform
    audio_features: Float[torch.Tensor, "batch time 1024"]
