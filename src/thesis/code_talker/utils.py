""" some utility for the code talker """

import torch
from jaxtyping import Float

from thesis.data_management.data_classes import UnbatchedFlameParams


def flame_params_to_code(flame_params: UnbatchedFlameParams) -> Float[torch.Tensor, "... 413"]:
    """ Concatenates the flame params"""
    return torch.cat([
        flame_params.shape,
        flame_params.expr,
        flame_params.neck,
        flame_params.jaw,
        flame_params.eye,
        flame_params.scale,
    ],
                     dim=-1)


def flame_code_to_params(flame_code: Float[torch.Tensor, "... 413"],) -> UnbatchedFlameParams:
    """ Splits the flame code into the flame params"""
    return UnbatchedFlameParams(
        shape=flame_code[..., :300],
        expr=flame_code[..., 300:400],
        neck=flame_code[..., 400:403],
        jaw=flame_code[..., 403:406],
        eye=flame_code[..., 406:412],
        scale=flame_code[..., 412:413],
    )
