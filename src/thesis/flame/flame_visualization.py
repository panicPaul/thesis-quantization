""" Flame visualization module. """

from thesis.data_management import UnbatchedFlameParams
from thesis.flame import FlameHead


def _generate_image(
    flame_params: UnbatchedFlameParams,
    flame_head: FlameHead,
    image_height: int,
    image_width: int,
    time_step: int = 0,
):
    # ): -> Float[torch.Tensor, "h w 3"]:
    """
    Generates an image from the flame parameters.
    """
    flame_head.to(flame_params.shape.device)
    shape = flame_params.shape[time_step:time_step + 1]
    expr = flame_params.expr[time_step:time_step + 1]
    neck = flame_params.neck[time_step:time_step + 1]
    jaw = flame_params.jaw[time_step:time_step + 1]
    eye = flame_params.eye[time_step:time_step + 1]
    scale = flame_params.scale[time_step:time_step + 1]
    flame_params = UnbatchedFlameParams(shape, expr, neck, jaw, eye, scale)
    vertices = flame_head.forward(flame_params).squeeze(0).detach().cpu().numpy()
    faces = flame_head.faces.detach().cpu().numpy()
    return vertices, faces


def flame_video(flame_params: UnbatchedFlameParams, output_path: str, fps: int = 24) -> None:
    """ Generates a video, visualizing the flame parameters."""

    pass
