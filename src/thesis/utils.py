""" Utilities."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from beartype import BeartypeConf, BeartypeStrategy, beartype
from jaxtyping import Float, Int, UInt8

from thesis.constants import SEGMENTATION_CLASSES

nobeartype = beartype(conf=BeartypeConf(strategy=BeartypeStrategy.O0))


def assign_segmentation_class(
        segmentation_mask: Float[torch.Tensor, "c h w 3"]) -> Int[torch.Tensor, "c h w"]:
    """Assign the colored pixel to it's corresponding class."""
    color_classes = torch.stack([
        torch.tensor(k, dtype=torch.float32, device=segmentation_mask.device) / 255
        for k in SEGMENTATION_CLASSES.keys()
    ])
    color_values = torch.tensor(
        list(SEGMENTATION_CLASSES.values()), dtype=torch.int64, device=segmentation_mask.device)

    def _assign_pixel(pixel: Float[torch.Tensor, "3"],
                      colors: Float[torch.Tensor, "c 3"]) -> Int[torch.Tensor, ""]:
        key_idx = torch.argmin(torch.linalg.norm(colors - pixel, dim=-1))
        return key_idx

    keys = torch.vmap(
        torch.vmap(torch.vmap(_assign_pixel, in_dims=(0, None)), in_dims=(0, None)),
        in_dims=(0, None))(segmentation_mask, color_classes)
    return color_values[keys]


def rotation_matrix_to_quaternion(
        rotation: Float[torch.Tensor, "... 3 3"]) -> Float[torch.Tensor, "... 4"]:
    """Convert a rotation matrix to a quaternion."""
    if rotation.size(-1) != 3 or rotation.size(-2) != 3:
        raise ValueError("Invalid rotation matrix shape. Expected (..., 3, 3)")

    batch_dim = rotation.shape[:-2]

    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]

    trace = m00 + m11 + m22

    def case_1():
        r = torch.sqrt(1 + trace)
        s = 0.5 / r
        w = 0.5 * r
        x = (m21-m12) * s
        y = (m02-m20) * s
        z = (m10-m01) * s
        return torch.stack([w, x, y, z], dim=-1)

    def case_2():
        r = torch.sqrt(1 + m00 - m11 - m22)
        s = 2.0 * r
        w = (m21-m12) / s
        x = 0.25 * s
        y = (m01+m10) / s
        z = (m02+m20) / s
        return torch.stack([w, x, y, z], dim=-1)

    def case_3():
        r = torch.sqrt(1 - m00 + m11 - m22)
        s = 2.0 * r
        w = (m02-m20) / s
        x = (m01+m10) / s
        y = 0.25 * s
        z = (m12+m21) / s
        return torch.stack([w, x, y, z], dim=-1)

    def case_4():
        r = torch.sqrt(1 - m00 - m11 + m22)
        s = 2.0 * r
        w = (m10-m01) / s
        x = (m02+m20) / s
        y = (m12+m21) / s
        z = 0.25 * s
        return torch.stack([w, x, y, z], dim=-1)

    where_1 = trace > 0
    where_2 = (m00 > m11) & (m00 > m22) & ~where_1
    where_3 = (m11 > m22) & ~where_1 & ~where_2
    where_4 = ~where_1 & ~where_2 & ~where_3

    result = torch.empty(batch_dim + (4,), dtype=rotation.dtype, device=rotation.device)

    if where_1.any():
        result[where_1] = case_1()[where_1]
    if where_2.any():
        result[where_2] = case_2()[where_2]
    if where_3.any():
        result[where_3] = case_3()[where_3]
    if where_4.any():
        result[where_4] = case_4()[where_4]

    return result


def quaternion_multiplication(
    quat_1: Float[torch.Tensor, "... 4"],
    quat_2: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 4"]:
    """Multiply two quaternions."""
    if quat_1.shape[-1] != 4 or quat_2.shape[-1] != 4:
        raise ValueError("Invalid quaternion shape. Expected (..., 4)")

    w1, x1, y1, z1 = torch.unbind(quat_1, -1)
    w2, x2, y2, z2 = torch.unbind(quat_2, -1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def datum_to_device(datum, device: torch.device | str):
    """Move the datum to the specified device."""

    new_datum = []
    for d in datum:
        if isinstance(d, torch.Tensor):
            new_datum.append(d.to(device))
        else:
            new_datum.append(datum_to_device(d, device))
    return type(datum)(*new_datum)


def apply_se3_to_point(
    rotation: Float[torch.Tensor, "... 3 3"],
    translation: Float[torch.Tensor, "... 3"],
    point: Float[torch.Tensor, "... 3"],
) -> Float[torch.Tensor, "... 3"]:
    """Apply the SE3 transform to a single point."""
    return torch.einsum("...ij,...j->...i", rotation, point) + translation


def apply_se3_to_orientation(
    rotation: Float[torch.Tensor, "... 3 3"],
    orientation: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 4"]:
    """Apply the SE3 transform to an orientation."""
    orientation = orientation / torch.norm(orientation, dim=-1, keepdim=True)
    rotation_quat = rotation_matrix_to_quaternion(rotation)
    return quaternion_multiplication(rotation_quat, orientation)


def generate_mesh_image(
    vertex_positions: Float[torch.Tensor, 'n_vertices 3'],
    faces: Int[torch.Tensor, 'n_faces 3'],
    image_height: int = 1_000,
    image_width: int = 1_000,
) -> UInt8[np.ndarray, "h w 3"]:
    """
    Generates an image of the mesh at the given time step.

    Args:
        vertex_positions: The vertex positions.
        faces: The faces of the mesh.
        image_height: The height of the image.
        image_width: The width of the image.

    Returns:
        The image of the mesh.
    """

    # Generate the vertices and faces
    vertices = vertex_positions.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()

    # Generate the matplotlib image
    matplotlib.use('agg')
    fig = plt.figure(figsize=(image_width / 100, image_height / 100))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        vertices[:, 2],
        vertices[:, 0],
        faces,
        vertices[:, 1],
        shade=True,
    )
    ax.view_init(azim=10, elev=10)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.set_axis_off()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close(fig)
    return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
