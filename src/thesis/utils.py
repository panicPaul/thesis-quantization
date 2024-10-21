""" Utilities."""

import torch
from jaxtyping import Float


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


def apply_se3(
    rotation: Float[torch.Tensor, "... 3 3"],
    translation: Float[torch.Tensor, "... 3"],
    points: Float[torch.Tensor, "... 3"],
    orientations: Float[torch.Tensor, "... 4]"] | None = None,
) -> (Float[torch.Tensor, "... 3"]
      | tuple[Float[torch.Tensor, "... 3"], Float[torch.Tensor, "... 4"]]):
    """Apply the SE3 transform to the points.

    Args:
        rotation: Rotation matrix. Shape: `(..., 3, 3)`.
        translation: Translation vector. Shape: `(..., 3)`.
        points: Points to transform. Shape: `(..., 3)`.
        orientations: Orientations to apply. Shape: `(..., 4)`.

    Returns:
        Either the transformed points or the transformed points and orientations.
    """
    points = torch.einsum("...ij,...j->...i", rotation, points) + translation
    if orientations is not None:
        orientations = torch.norm(orientations, dim=-1, keepdim=True)
        rotation_quat = rotation_matrix_to_quaternion(rotation)
        orientations = quaternion_multiplication(rotation_quat, orientations)
        return points, orientations
    return points