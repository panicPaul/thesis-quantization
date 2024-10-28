""" Constants for the project. """
import torch

DATA_DIR_NERSEMBLE = "../new_master_thesis/data/nersemble/Paul-audio-856/856"
"""
Directory containing the data for the NERsemble project.
"""

# from right to left, even numbers are level, while odd numbers are from below
TRAIN_CAMS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
TEST_CAMS = [8]
SERIALS = sorted([
    "220700191",
    "221501007",
    "222200036",
    "222200038",
    "222200039",
    "222200040",
    "222200041",
    "222200042",
    "222200043",
    "222200044",
    "222200045",
    "222200046",
    "222200047",
    "222200048",
    "222200049",
])
TRAIN_SEQUENCES = list(range(3, 80))
TEST_SEQUENCES = list(range(80, 102))

# flame constants
FLAME_MESH_PATH = "assets/flame/head_template_mesh.obj"
""" Path to the FLAME head template mesh. """
FLAME_LMK_PATH = "assets/flame/landmark_embedding_with_eyes.npy"
""" Path to the FLAME landmark embedding. """  # do I even need this?
# can be downloaded from https://flame.is.tue.mpg.de/download.php
FLAME_MODEL_PATH = "assets/flame/flame2023.pkl"  # FLAME 2023 (versions w/ jaw rotation)
""" Path to the FLAME model. """
FLAME_PARTS_PATH = "assets/flame/FLAME_masks.pkl"  # FLAME Vertex Masks
""" Path to the FLAME parts. """

# should all be from the same sequence and timestep
DEFAULT_SE3_ROTATION = torch.tensor(
    [[-0.85767704, 0.25412178, -0.44700348], [-0.12943351, -0.9480447, -0.29061666],
     [-0.4976313, -0.19139802, 0.8460081]],
    dtype=torch.float32)
""" Default SE3 rotation. """
DEFAULT_SE3_TRANSLATION = torch.tensor([0.04575331, 0.06610457, -0.05820158], dtype=torch.float32)
""" Default SE3 translation. """
CANONICAL_PCD = '/home/schlack/master-thesis/data/Paul-audio-85/085/sequences/SEN-05-glow_eyes_sweet_girl/timesteps/frame_00072/colmap/pointclouds/pointcloud_16.pcd'  # noqa
""" Canonical point cloud. """

SEGMENTATION_CLASSES = {
    (0, 0, 0): 0,
    (207, 2, 252): 0,
    (0, 255, 0): 1,
    (1, 171, 236): 2,
    (255, 127, 0): 3,
    (8, 208, 126): 4,
    (83, 130, 55): 5,
    (8, 4, 195): 6,
    (236, 186, 110): 7,
    (141, 223, 0): 8,
    (127, 255, 255): 9,
    (255, 255, 0): 10,
    (178, 139, 210): 11,
    (167, 5, 72): 12,
    (127, 255, 127): 13,
    (73, 91, 168): 14,
}
"""
Categories:
    - 0: background
    - 1: neck
    - 2: jumper
    - 3: face
    - 4: hair
    - 5: left ear
    - 6: right ear
    - 7: upper lip
    - 8: lower lip
    - 9: nose
    - 10: left eye
    - 11: right eye
    - 12: left eyebrow
    - 13: right eyebrow
    - 14: inner mouth
"""
