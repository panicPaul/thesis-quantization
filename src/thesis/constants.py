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

DEFAULT_SE3_ROTATION = torch.tensor(
    [[0.8912, 0.0324, 0.4526], [-0.1756, 0.9446, 0.2781], [-0.4186, -0.3275, 0.8472]],
    dtype=torch.float32)
""" Default SE3 rotation. """
DEFAULT_SE3_TRANSLATION = torch.tensor([0.0602, 0.0821, -0.1438], dtype=torch.float32)
""" Default SE3 translation. """

CANONICAL_PCD = '/home/schlack/master-thesis/data/Paul-audio-85/085/sequences/SEN-05-glow_eyes_sweet_girl/timesteps/frame_00072/colmap/pointclouds/pointcloud_16.pcd'  # noqa
