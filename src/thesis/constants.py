""" Constants for the project. """

DATA_DIR_NERSEMBLE = "../new_master_thesis/data/nersemble/Paul-audio-856/856"
"""
Directory containing the data for the NERsemble project.
"""

# from right to left, even numbers are level, while odd numbers are from below
TRAIN_CAMS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
TEST_CAMS = [8]
SERIALS = sorted(
    [
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
    ]
)
TRAIN_SEQUENCES = list(range(3, 80))
TEST_SEQUENCES = list(range(80, 102))
