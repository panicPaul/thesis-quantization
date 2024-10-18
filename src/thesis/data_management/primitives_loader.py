"""
Some data, such as images does not exist as one contiguous array, but rather as a
collection of files on disk. In this case, we need a class to load the data from disk
and provide it in a format that can be used by the rest of the code.
"""

import json
import os

import numpy as np
import torch
from einops import repeat
from jaxtyping import Float
from PIL import Image

from thesis.constants import DATA_DIR_NERSEMBLE, TRAIN_CAMS
from thesis.data_management.data_classes import (
    UnbatchedFlameParams,
    UnbatchedSE3Transform,
)

# ==================================================================================== #
#                                 Images                                               #
# ==================================================================================== #


class _SequenceLoader:
    def init(
        self,
        sequence: str | int,
        data_dir: str,
        image_downsampling_factor: int | float,
        cameras: list[int],
    ) -> None:
        """Init function is sh"""
        if isinstance(sequence, int):
            assert sequence in range(3, 102), "Invalid sequence number."
        if isinstance(sequence, int):
            sequence = f"sequence_{sequence:04d}"
        # NOTE: not sure if using class variables actually helps with multi threading,
        #       but just to be safe, I will use them here.
        self.sequence = sequence
        self.image_downsampling_factor = image_downsampling_factor
        self.data_dir = data_dir
        camera_config = json.load(
            open(os.path.join(data_dir, "calibration/config.json"))
        )
        self.image_width = int(
            camera_config["image_size"][0] // (2 * image_downsampling_factor)
        )
        self.image_height = int(
            camera_config["image_size"][1] // (2 * image_downsampling_factor)
        )
        self.camera_params = json.load(
            open(os.path.join(data_dir, "calibration/camera_params.json"))
        )

        self.serials = [
            list(self.camera_params["world_2_cam"].keys())[c] for c in cameras
        ]
        self.camera_ids = np.array(cameras)

        # get the time steps
        frames_dir = os.path.join(data_dir, "sequences", sequence, "timesteps")
        frames = os.listdir(frames_dir)
        self.frames = sorted([int(f.split("_")[1]) for f in frames])


class ImageSequenceLoader(_SequenceLoader):
    """Loads the images for a given sequence from disk."""

    def __init__(
        self,
        sequence: str | int,
        data_dir: str = DATA_DIR_NERSEMBLE,
        image_downsampling_factor: int | float = 1,
        cameras: list[int] = TRAIN_CAMS,
    ) -> None:
        """
        Args:
            sequence (str): Sequence name.
            data_dir (str): Directory containing the data.
            image_downsampling_factor (int | float): Downsampling factor for the images,
                masks and intrinsics.
            cameras (list[int]): List of camera IDs to use.
        """
        self.init(sequence, data_dir, image_downsampling_factor, cameras)

    def __getitem__(
        self,
        idx: (
            int
            | slice
            | tuple[int, int]
            | tuple[slice, int]
            | tuple[int, slice]
            | tuple[slice, slice]
            | tuple[int, tuple]
            | tuple[slice, tuple]
        ),
    ) -> (
        Float[torch.Tensor, "time cam H W 3"]
        | Float[torch.Tensor, "time H W 3"]
        | Float[torch.Tensor, "cam H W 3"]
        | Float[torch.Tensor, "H W 3"]
    ):
        """
        Get the image for a given sequence, time step, and camera.

        Args:
            sequence (str): Sequence name.
            idx: Index

        Returns:
            (torch.Tensor): Image tensor. Shape (..., cam, H, W, 3).
        """

        def load_single_image(t: int, c: int) -> Float[torch.Tensor, "H W 3"]:
            filename = os.path.join(
                self.data_dir,
                "sequences",
                self.sequence,
                "timesteps",
                f"frame_{t:05d}",
                "images-2x",
                f"cam_{self.serials[c]}.jpg",
            )
            image = Image.open(filename)
            if self.image_downsampling_factor > 1:
                image = image.resize(
                    (
                        int(image.size[0] // self.image_downsampling_factor),
                        int(image.size[1] // self.image_downsampling_factor),
                    )
                )
            return torch.tensor(np.array(image).astype(np.float32)) / 255.0

        match idx:
            case int(t):
                # Single time step, all cameras
                return torch.stack(
                    [load_single_image(t, c) for c in range(len(self.serials))]
                )

            case slice() as s:
                # Multiple time steps, all cameras
                time_steps = range(*s.indices(len(self)))
                return torch.stack(
                    [
                        torch.stack(
                            [load_single_image(t, c) for c in range(len(self.serials))]
                        )
                        for t in time_steps
                    ]
                )

            case (int(t), int(c)):
                # Single time step, single camera
                return load_single_image(t, c)

            case (slice() as s, int(c)):
                # Multiple time steps, single camera
                time_steps = range(*s.indices(len(self)))
                return torch.stack([load_single_image(t, c) for t in time_steps])

            case (int(t), slice() as s):
                # Single time step, multiple cameras
                camera_ids = range(*s.indices(len(self.serials)))
                return torch.stack([load_single_image(t, c) for c in camera_ids])

            case (slice() as s, slice() as c):
                # Multiple time steps, multiple cameras
                time_steps = range(*s.indices(len(self)))
                camera_ids = range(*c.indices(len(self.serials)))
                return torch.stack(
                    [
                        torch.stack([load_single_image(t, c) for c in camera_ids])
                        for t in time_steps
                    ]
                )

            case (int(t), tuple() as c):
                # Single time step, multiple cameras (selected)
                camera_ids = c
                return torch.stack([load_single_image(t, c) for c in camera_ids])

            case (slice() as s, tuple() as c):
                # Multiple time steps, multiple cameras (selected)
                time_steps = range(*s.indices(len(self)))
                camera_ids = c
                return torch.stack(
                    [
                        torch.stack([load_single_image(t, c) for c in camera_ids])
                        for t in time_steps
                    ]
                )

            case _:
                raise ValueError("Invalid index.")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return len(self), len(self.camera_ids), self.image_height, self.image_width, 3


# ==================================================================================== #
#                                 Masks                                                #
# ==================================================================================== #


class SegmentationMaskSequenceLoader(_SequenceLoader):
    """Loads the segmentation masks for a given sequence from disk."""

    def __init__(
        self,
        sequence: str | int,
        data_dir: str = DATA_DIR_NERSEMBLE,
        image_downsampling_factor: int | float = 1,
        cameras: list[int] = TRAIN_CAMS,
    ) -> None:
        """
        Args:
            sequence (str): Sequence name.
            data_dir (str): Directory containing the data.
            image_downsampling_factor (int | float): Downsampling factor for the images,
                masks and intrinsics.
            cameras (list[int]): List of camera IDs to use.
        """
        self.init(sequence, data_dir, image_downsampling_factor, cameras)

    def __getitem__(
        self,
        idx: (
            int
            | slice
            | tuple[int, int]
            | tuple[slice, int]
            | tuple[int, slice]
            | tuple[slice, slice]
            | tuple[int, tuple]
            | tuple[slice, tuple]
        ),
    ) -> (
        Float[torch.Tensor, "time cam H W"]
        | Float[torch.Tensor, "time H W"]
        | Float[torch.Tensor, "cam H W"]
        | Float[torch.Tensor, "H W"]
    ):
        """
        Get the image for a given sequence, time step, and camera.

        Args:
            sequence (str): Sequence name.
            idx: Index

        Returns:
            (torch.Tensor): Image tensor. Shape (..., cam, H, W).
        """

        def load_single_image(t: int, c: int) -> Float[torch.Tensor, "H W"]:
            filename = os.path.join(
                self.data_dir,
                "sequences",
                self.sequence,
                "timesteps",
                f"frame_{t:05d}",
                "alpha_map",
                f"cam_{self.serials[c]}.png",
            )
            image = Image.open(filename)
            image = image.resize(
                (
                    int(image.size[0] // (self.image_downsampling_factor * 2)),
                    int(image.size[1] // (self.image_downsampling_factor * 2)),
                )
            )
            return torch.tensor(np.array(image).astype(np.float32)) / 255.0

        match idx:
            case int(t):
                # Single time step, all cameras
                return torch.stack(
                    [load_single_image(t, c) for c in range(len(self.serials))]
                )

            case slice() as s:
                # Multiple time steps, all cameras
                time_steps = range(*s.indices(len(self)))
                return torch.stack(
                    [
                        torch.stack(
                            [load_single_image(t, c) for c in range(len(self.serials))]
                        )
                        for t in time_steps
                    ]
                )

            case (int(t), int(c)):
                # Single time step, single camera
                return load_single_image(t, c)

            case (slice() as s, int(c)):
                # Multiple time steps, single camera
                time_steps = range(*s.indices(len(self)))
                return torch.stack([load_single_image(t, c) for t in time_steps])

            case (int(t), slice() as s):
                # Single time step, multiple cameras
                camera_ids = range(*s.indices(len(self.serials)))
                return torch.stack([load_single_image(t, c) for c in camera_ids])

            case (slice() as s, slice() as c):
                # Multiple time steps, multiple cameras
                time_steps = range(*s.indices(len(self)))
                camera_ids = range(*c.indices(len(self.serials)))
                return torch.stack(
                    [
                        torch.stack([load_single_image(t, c) for c in camera_ids])
                        for t in time_steps
                    ]
                )

            case (int(t), tuple() as c):
                # Single time step, multiple cameras (selected)
                camera_ids = c
                return torch.stack([load_single_image(t, c) for c in camera_ids])

            case (slice() as s, tuple() as c):
                # Multiple time steps, multiple cameras (selected)
                time_steps = range(*s.indices(len(self)))
                camera_ids = c
                return torch.stack(
                    [
                        torch.stack([load_single_image(t, c) for c in camera_ids])
                        for t in time_steps
                    ]
                )

            case _:
                raise ValueError("Invalid index.")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return len(self), len(self.camera_ids), self.image_height, self.image_width


# ==================================================================================== #
#                                   Flame Param                                        #
# ==================================================================================== #


class FlameParamsSequenceLoader(_SequenceLoader):
    def __init__(
        self,
        sequence: str | int,
        data_dir: str = DATA_DIR_NERSEMBLE,
        image_downsampling_factor: int | float = 1,
        cameras: list[int] = TRAIN_CAMS,
        cache: bool = True,
    ) -> None:
        """
        Args:
            sequence (str): Sequence name.
            data_dir (str): Directory containing the data.
            image_downsampling_factor (int | float): Downsampling factor for the images,
                masks and intrinsics.
            cameras (list[int]): List of camera IDs to use.
        """
        self.init(sequence, data_dir, image_downsampling_factor, cameras)
        self.filename = os.path.join(
            self.data_dir,
            "sequences",
            self.sequence,
            "annotations",
            "tracking/FLAME2023_v2",
            "tracked_flame_params.npz",
        )
        self.cache = cache
        if cache:
            with np.load(self.filename) as data:
                self.shape = torch.tensor(data["shape"][0], dtype=torch.float32)
                self.expression = torch.tensor(data["expression"], dtype=torch.float32)
                self.neck = torch.tensor(data["neck"], dtype=torch.float32)
                self.jaw = torch.tensor(data["jaw"], dtype=torch.float32)
                self.eyes = torch.tensor(data["eyes"], dtype=torch.float32)
                self.scale = torch.tensor(data["scale"][0], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, time_step: int | slice) -> UnbatchedFlameParams:
        if self.cache:
            shape = self.shape
            expression = self.expression[time_step]
            neck = self.neck[time_step]
            jaw = self.jaw[time_step]
            eyes = self.eyes[time_step]
            scale = self.scale
        else:
            with np.load(self.filename) as data:
                shape = torch.tensor(data["shape"][0], dtype=torch.float32)
                expression = torch.tensor(
                    data["expression"][time_step], dtype=torch.float32
                )
                neck = torch.tensor(data["neck"][time_step], dtype=torch.float32)
                jaw = torch.tensor(data["jaw"][time_step], dtype=torch.float32)
                eyes = torch.tensor(data["eyes"][time_step], dtype=torch.float32)
                scale = torch.tensor(data["scale"][0], dtype=torch.float32)
        if expression.ndim == 2:
            shape = repeat(shape, "f -> time f", time=expression.shape[0])
            scale = repeat(scale, "f -> time f", time=expression.shape[0])

        return UnbatchedFlameParams(shape, expression, neck, jaw, eyes, scale)


# ==================================================================================== #
#                                 SE(3) Transforms                                     #
# ==================================================================================== #


class SE3TransformSequenceLoader(_SequenceLoader):
    def __init__(
        self,
        sequence: str | int,
        data_dir: str = DATA_DIR_NERSEMBLE,
        image_downsampling_factor: int | float = 1,
        cameras: list[int] = TRAIN_CAMS,
        cache: bool = True,
    ) -> None:
        """
        Args:
            sequence (str): Sequence name.
            data_dir (str): Directory containing the data.
            image_downsampling_factor (int | float): Downsampling factor for the images,
                masks and intrinsics.
            cameras (list[int]): List of camera IDs to use.
            cache (bool): Whether to cache the data in memory.
        """
        self.init(sequence, data_dir, image_downsampling_factor, cameras)
        self.filename = os.path.join(
            self.data_dir,
            "sequences",
            self.sequence,
            "annotations",
            "tracking/FLAME2023_v2",
            "tracked_flame_params.npz",
        )
        self.cache = cache
        if cache:
            with np.load(self.filename) as data:
                self.rotation = torch.tensor(
                    data["rotation_matrices"], dtype=torch.float32
                )
                self.translation = torch.tensor(
                    data["translation"], dtype=torch.float32
                )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, time_step: int | slice) -> UnbatchedSE3Transform:

        if self.cache:
            rotation = self.rotation[time_step]
            translation = self.translation[time_step]
        else:
            with np.load(self.filename) as data:
                rotation = torch.tensor(
                    data["rotation_matrices"][time_step], dtype=torch.float32
                )
                translation = torch.tensor(
                    data["translation"][time_step], dtype=torch.float32
                )

        return UnbatchedSE3Transform(rotation, translation)


# ==================================================================================== #
#                                 Audio Features                                       #
# ==================================================================================== #


class AudioFeaturesSequenceLoader(_SequenceLoader):
    def __init__(
        self,
        sequence: str | int,
        data_dir: str = DATA_DIR_NERSEMBLE,
        image_downsampling_factor: int | float = 1,
        cameras: list[int] = TRAIN_CAMS,
        cache: bool = True,
        cleaned: bool = True,
    ) -> None:
        """
        Args:
            sequence (str): Sequence name.
            data_dir (str): Directory containing the data.
            image_downsampling_factor (int | float): Downsampling factor for the images,
                masks and intrinsics.
            cameras (list[int]): List of camera IDs to use.
            cache (bool): Whether to cache the data in memory.
        """
        self.init(sequence, data_dir, image_downsampling_factor, cameras)
        self.filename = os.path.join(
            self.data_dir,
            "sequences",
            self.sequence,
            "audio",
            "audio_features.pt" if not cleaned else "audio_features_cleaned.pt",
        )
        self.cache = cache
        if cache:
            self.audio_features = torch.load(self.filename, weights_only=False)[
                "audio_features"
            ]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(
        self, time_step: int | slice
    ) -> Float[torch.Tensor, "time 1024"] | Float[torch.Tensor, "1024"]:
        if self.cache:
            return self.audio_features[time_step]
        else:
            return torch.load(self.filename, weights_only=False)["audio_features"][
                time_step
            ]
