""" Data management for a single sequence."""

import json
import os

import numpy as np
import open3d as o3
import soundfile as sf
import torch
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Intrinsics, Pose
from einops import repeat
from jaxtyping import Float

from thesis.constants import DATA_DIR_NERSEMBLE, TRAIN_CAMS
from thesis.data_management.data_classes import SingleFrameData
from thesis.data_management.primitives_loader import (
    AlphaMapSequenceLoader,
    AudioFeaturesSequenceLoader,
    FlameParamsSequenceLoader,
    ImageSequenceLoader,
    SE3TransformSequenceLoader,
    SegmentationMaskSequenceLoader,
)

# ==================================================================================== #
#                                 Sequence Manager                                     #
# ==================================================================================== #


class SequenceManager:
    """
    DataManager for the NeRSemble dataset.
    Inspired by https://github.com/tobias-kirschstein/nersemble
    """

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
        if isinstance(sequence, int) and data_dir == DATA_DIR_NERSEMBLE:
            assert sequence in range(3, 102), "Invalid sequence number."
            sequence = f"sequence_{sequence:04d}"
        elif isinstance(sequence, int):
            sequence = f"sequence_{sequence:04d}"
        self.id = int(sequence.split("_")[-1])
        # NOTE: not sure if using class variables actually helps with multi threading,
        #       but just to be safe, I will use them here.
        self.sequence = sequence
        self.image_downsampling_factor = image_downsampling_factor
        self.data_dir = data_dir
        camera_config = json.load(open(os.path.join(data_dir, "calibration/config.json")))
        self.image_width = int(camera_config["image_size"][0] // (2*image_downsampling_factor))
        self.image_height = int(camera_config["image_size"][1] // (2*image_downsampling_factor))
        self.camera_params = json.load(
            open(os.path.join(data_dir, "calibration/camera_params.json")))

        self.serials = [list(self.camera_params["world_2_cam"].keys())[c] for c in cameras]
        self.camera_ids = np.array(cameras)

        # get the time steps
        frames_dir = os.path.join(data_dir, "sequences", sequence, "timesteps")
        frames = os.listdir(frames_dir)
        self.frames = sorted([int(f.split("_")[1]) for f in frames])

        # get the primitive loaders
        self.images = ImageSequenceLoader(
            sequence=sequence,
            data_dir=data_dir,
            image_downsampling_factor=image_downsampling_factor,
            cameras=cameras)
        self.alpha_maps = AlphaMapSequenceLoader(
            sequence=sequence,
            data_dir=data_dir,
            image_downsampling_factor=image_downsampling_factor,
            cameras=cameras)
        self.segmentation_masks = SegmentationMaskSequenceLoader(
            sequence=sequence,
            data_dir=data_dir,
            image_downsampling_factor=image_downsampling_factor,
            cameras=cameras)
        self.flame_params = FlameParamsSequenceLoader(
            sequence=sequence,
            data_dir=data_dir,
            image_downsampling_factor=image_downsampling_factor,
            cameras=cameras)
        self.se3_transforms = SE3TransformSequenceLoader(sequence, data_dir,
                                                         image_downsampling_factor, cameras)
        self.audio_features = AudioFeaturesSequenceLoader(
            sequence,
            data_dir,
            image_downsampling_factor,
            cameras,
            cleaned=False,  # DO NOT CHANGE!
        )

        # Compute the jaw norms and save the resulting sorted indices
        jaw_norms = torch.norm(self.flame_params[:].jaw, dim=-1)
        self.jaw_norms_sorted_indices = torch.argsort(jaw_norms, descending=True)
        self.jaw_sample_probs = jaw_norms / jaw_norms.sum()

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def cameras(
            self) -> tuple[Float[torch.Tensor, "3 3"], Float[torch.Tensor, "cam 4 4"], list[str]]:
        # auto-caching of the cameras
        if not hasattr(self, "_cameras"):
            self._cameras = self.load_camera(slice(0, None, None))
        return self._cameras

    @property
    def color_correction(self,) -> Float[torch.Tensor, "cam 3 3"] | Float[torch.Tensor, "3 3"]:
        # auto-caching of the color correction
        if not hasattr(self, "_color_correction"):
            self._color_correction = self.load_color_correction(slice(0, None, None))
        return self._color_correction

    @property
    def n_cameras(self) -> int:
        """Number of cameras in the sequence."""
        return len(self.serials)

    @property
    def n_time_steps(self) -> int:
        """Number of time steps in the sequence."""
        return len(self.frames)

    # ================================================================================ #

    def load_camera(
        self,
        camera_id: int | slice,
    ) -> tuple[Float[torch.Tensor, "3 3"], Float[torch.Tensor, "cam 4 4"], list[str]]:
        """
        Return the camera for a given camera ID.

        Args:
            camera_id (int | slice): Camera ID.

        Returns:
            intrinsics, world_2_cam, serials
        """
        # Calculate intrinsics
        intrinsics = Intrinsics(self.camera_params["intrinsics"])
        intrinsics = intrinsics.rescale(0.5 * (2**-(self.image_downsampling_factor - 1)))
        intrinsics = np.array(intrinsics).astype(np.float32)
        intrinsics = torch.tensor(intrinsics)

        # Calculate World2Cam
        if isinstance(camera_id, slice):
            serials = self.serials[camera_id]
        else:
            serials = [self.serials[camera_id]]
        world_2_cam = []
        for s in serials:
            e = self.camera_params["world_2_cam"][s]
            e = Pose(
                e,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                pose_type=PoseType.WORLD_2_CAM,
            )
            e = np.array(e).astype(np.float32)
            world_2_cam.append(e)
        world_2_cam = np.stack(world_2_cam, axis=0)
        world_2_cam = torch.tensor(world_2_cam)

        # Get the serials
        if isinstance(camera_id, int):
            serials = [self.serials[camera_id]]
        else:
            serials = self.serials[camera_id]

        return intrinsics, world_2_cam, serials

    def load_point_cloud(
            self, time_step: int) -> tuple[Float[torch.Tensor, "N 3"], Float[torch.Tensor, "N 3"]]:
        """
        Get the point cloud for a given sequence and time step.

        Args:
            time_step (int): Time step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple with the following elements:
                - points (torch.Tensor): Point cloud tensor. Shape (N, 3).
                - colors (torch.Tensor): Color tensor. Shape (N, 3).

        """
        if False:
            filename = os.path.join(
                self.data_dir,
                "sequences",
                self.sequence,
                "timesteps",
                f"frame_{time_step * 3:05d}",  # NOTE: they only have every third frame
                "colmap/pointclouds",
                "pointcloud_16.pcd",
            )
        else:
            filename = "data/nersemble/pointcloud_16.pcd"  # TODO: remove hardcoding

        point_cloud = o3.io.read_point_cloud(
            filename,
            remove_nan_points=True,
            remove_infinite_points=True,
        )
        points = np.array(point_cloud.points).astype(np.float32)
        points = torch.tensor(points)
        colors = np.array(point_cloud.colors).astype(np.float32)
        colors = torch.tensor(colors)

        return points, colors

    def load_audio(self) -> tuple[np.ndarray, int]:
        """
        Get the audio for the sequence.

        Returns:
            (np.ndarray): Audio tensor.
        """
        filename = os.path.join(
            self.data_dir,
            "sequences",
            self.sequence,
            "audio",
            "audio_recording_cleaned.ogg",
        )
        return sf.read(filename)

    def load_color_correction(
            self,
            camera_id: int | slice) -> Float[torch.Tensor, "cam 3 3"] | Float[torch.Tensor, "3 3"]:
        """
        Get the color correction matrix for a given camera.

        Args:
            camera_id (int): Camera ID.

        Returns:
            (torch.Tensor): Color correction matrix. Shape (cam, 3, 3).
        """

        def load_single_color_correction(c: int) -> Float[torch.Tensor, "3 3"]:
            filename = os.path.join(
                self.data_dir,
                "sequences",
                self.sequence,
                "annotations",
                "color_correction",
                f"{self.serials[c]}.npy",
            )
            # we don't need the alpha channel in the color correction matrix
            return torch.tensor(np.load(filename).astype(np.float32)[:3, :3])

        match camera_id:
            case int(c):
                return load_single_color_correction(c)
            case slice() as s:
                return torch.stack(
                    [load_single_color_correction(c) for c in range(*s.indices(self.n_cameras))])

    def sample_n_cams(self, n_cams: int) -> list[int]:
        """
        Sample n cameras from the sequence. We could make it smarter by having a
        stack but random sampling should be fine for now.

        Args:
            n_cams (int): Number of cameras to sample.

        Returns:
            (list[int]): List of camera IDs.
        """
        return np.random.permutation(self.n_cameras)[:n_cams].tolist()

    def get_single_frame(self,
                         idx: int,
                         n_cams: int | None = None,
                         window_size: int = 1) -> SingleFrameData:
        """
        Get a single frame for gaussian splatting.

        Args:
            idx (int): Index of the frame.
            n_cams (int | None): Number of cameras to use. If None, all cameras are
                used. This is a bit of a hack to allow for batching across multiple
                dimensions.
        """
        if n_cams is not None:
            cameras = tuple(self.sample_n_cams(n_cams))
        else:
            cameras = tuple(range(self.n_cameras))
        intrinsics, e, _ = self.cameras
        image = self.images[idx, cameras].squeeze(0)  # remove time dimension
        alpha_map = self.alpha_maps[idx, cameras].squeeze(0)
        segmentation_mask = self.segmentation_masks[idx, cameras].squeeze(0)
        world_2_cam = e[cameras, :]
        if len(cameras) == 1:
            image = image.unsqueeze(0)
            segmentation_mask = segmentation_mask.unsqueeze(0)
            alpha_map = alpha_map.unsqueeze(0)
        intrinsics = repeat(intrinsics, "m n -> cam m n", cam=len(cameras))

        return SingleFrameData(
            image=image,
            alpha_map=alpha_map,
            segmentation_mask=segmentation_mask,
            intrinsics=intrinsics,
            world_2_cam=world_2_cam,
            color_correction=self.color_correction,
            se3_transform=self.se3_transforms[idx],
            sequence_id=torch.tensor(self.id),
            time_step=torch.tensor(idx),
            camera_indices=torch.tensor(cameras),
        )


# ==================================================================================== #
#                                Multi Sequence Manager                                #
# ==================================================================================== #


class MultiSequenceManager:
    """
    DataManager for multiple sequences.
    """

    def __init__(
        self,
        sequences: list[int],
        data_dir: str = DATA_DIR_NERSEMBLE,
        image_downsampling_factor: int | float = 1,
        cameras: list[int] = TRAIN_CAMS,
        window_size: int = 1,
    ) -> None:
        """
        Args:
            sequences (list[int]): List of sequence numbers.
            data_dir (str): Directory containing the data.
            image_downsampling_factor (int | float): Downsampling factor for the images,
                masks and intrinsics.
            cameras (list[int]): List of camera IDs to use.
            window_length (int): Length of the window. Used to compute the length of
                each of the sequences.
        """

        self.sequences = sequences
        self.cameras = cameras

        self.start_indices = []
        self.end_indices = []
        self.lengths = []
        self.sequence_managers = []
        cnt = 0
        self.window_size = window_size

        for sequence in sequences:
            sm = SequenceManager(
                sequence=sequence,
                data_dir=data_dir,
                image_downsampling_factor=image_downsampling_factor,
                cameras=cameras,
            )
            self.sequence_managers.append(sm)
            self.start_indices.append(cnt)
            cnt += len(sm) - window_size + 1
            self.end_indices.append(cnt)
            # NOTE: this allows for returning entire sequences, by setting window_size to something
            #       larger than the sequence length.
            self.lengths.append(len(sm) - window_size + 1)

        self.image_width = self.sequence_managers[0].image_width
        self.image_height = self.sequence_managers[0].image_height
        self.n_cameras = len(cameras)

    def __len__(self) -> int:
        """Returns total number of frames in all sequences."""
        return sum(self.lengths)

    def global_index_to_sequence_idx(self, idx: int) -> tuple[int, int]:
        """
        Args:
            idx (int): Global index.
        Returns:
            (int, int): Sequence index, frame index.
        """

        left, right = 0, len(self.start_indices) - 1
        while left <= right:
            mid = (left+right) // 2
            if self.start_indices[mid] <= idx < self.end_indices[mid]:
                return mid, idx - self.start_indices[mid]
            elif idx < self.start_indices[mid]:
                right = mid - 1
            else:
                left = mid + 1

        raise ValueError(f"Index {idx} out of bounds.")

    def __getitem__(self, idx: int) -> SequenceManager:
        assert 0 <= idx < len(self), "Index out of bounds."
        return self.sequence_managers[idx]
