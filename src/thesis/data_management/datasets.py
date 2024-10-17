""" Single sequence datasets. """

import torch
from jaxtyping import Float
from torch.utils.data import Dataset

import thesis.data_management.data_classes as dc
from thesis.constants import DATA_DIR_NERSEMBLE, TRAIN_CAMS
from thesis.data_management.sequence_manager import (
    MultiSequenceManager,
    SequenceManager,
)

# ==================================================================================== #
#                              Single Sequence Dataset                                 #
# ==================================================================================== #


class SingleSequenceDataset(Dataset):
    """Dataset for single sequence data."""

    def __init__(
        self,
        sequence: int | str,
        cameras: list[int] = TRAIN_CAMS,
        image_downsampling_factor: int | float = 1,
        start_idx: int = 0,
        end_idx: int | None = None,
        n_cameras_per_frame: int | None = None,
        data_dir: str = DATA_DIR_NERSEMBLE,
    ) -> None:
        """
        Args:
            sequence (int | str): Sequence number.
            cameras (list[int]): List of camera IDs to use.
            image_downsampling_factor (int | float): Downsampling factor for the images,
                masks and intrinsics.
            start_idx (int): Start index of the sequence.
            end_idx (int | None): End index of the sequence. If None, the sequence
                length is used.
            n_cameras_per_frame (int | None): Number of cameras per frame. If None,
                all cameras are used. Otherwise they are randomly sampled.
            data_dir (str): Directory containing the data.
        """

        self.sequence_manager = SequenceManager(
            sequence=sequence,
            cameras=cameras,
            image_downsampling_factor=image_downsampling_factor,
            data_dir=data_dir,
        )
        self.n_cameras_per_frame = n_cameras_per_frame
        self.start_idx = start_idx
        self.end_idx = len(self.sequence_manager) if end_idx is None else end_idx
        assert self.start_idx < self.end_idx, "Start index must be less than end index."

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int) -> dc.SingleFrameData:
        idx = idx + self.start_idx
        return self.sequence_manager.get_single_frame(idx, self.n_cameras_per_frame)

    def prepare_data(
        self, batch: dc.SingleFrameData, device: torch.device | str = "cuda"
    ) -> dc.SingleFrameData:
        se3_transform = dc.UnbatchedSE3Transform(
            rotation=batch.se3_transform.rotation.to(device),
            translation=batch.se3_transform.translation.to(device),
        )
        return dc.SingleFrameData(
            image=batch.image.to(device),
            mask=batch.mask.to(device),
            intrinsics=batch.intrinsics.to(device),
            extrinsics=batch.extrinsics.to(device),
            color_correction=batch.color_correction.to(device),
            se3_transform=se3_transform,
            sequence_id=batch.sequence_id.to(device),
            time_step=batch.time_step.to(device),
        )


# ==================================================================================== #
#                              Multi-sequence Dataset                                  #
# ==================================================================================== #


class MultiSequenceDataset(Dataset):
    """Dataset for multiple sequences."""

    def __init__(
        self,
        sequences: list[int | str],
        cameras: list[int] = TRAIN_CAMS,
        image_downsampling_factor: int | float = 1,
        n_cameras_per_frame: int | None = None,
        data_dir: str = DATA_DIR_NERSEMBLE,
    ) -> None:
        """
        Args:
            sequences (list[int | str]): List of sequence numbers.
            cameras (list[int]): List of camera IDs to use.
            image_downsampling_factor (int | float): Downsampling factor for the images,
                masks and intrinsics.
            n_cameras_per_frame (int | None): Number of cameras per frame. If None,
                all cameras are used. Otherwise they are randomly sampled.
            data_dir (str): Directory containing the data.
        """

        self.msm = MultiSequenceManager(
            sequences=sequences,
            cameras=cameras,
            image_downsampling_factor=image_downsampling_factor,
            data_dir=data_dir,
        )
        self.n_cameras_per_frame = n_cameras_per_frame

    def __len__(self) -> int:
        return len(self.msm)

    def __getitem__(self, idx: int) -> dc.SingleFrameData:
        sequence_idx, frame_idx = self.msm.global_index_to_sequence_idx(idx)
        return self.msm[sequence_idx].get_single_frame(
            frame_idx, self.n_cameras_per_frame
        )

    def prepare_data(
        self, batch: dc.SingleFrameData, device: torch.device | str = "cuda"
    ) -> dc.SingleFrameData:
        se3_transform = dc.UnbatchedSE3Transform(
            rotation=batch.se3_transform.rotation.to(device),
            translation=batch.se3_transform.translation.to(device),
        )
        return dc.SingleFrameData(
            image=batch.image.to(device),
            mask=batch.mask.to(device),
            intrinsics=batch.intrinsics.to(device),
            extrinsics=batch.extrinsics.to(device),
            color_correction=batch.color_correction.to(device),
            se3_transform=se3_transform,
            sequence_id=batch.sequence_id,
            time_step=batch.time_step,
        )


# ==================================================================================== #
#                              Quantization Dataset                                    #
# ==================================================================================== #


class QuantizationDataset(Dataset):
    """Dataset for quantization training."""

    def __init__(
        self,
        sequences: list[int | str],
        data_dir: str = DATA_DIR_NERSEMBLE,
        window_size: int = 16,
    ) -> None:
        """
        Args:
            sequences (list[int | str]): List of sequence numbers..
            data_dir (str): Directory containing the data.
        """

        self.msm = MultiSequenceManager(
            sequences=sequences,
            data_dir=data_dir,
            window_size=window_size,
        )
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.msm)

    def __getitem__(self, idx: int):
        sequence_idx, frame_idx = self.msm.global_index_to_sequence_idx(idx)
        flame_params = tuple(
            self.msm[sequence_idx].flame_params[
                frame_idx : frame_idx + self.window_size
            ]
        )
        se3_transforms = tuple(
            self.msm[sequence_idx].se3_transforms[
                frame_idx : frame_idx + self.window_size
            ]
        )
        audio_features = self.msm[sequence_idx].audio_features[
            frame_idx : frame_idx + self.window_size
        ]
        return flame_params, se3_transforms, audio_features

    @classmethod
    def prepare_data(
        cls, batch, device: torch.device | str = "cuda"
    ) -> tuple[dc.FlameParams, dc.SE3Transform, Float[torch.Tensor, "batch time 1024"]]:
        flame_params, se3_transforms, audio_features = batch
        flame_params = dc.FlameParams(*[fp.to(device) for fp in flame_params])
        se3_transforms = dc.SE3Transform(*[st.to(device) for st in se3_transforms])
        audio_features = audio_features.to(device)
        return flame_params, se3_transforms, audio_features
