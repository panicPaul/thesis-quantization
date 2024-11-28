""" Single sequence datasets. """

import numpy as np
import torch
from jaxtyping import Float
from torch.utils.data import Dataset

import thesis.data_management.data_classes as dc
from thesis.constants import DATA_DIR_NERSEMBLE, TRAIN_CAMS, TRAIN_SEQUENCES
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
        length_multiplier: int = 1,
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
            length_multiplier (int): hacky way to increase the length of the dataset. This is
                useful for preventing excessive re-loading of the data.
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
        self.length_multiplier = length_multiplier
        assert self.start_idx < self.end_idx, "Start index must be less than end index."

    def __len__(self) -> int:
        return (self.end_idx - self.start_idx) * self.length_multiplier

    def __getitem__(self, idx: int) -> dc.SingleFrameData:
        idx = idx % (self.end_idx - self.start_idx)
        idx += self.start_idx
        return self.sequence_manager.get_single_frame(idx, self.n_cameras_per_frame)


# ==================================================================================== #
#                         Sequential Multi-sequence Dataset                            #
# ==================================================================================== #
class SequentialMultiSequenceDataset(Dataset):
    """Dataset for multi sequence data."""

    def __init__(
        self,
        sequences: list[int] = TRAIN_SEQUENCES,
        cameras: list[int] = TRAIN_CAMS,
        image_downsampling_factor: int | float = 1,
        n_cameras_per_frame: int | None = None,
        data_dir: str = DATA_DIR_NERSEMBLE,
    ) -> None:
        """
        Args:
            sequences (list[int | str]): Sequence number.
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
        return self.msm[sequence_idx].get_single_frame(frame_idx, self.n_cameras_per_frame)


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
        window_size: int = 1,
        over_sample_open_jaw: bool = False,
        over_sample_probability: float = 0.5,
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
            window_size (int): Size of the window around the frame.
            over_sample_open_jaw (bool): Whether to over-sample frames with open jaw.
            over_sample_probability (float): Probability of over-sampling frames with open jaw.
        """

        self.msm = MultiSequenceManager(
            sequences=sequences,
            cameras=cameras,
            image_downsampling_factor=image_downsampling_factor,
            data_dir=data_dir,
        )
        self.n_cameras_per_frame = n_cameras_per_frame
        self.window_size = window_size
        self.over_sample_open_jaw = over_sample_open_jaw
        self.over_sample_probability = over_sample_probability

    def __len__(self) -> int:
        return len(self.msm)

    def _over_sample(self, sequence_idx: int, frame_idx: int) -> int:
        """
        Over-sample frames with open jaw.

        Args:
            sequence_idx (int): Sequence index.
            frame_idx (int): Frame index.

        Returns:
            int: New frame index drawn from the distribution induced by the jaw pose.
        """
        probs = self.msm[sequence_idx].jaw_sample_probs.numpy()
        sample = np.random.choice(np.arange(len(probs)), p=probs)
        return int(sample)

    def __getitem__(
        self, idx: int
    ) -> tuple[dc.SingleFrameData, dc.UnbatchedFlameParams, Float[torch.Tensor, "time 1024"]]:
        """ Get a single frame and flame and audio prior information. """
        sequence_idx, frame_idx = self.msm.global_index_to_sequence_idx(idx)
        if self.over_sample_open_jaw and torch.rand(1).item() < self.over_sample_probability:
            frame_idx = self._over_sample(sequence_idx, frame_idx)
        single_frame = self.msm[sequence_idx].get_single_frame(frame_idx, self.n_cameras_per_frame)
        window_indices = slice(
            max(frame_idx - self.window_size // 2, 0),
            min(frame_idx + self.window_size // 2 + 1, len(self.msm[sequence_idx])))
        flame_params = self.msm[sequence_idx].flame_params[window_indices]
        audio_features = self.msm[sequence_idx].audio_features[window_indices]
        if frame_idx - self.window_size // 2 < 0:
            # left padding
            padding_size = self.window_size // 2 - frame_idx
            shape_padding = flame_params.shape[0].unsqueeze(0).repeat(padding_size, 1)
            expr_padding = flame_params.expr[0].unsqueeze(0).repeat(padding_size, 1)
            neck_padding = flame_params.neck[0].unsqueeze(0).repeat(padding_size, 1)
            jaw_padding = flame_params.jaw[0].unsqueeze(0).repeat(padding_size, 1)
            eye_padding = flame_params.eye[0].unsqueeze(0).repeat(padding_size, 1)
            scale_padding = flame_params.scale[0].unsqueeze(0).repeat(padding_size, 1)
            flame_params = dc.UnbatchedFlameParams(
                shape=torch.cat([shape_padding, flame_params.shape]),
                expr=torch.cat([expr_padding, flame_params.expr]),
                neck=torch.cat([neck_padding, flame_params.neck]),
                jaw=torch.cat([jaw_padding, flame_params.jaw]),
                eye=torch.cat([eye_padding, flame_params.eye]),
                scale=torch.cat([scale_padding, flame_params.scale]),
            )
            audio_padding = audio_features[0].unsqueeze(0).repeat(padding_size, 1)
            audio_features = torch.cat([audio_padding, audio_features])
        if frame_idx + self.window_size // 2 + 1 >= len(self.msm[sequence_idx]):
            # right padding
            padding_size = frame_idx + self.window_size // 2 + 1 - len(self.msm[sequence_idx])
            shape_padding = flame_params.shape[-1].unsqueeze(0).repeat(padding_size, 1)
            expr_padding = flame_params.expr[-1].unsqueeze(0).repeat(padding_size, 1)
            neck_padding = flame_params.neck[-1].unsqueeze(0).repeat(padding_size, 1)
            jaw_padding = flame_params.jaw[-1].unsqueeze(0).repeat(padding_size, 1)
            eye_padding = flame_params.eye[-1].unsqueeze(0).repeat(padding_size, 1)
            scale_padding = flame_params.scale[-1].unsqueeze(0).repeat(padding_size, 1)
            flame_params = dc.UnbatchedFlameParams(
                shape=torch.cat([flame_params.shape, shape_padding]),
                expr=torch.cat([flame_params.expr, expr_padding]),
                neck=torch.cat([flame_params.neck, neck_padding]),
                jaw=torch.cat([flame_params.jaw, jaw_padding]),
                eye=torch.cat([flame_params.eye, eye_padding]),
                scale=torch.cat([flame_params.scale, scale_padding]),
            )
            audio_padding = audio_features[-1].unsqueeze(0).repeat(padding_size, 1)
            audio_features = torch.cat([audio_features, audio_padding])

        return single_frame, flame_params, audio_features


# ==================================================================================== #
#                              Quantization Dataset                                    #
# ==================================================================================== #


class QuantizationDataset(Dataset):
    """Dataset for quantization training."""

    def __init__(
        self,
        sequences: list[int | str],
        data_dir: str = DATA_DIR_NERSEMBLE,
        window_size: int | None = 16,
    ) -> None:
        """
        Args:
            sequences (list[int | str]): List of sequence numbers..
            data_dir (str): Directory containing the data.
        """

        self.msm = MultiSequenceManager(
            sequences=sequences,
            data_dir=data_dir,
            window_size=window_size if window_size is not None else 1,
        )
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.msm) if self.window_size is not None else len(self.msm.sequences)

    def __getitem__(self, idx: int):
        if self.window_size is None:
            sequence_idx = idx
            flame_params = tuple(self.msm[sequence_idx].flame_params[:])
            se3_transforms = tuple(self.msm[sequence_idx].se3_transforms[:])
            audio_features = self.msm[sequence_idx].audio_features[:]
            return flame_params, se3_transforms, audio_features
        else:
            sequence_idx, frame_idx = self.msm.global_index_to_sequence_idx(idx)
            flame_params = tuple(self.msm[sequence_idx].flame_params[frame_idx:frame_idx
                                                                     + self.window_size])
            se3_transforms = tuple(self.msm[sequence_idx].se3_transforms[frame_idx:frame_idx
                                                                         + self.window_size])
            audio_features = self.msm[sequence_idx].audio_features[frame_idx:frame_idx
                                                                   + self.window_size]
        return flame_params, se3_transforms, audio_features

    @classmethod
    def prepare_data(
        cls,
        batch,
        device: torch.device | str = "cuda"
    ) -> tuple[dc.FlameParams, dc.SE3Transform, Float[torch.Tensor, "batch time 1024"]]:
        flame_params, se3_transforms, audio_features = batch
        flame_params = dc.FlameParams(*[fp.to(device) for fp in flame_params])
        se3_transforms = dc.SE3Transform(*[st.to(device) for st in se3_transforms])
        audio_features = audio_features.to(device)
        return flame_params, se3_transforms, audio_features


class AudioQuantizationDataset(Dataset):

    def __init__(self, mode="train", window_size: int = 16) -> None:
        """
        Dataset to train on my laptop.

        Args:
            mode (str, optional): "train" or "test". Defaults to "train".
            window_size (int): Defaults to 16.
        """
        data = torch.load("audio_features.pt", weights_only=False)
        if mode == "train":
            self.data = data[:80]
        else:
            self.data = data[80:]
        cnt = 0
        self.start_indices = []
        self.end_indices = []
        for i in range(len(self.data)):
            self.start_indices.append(cnt)
            cnt += self.data[i].shape[0] - window_size + 1
            self.end_indices.append(cnt)

    def __len__(self) -> int:
        return sum(self.end_indices) - sum(self.start_indices)

    def __getitem__(self, idx):
        # binary search
        i = 0
        j = len(self.start_indices)
        while i < j:
            mid = (i+j) // 2
            if idx >= self.end_indices[mid]:
                i = mid + 1
            else:
                j = mid
        idx -= self.start_indices[i]
        return self.data[i][idx:idx + 16]
