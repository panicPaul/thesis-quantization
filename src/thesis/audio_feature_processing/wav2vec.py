""" Bert tokenizer for audio processing. """

import os
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from jaxtyping import Float
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2BertModel

from thesis.constants import DATA_DIR_NERSEMBLE


def get_processor_and_model():  # -> tuple[AutoProcessor, Wav2Vec2BertModel]:
    """
    Helper function to get the processor and the model for the Wav2Vec2 model.

    Returns:
        tuple[AutoProcessor, Wav2Vec2BertModel]: The preprocessor and the model.
    """
    processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
    model = Wav2Vec2BertModel.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")

    return processor, model  # type: ignore


def process_sequence_concatenation(
    audio: Float[np.ndarray, "time"],
    sampling_rate: int,
    n_frames: int,
    processor: AutoProcessor,
    model: Wav2Vec2BertModel,
) -> Float[torch.Tensor, "time 3075"]:
    """
    Process the audio sequence with the Wav2Vec2 model. Because the hidden state
    sequence does not align with the frames, we need to align them. To that end we
    return the three closest hidden states to each frame. The last three hidden states
    are how far the hidden states are from the frame.

    Args:
        audio (Float[np.ndarray, "time"]): The audio sequence.
        sampling_rate (int): The sampling rate of the audio.
        n_frames (int): The number of video frames in the sequence.
        processor (AutoProcessor): The processor for the Wav2Vec2 model.
        model (Wav2Vec2BertModel): The Wav2Vec2 model.

    Returns:
        Float[torch.Tensor, "time 3075"]: The processed sequence.
    """

    # Process the audio
    inputs = processor(
        audio, sampling_rate=sampling_rate, return_tensors="pt")  # type: ignore # noqa

    # Get the last hidden states
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state.squeeze(0)

    # Calculate the frame indices, as well as their previous and next frames
    n_samples = last_hidden_states.shape[0]
    padded_hidden_states = torch.cat([
        last_hidden_states[0].unsqueeze(0),
        last_hidden_states,
        last_hidden_states[-1].unsqueeze(0),
    ])
    frame_indices = torch.linspace(0, n_samples - 1, n_frames)
    closest_frame = frame_indices.long() + 1
    indices = closest_frame.unsqueeze(0) + torch.arange(-1, 2).unsqueeze(1)
    indices = indices.permute(1, 0)
    indices = indices.clamp(0, n_samples + 1)

    # Get the hidden states
    hidden_states = padded_hidden_states[indices]
    hidden_states = hidden_states.permute(0, 2, 1).flatten(1)

    # Get the index distance to the closest three frames
    distance = indices - 1 - frame_indices.unsqueeze(1)

    # Concatenate the hidden states with the distance
    hidden_states = torch.cat([hidden_states, distance], dim=1)

    return hidden_states


def process_sequence_interpolation(
    audio: Float[np.ndarray, "time"],
    sampling_rate: int,
    n_frames: int,
    processor,  #: AutoProcessor,
    model,  #: Wav2Vec2BertModel,
    device: torch.device | str = "cuda",
) -> Float[torch.Tensor, "time 1024"]:
    """
    Process the audio sequence with the Wav2Vec2 model. Because the hidden state
    sequence does not align with the frames, we need to align them. To that end we
    return the interpolation between the two closest hidden states to each frame.

    Args:
        audio (Float[np.ndarray, "time"]): The audio sequence.
        sampling_rate (int): The sampling rate of the audio.
        n_frames (int): The number of video frames in the sequence.
        processor (AutoProcessor): The processor for the Wav2Vec2 model.
        model (Wav2Vec2BertModel): The Wav2Vec2 model.
        device (torch.device | str): The device to use for processing.

    Returns:
        Float[torch.Tensor, "time 1024"]: The processed sequence.
    """

    # Process the audio
    inputs = processor(
        audio, sampling_rate=sampling_rate, return_tensors="pt")  # type: ignore # noqa
    inputs = inputs.to(device)

    # Get the last hidden states
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state.squeeze(0)  # has shape (audio_steps, 1024)

    # Calculate the ratio between hidden states and frames
    hidden_state_count = hidden_states.shape[0]
    ratio = hidden_state_count / n_frames

    # Initialize the output tensor
    interpolated_states = torch.zeros((n_frames, hidden_states.shape[1]), device=device)

    # Perform linear interpolation
    for i in range(n_frames):
        frame_pos = i * ratio
        lower_idx = int(frame_pos)
        upper_idx = min(lower_idx + 1, hidden_state_count - 1)

        interpolation_factor = frame_pos - lower_idx

        lower_state = hidden_states[lower_idx]
        upper_state = hidden_states[upper_idx]

        interpolated_state = (
            1-interpolation_factor) * lower_state + interpolation_factor*upper_state
        interpolated_states[i] = interpolated_state

    return interpolated_states


def save_audio_features(
    data_directory: str = DATA_DIR_NERSEMBLE,
    sequence_names: list[str] | str = [f"sequence_{i:04d}" for i in range(3, 102)],
    type: Literal["concatenation", "interpolation"] = "interpolation",
    cleaned: bool = True,
    device: torch.device | str = "cuda",
) -> None:
    """
    Process the audio features for the dataset and save them.

    Args:
        data_manager (DataManager): The data manager for the dataset.
        type (Literal["concatenation", "interpolation"]): The type of processing to
            use.
    """

    # Set up everything
    processor, model = get_processor_and_model()
    model = model.to(device)  # type: ignore

    # Process the audio features
    for sequence_name in tqdm(sequence_names):
        audio_path = ((f"{data_directory}/sequences/{sequence_name}/audio/"
                       "audio_recording_cleaned.ogg") if cleaned else
                      (f"{data_directory}/sequences/{sequence_name}/audio/"
                       "audio_recording_resampled.ogg"))

        audio, sampling_rate = sf.read(audio_path)
        frame_dir = f"{data_directory}/sequences/{sequence_name}/timesteps"
        n_frames = len(os.listdir(frame_dir))
        match type:
            case "concatenation":
                hidden_states = process_sequence_concatenation(
                    audio=audio,
                    sampling_rate=sampling_rate,
                    n_frames=n_frames,
                    processor=processor,
                    model=model,
                )
            case "interpolation":
                hidden_states = process_sequence_interpolation(
                    audio=audio,
                    sampling_rate=sampling_rate,
                    n_frames=n_frames,
                    processor=processor,
                    model=model,
                    device=device,
                )

        # Save the hidden states
        audio_directory = os.path.dirname(audio_path)
        output_path = (
            os.path.join(audio_directory, "audio_features_cleaned.pt")
            if cleaned else os.path.join(audio_directory, "audio_features.pt"))
        checkpoint = {"audio_features": hidden_states.to("cpu")}
        torch.save(checkpoint, output_path)


#  ======================= Main =======================

if __name__ == "__main__":
    save_audio_features(cleaned=False)
