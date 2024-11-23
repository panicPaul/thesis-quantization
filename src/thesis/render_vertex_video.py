""" Render the vertex positions. """

import os

import cv2
import torch
from jaxtyping import Float, Int
from tqdm import tqdm

from thesis.code_talker.stage1_runner import Stage1Runner
from thesis.code_talker.stage2_runner import Stage2Runner
from thesis.data_management import SequenceManager, UnbatchedFlameParams
from thesis.flame import FlameHead, FlameHeadWithInnerMouth
from thesis.video_utils import add_audio, render_mesh_image


def render_vertex_video(
    vertices: Float[torch.Tensor, 'time num_vertices 3'],
    faces: Int[torch.Tensor, 'num_faces 3'],
    output_path: str,
    audio_path: str | None = None,
    fps: int = 30,
    quick_time_compatible: bool = False,
) -> None:
    """
    Args:
        vertices (torch.Tensor): The vertices of the mesh at each time step. Has shape
            (time, num_vertices, 3).
        faces (torch.Tensor): The faces of the mesh. Has shape (num_faces, 3).
        output_path (str): The path to save the video to.
        audio_path (str | None): The path to an audio file to add to the video.
        fps (int): The frames per second of the video.
        quick_time_compatible (bool): Whether to make the video QuickTime compatible.
    """

    # Generate the images
    images = []
    for time_step in tqdm(range(vertices.shape[0]), 'Rendering video'):
        image = render_mesh_image(vertices[time_step], faces)
        images.append(image)

    # Save the images as a video
    width, height, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        filename=str(output_path),
        fourcc=fourcc,
        fps=int(fps),
        frameSize=(int(width), int(height)))
    for frame in tqdm(images, desc="Saving video"):
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    if audio_path is not None:
        add_audio(output_path, audio_path, fps, quicktime_compatible=quick_time_compatible)


def render_vertex_reconstruction(
    checkpoint_path: str,
    sequence: int,
    output_path: str,
    quick_time_compatible: bool = False,
) -> None:
    """
    Evaluation of the VQ-VAE model on a sequence.

    Args:
        checkpoint_path (str): The path to the checkpoint of the VQ-VAE model.
        sequence (int): The sequence to evaluate the model on.
        output_path (str): The path to save the video to.
        quick_time_compatible (bool): Whether to make the video QuickTime compatible.
    """

    runner = Stage1Runner.load_from_checkpoint(checkpoint_path)
    runner.eval()
    runner.cuda()
    flame_head = FlameHeadWithInnerMouth()
    flame_head.cuda()
    disable_neck = runner.config.disable_neck

    # Load the sequence
    sm = SequenceManager(sequence=sequence)
    flame_params = sm.flame_params[:]
    flame_params = UnbatchedFlameParams(
        shape=flame_params.shape.cuda(),
        expr=flame_params.expr.cuda(),
        neck=flame_params.neck.cuda()
        if not disable_neck else torch.zeros_like(flame_params.neck).cuda(),
        jaw=flame_params.jaw.cuda(),
        eye=flame_params.eye.cuda(),
        scale=flame_params.scale.cuda(),
    )

    # Get vertices
    vertices = runner.predict(flame_params)
    if isinstance(vertices, UnbatchedFlameParams):
        vertices = flame_head.forward(vertices)
    faces = flame_head.faces

    # Render the video
    audio_path = os.path.join(
        sm.data_dir,
        "sequences",
        sm.sequence,
        "audio",
        "audio_recording_cleaned.ogg",
    )
    render_vertex_video(
        vertices, faces, output_path, audio_path, quick_time_compatible=quick_time_compatible)


def render_audio_prediction(
    checkpoint_path: str,
    output_path: str,
    audio_path: str | None = None,
    sequence: int | None = None,
    quick_time_compatible: bool = False,
) -> None:
    """
    Full audio to vertex prediction.

    Args:
        checkpoint_path (str): The path to the checkpoint of the VQ-VAE model.
        output_path (str): The path to save the video to.
        audio_path (str | None): The path to the audio file to use. Either this or `sequence` must
            be provided.
        sequence (int | None): The sequence to use. Either this or `audio_path` must be provided.
        quick_time_compatible (bool): Whether to make the video QuickTime compatible.
    """

    assert (audio_path is None) != (sequence is None), \
        "Either audio_path or sequence must be provided."

    if sequence is not None:
        sm = SequenceManager(sequence=sequence)
        audio_features = sm.audio_features[:]
        audio_features = audio_features.cuda()
        audio_path = os.path.join(
            sm.data_dir,
            "sequences",
            sm.sequence,
            "audio",
            "audio_recording_cleaned.ogg",
        )
    else:
        # TODO: wav2vec...
        raise NotImplementedError

    runner = Stage2Runner.load_from_checkpoint(checkpoint_path)
    runner.eval()
    runner.cuda()
    flame_head = FlameHeadWithInnerMouth()
    flame_head.cuda()

    # Get vertices
    vertices = runner.predict(audio_features)
    if isinstance(vertices, UnbatchedFlameParams):
        vertices = flame_head.forward(vertices)
    faces = flame_head.faces

    # Render the video
    render_vertex_video(
        vertices, faces, output_path, audio_path, quick_time_compatible=quick_time_compatible)


def render_flame(
    sequence: int,
    output_path: str,
    use_neck: bool = True,
    quick_time_compatible: bool = False,
) -> None:
    # Load the sequence
    sm = SequenceManager(sequence=sequence)
    flame_params = sm.flame_params[:]
    flame_params = UnbatchedFlameParams(
        shape=flame_params.shape.cuda(),
        expr=flame_params.expr.cuda(),
        neck=flame_params.neck.cuda() if use_neck else torch.zeros_like(flame_params.neck).cuda(),
        jaw=flame_params.jaw.cuda(),
        eye=flame_params.eye.cuda(),
        scale=flame_params.scale.cuda(),
    )
    flame_head = FlameHead()
    flame_head.cuda()

    # Get vertices
    vertices = flame_head.forward(flame_params)
    faces = flame_head.faces

    # Render the video
    audio_path = os.path.join(
        sm.data_dir,
        "sequences",
        sm.sequence,
        "audio",
        "audio_recording_cleaned.ogg",
    )
    render_vertex_video(
        vertices, faces, output_path, audio_path, quick_time_compatible=quick_time_compatible)


# ==================================================================================== #
#                                       Main                                           #
# ==================================================================================== #

if __name__ == '__main__':

    mode = 'vertex_reconstruction'
    # mode = 'audio_pred_sequence'
    # mode = 'flame'
    quick_time_compatible = False
    sequence = 100

    match mode:
        case 'vertex_reconstruction':
            # vertex reconstruction
            # checkpoint_path = 'tb_logs/vector_quantization/default_flame_vertex_loss/version_2/checkpoints/epoch=199-step=15400.ckpt'  # noqa
            checkpoint_path = 'tb_logs/vector_quantization/final_baseline_l2/version_0/checkpoints/epoch=199-step=15400.ckpt'  # noqa
            output_path = f'tmp/vq_reconstruction_sequence_{sequence}.mp4'
            render_vertex_reconstruction(
                checkpoint_path,
                sequence,
                output_path,
                quick_time_compatible=quick_time_compatible)

        case 'audio_pred_sequence':
            checkpoint_path = 'tb_logs/audio_prediction/code_head/version_0/checkpoints/epoch=99-step=7700.ckpt'  # noqa
            output_path = f'tmp/audio_pred_sequence_{sequence}.mp4'
            render_audio_prediction(
                checkpoint_path,
                output_path,
                sequence=sequence,
                quick_time_compatible=quick_time_compatible,
            )

        case 'flame':
            output_path = f'tmp/flame_sequence_{sequence}.mp4'
            use_neck = False
            render_flame(
                sequence,
                output_path,
                quick_time_compatible=quick_time_compatible,
                use_neck=use_neck,
            )

        case _:
            raise ValueError(f"Invalid mode: {mode}")
