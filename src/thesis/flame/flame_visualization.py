""" Flame visualization module. """

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from thesis.data_management import UnbatchedFlameParams
from thesis.flame import FlameHead
from thesis.video_utils import add_audio


def _generate_image(
    flame_params: UnbatchedFlameParams,
    flame_head: FlameHead,
    image_height: int = 1_000,
    image_width: int = 1_000,
    time_step: int = 0,
):
    # ): -> Float[torch.Tensor, "h w 3"]:
    """
    Generates an image from the flame parameters.
    """

    # Generate the vertices and faces
    flame_head.to(flame_params.shape.device)
    shape = flame_params.shape[time_step:time_step + 1]
    expr = flame_params.expr[time_step:time_step + 1]
    neck = flame_params.neck[time_step:time_step + 1]
    jaw = flame_params.jaw[time_step:time_step + 1]
    eye = flame_params.eye[time_step:time_step + 1]
    scale = flame_params.scale[time_step:time_step + 1]
    flame_params = UnbatchedFlameParams(shape, expr, neck, jaw, eye, scale)
    vertices = flame_head.forward(flame_params).squeeze(0).detach().cpu().numpy()
    faces = flame_head.faces.detach().cpu().numpy()

    # Generate the matplotlib image
    fig = plt.figure(figsize=(image_width / 100, image_height / 100))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 2], vertices[:, 0], faces, vertices[:, 1], shade=True)
    ax.view_init(azim=10, elev=10)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.set_axis_off()

    # Convert the matplotlib image to a numpy array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def generate_flame_video(
    flame_params: UnbatchedFlameParams,
    output_path: str,
    fps: int = 24,
    audio_path: str | None = None,
) -> None:
    """ Generates a video, visualizing the flame parameters."""

    # Generate the images
    images = []
    flame_head = FlameHead()
    for time_step in tqdm(range(flame_params.shape.shape[0]), desc="Generating images"):
        image = _generate_image(flame_params, flame_head, time_step=time_step)
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
        add_audio(output_path, audio_path, fps)


def main():
    from thesis.audio_to_flame.windowed import prediction_loop
    from thesis.data_management import SequenceManager

    if True:
        ckpt_path = 'tb_logs/audio2flame/my_model/version_0/checkpoints/epoch=24-step=9375.ckpt'
        audio_path = 'tmp/audio_recording_cleaned_s3.ogg'
        params = prediction_loop(ckpt_path, audio_path)
        generate_flame_video(params, 'tmp/flame_video.mp4', 24, audio_path)
    else:
        # s100
        audio_path = 'tmp/audio_recording_cleaned.ogg'
        sm = SequenceManager(sequence=100)
        expr = sm.flame_params[:].expr.cuda()
        jaw = sm.flame_params[:].jaw.cuda()
        time = expr.shape[0]
        params = UnbatchedFlameParams(
            shape=torch.zeros((time, 300), device='cuda'),
            expr=expr,
            neck=torch.zeros((time, 3), device='cuda'),
            jaw=jaw,
            eye=torch.zeros((time, 6), device='cuda'),
            scale=torch.ones((time, 1), device='cuda'),
        )
        generate_flame_video(params, 'tmp/flame_video_gt.mp4', 30, audio_path)


if __name__ == '__main__':
    main()
