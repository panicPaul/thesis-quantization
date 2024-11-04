import numpy as np
import torch

from thesis.code_talker.stage2_runner import Stage2Runner
from thesis.data_management import SequenceManager

ckpt = 'tb_logs/audio_prediction/prediction_default/version_0/checkpoints/epoch=99-step=7700.ckpt'
model = Stage2Runner.load_from_checkpoint(ckpt)
sm = SequenceManager(3)
audio_features = sm.audio_features[:]
print(f'audio_features.shape: {audio_features.shape}')
audio_features = audio_features.to('cuda').unsqueeze(0)
vertices = model.predict(audio_features)
# save vertices as tmp.pt
vertices = torch.save(vertices, 'tmp.pt')
