import torch
import torch.nn as nn

from thesis.code_talker.stage1_runner import Stage1Runner
from thesis.code_talker.models.utils import (
    PeriodicPositionalEncoding,
    enc_dec_mask,
    init_biased_mask,
)
from jaxtyping import Float


class CodeTalker(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        positional_encoding_period: int,
        n_head: int,
        num_layers: int,
        vq_vae_pretrained_path: str,
        reg_weight: float,
        motion_weight: float,
    ) -> None:
        super().__init__()
        """
        Args:
            feature_dim: dimension of the feature space
            vertices_dim: dimension of the vertices space
            positional_encoding_period: period of the positional encoding
            n_head: number of heads in the transformer
            num_layers: number of layers in the transformer
            zquant_dim: dimension of the quantized feature
            face_quan_num: number of quantized features
            vq_vae_pretrained_path: path to the pretrained VQ-VAE model
            reg_weight: weight for the regularization loss
            motion_weight: weight for the motion loss
        """
        self.reg_weight = reg_weight
        self.motion_weight = motion_weight
        self.dataset = "vocaset"

        # load pretrained VQ-VAE model
        stage_1_runner = Stage1Runner.load_from_checkpoint(vq_vae_pretrained_path)
        self.autoencoder = stage_1_runner.model
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.disable_neck = stage_1_runner.config.disable_neck

        # audio map
        self.audio_feature_map = nn.Linear(1024, feature_dim)

        # motion encoder
        vertices_dim = self.autoencoder.input_dim
        self.vertices_map = nn.Linear(vertices_dim, feature_dim)

        # periodic positional encoding
        self.PPE = PeriodicPositionalEncoding(feature_dim, period=positional_encoding_period)

        # temporal bias
        self.biased_mask = init_biased_mask(
            n_head=4, max_seq_len=600, period=positional_encoding_period)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=n_head, dim_feedforward=2 * feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # motion decoder
        self.face_quan_num = self.autoencoder.face_quan_num
        self.zquant_dim = self.autoencoder.zquant_dim
        self.feat_map = nn.Linear(feature_dim, self.face_quan_num * self.zquant_dim, bias=False)
        nn.init.constant_(self.feat_map.weight, 0)

    @property
    def device(self) -> torch.device:
        """ Assume all parameters are on the same device. """
        return next(self.parameters()).device

    def forward(
        self,
        audio_features: Float[torch.Tensor, 'batch time audio_feature_dim'],
        template: Float[torch.Tensor, 'batch vertices_dim'],
        vertices: Float[torch.Tensor, 'batch time vertices_dim'],
    ) -> tuple[
            Float[torch.Tensor, ""],
            Float[torch.Tensor, ""],
            Float[torch.Tensor, ""],
    ]:
        """
        Args:
            audio_features (torch.Tensor): audio features of shape (batch, time, audio_feature_dim)
            template (torch.Tensor): template vertices of shape (batch, vertices_dim)
            vertices (torch.Tensor): ground truth vertices of shape (batch, time, vertices_dim)

        Returns:
            tuple: A tuple containing the loss, motion loss, and regularization loss.
        """
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1)  # (1,1,V*3)

        # audio feature processing
        audio_features = self.audio_feature_map(audio_features)

        # gt motion feature extraction
        #feat_q_gt, _ = self.autoencoder.get_quant(vertices - template)
        feat_q_gt, _, _ = self.autoencoder.encode(vertices - template)
        feat_q_gt = feat_q_gt.permute(0, 2, 1)

        # auto-regressive facial motion prediction with teacher-forcing
        vertices_input = torch.cat((template, vertices[:, :-1]), 1)  # shift one position
        vertices_input = vertices_input - template
        vertices_input = self.vertices_map(vertices_input)
        vertices_input = vertices_input
        vertices_input = self.PPE(vertices_input)
        tgt_mask = self.biased_mask[:, :vertices_input.shape[1], :vertices_input.shape[1]].clone(
        ).detach().to(device=self.device)
        memory_mask = enc_dec_mask(self.device, self.dataset, vertices_input.shape[1],
                                   audio_features.shape[1])
        feat_out = self.transformer_decoder(
            vertices_input, audio_features, tgt_mask=tgt_mask, memory_mask=memory_mask)
        feat_out = self.feat_map(feat_out)
        feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1] * self.face_quan_num, -1)

        # feature quantization
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)

        # feature decoding
        vertices_out = self.autoencoder.decode(feat_out_q)
        vertices_out = vertices_out + template

        # loss
        loss_motion = nn.functional.mse_loss(vertices_out, vertices)  # (batch, seq_len, V*3)
        # loss_motion = nn.functional.l1_loss(vertices_out, vertices)
        loss_reg = nn.functional.mse_loss(feat_out, feat_q_gt.detach())

        return self.motion_weight * loss_motion + self.reg_weight * loss_reg, loss_motion, loss_reg

    def predict(
        self,
        audio_features: Float[torch.Tensor, 'batch time audio_feature_dim'],
        template: Float[torch.Tensor, 'batch vertices_dim'],
    ):
        raise NotImplementedError("This method is not implemented yet.")
        template = template.unsqueeze(1)  # (1,1, V*3)

        # audio feature processing
        audio_features = self.audio_feature_map(audio_features)  # (batch, time, feature_dim)
        frame_num = audio_features.shape[1]

        # autoregressive facial motion prediction
        for i in range(frame_num):
            if i == 0:
                vertices_emb = obj_embedding  # (1,1,feature_dim)
                style_emb = vertices_emb
                vertices_input = self.PPE(style_emb)
            else:
                vertices_input = self.PPE(vertices_emb)

            tgt_mask = self.biased_mask[:, :vertices_input.shape[1], :vertices_input
                                        .shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertices_input.shape[1],
                                       audio_features.shape[1])
            feat_out = self.transformer_decoder(
                vertices_input, audio_features, tgt_mask=tgt_mask, memory_mask=memory_mask)
            feat_out = self.feat_map(feat_out)

            feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1] * self.face_quan_num,
                                        -1)
            # predicted feature to quantized one
            feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
            # quantized feature to vertices
            if i == 0:
                vertices_out_q = self.autoencoder.decode(
                    torch.cat([feat_out_q, feat_out_q], dim=-1))
                vertices_out_q = vertices_out_q[:, 0].unsqueeze(1)
            else:
                vertices_out_q = self.autoencoder.decode(feat_out_q)

            if i != frame_num - 1:
                new_output = self.vertices_map(vertices_out_q[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertices_emb = torch.cat((vertices_emb, new_output), 1)

        # quantization and decoding
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
        vertices_out = self.autoencoder.decode(feat_out_q)

        vertices_out = vertices_out + template
        return vertices_out
