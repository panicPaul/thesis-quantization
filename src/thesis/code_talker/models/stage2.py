import torch
import torch.nn as nn
from jaxtyping import Float
from tqdm import tqdm

from thesis.code_talker.models.utils import (
    PeriodicPositionalEncoding,
    enc_dec_mask,
    init_biased_mask,
)
from thesis.code_talker.stage1_runner import Stage1Runner
from thesis.data_management import FlameParams
from thesis.flame import FlameHeadWithInnerMouth


def bufferize_module(module: nn.Module) -> nn.Module:
    """ Turns every parameter of a module into a buffer. Works in place. """

    tmp = {}
    for name, param in module.named_parameters():
        tmp[name] = param

    for name, param in tmp.items():
        if '.' not in name:
            delattr(module, name)
            module.register_buffer(name, param)
        else:
            submodule_name = name.split('.')[:-1]
            parameter_name = name.split('.')[-1]
            submodule_name = '.'.join(submodule_name)
            submodule = module.get_submodule(submodule_name)
            delattr(submodule, parameter_name)
            submodule.register_buffer(parameter_name, param)

    return module


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
        self.feature_dim = feature_dim

        self.flame_head = FlameHeadWithInnerMouth()

        # load pretrained VQ-VAE model
        stage_1_runner = Stage1Runner.load_from_checkpoint(vq_vae_pretrained_path)
        self.autoencoder = bufferize_module(stage_1_runner.model)
        # self.autoencoder = stage_1_runner.model
        self.flame_mode = self.autoencoder.flame_mode
        self.use_audio = self.autoencoder.use_audio
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
        self.code_dim = self.autoencoder.code_dim
        self.feat_map = nn.Linear(feature_dim, self.face_quan_num * self.code_dim, bias=False)
        nn.init.constant_(self.feat_map.weight, 0)

    @property
    def device(self) -> torch.device:
        """ Assume all parameters are on the same device. """
        return next(self.parameters()).device

    def compute_loss(
        self,
        target_latent: torch.Tensor,
        target_vertices: Float[torch.Tensor, 'batch time n_vertices 3'],
        target_flame_params: FlameParams,
        target_audio_features: Float[torch.Tensor, 'batch time audio_feature_dim'],
        pred_latent: torch.Tensor,
        pred_vertices: Float[torch.Tensor, 'batch time n_vertices 3'] | None = None,
        pred_flame_code: Float[torch.Tensor, 'batch time 103'] | None = None,
        pred_audio_features: Float[torch.Tensor, 'batch time audio_feature_dim'] | None = None,
    ) -> tuple[Float[torch.Tensor, ""], dict]:
        """Computes the loss of the model."""
        loss_dict = {}
        loss = torch.tensor(0.0, device=target_vertices.device, requires_grad=True)

        # Motion loss
        if self.flame_mode == 'vertex':
            vertex_loss = nn.functional.mse_loss(pred_vertices, target_vertices)
            vertex_loss_l1 = nn.functional.l1_loss(pred_vertices, target_vertices)
            loss_dict['vertex_loss'] = vertex_loss
            loss_dict['vertex_loss_l1'] = vertex_loss_l1  # just for logging
            loss = loss + vertex_loss
        else:
            target_flame_code = torch.cat([target_flame_params.expr, target_flame_params.jaw],
                                          dim=-1)
            flame_code_loss = nn.functional.l1_loss(pred_flame_code, target_flame_code)
            loss_dict['flame_code_loss'] = flame_code_loss
            loss = loss + flame_code_loss*1e-3
            loss = flame_code_loss
            flame_params = FlameParams(
                shape=target_flame_params.shape,
                expr=pred_flame_code[:, :, :100],
                neck=target_flame_params.neck,
                jaw=pred_flame_code[:, :, 100:],
                eye=target_flame_params.eye,
                scale=target_flame_params.scale,
            )
            pred_vertices = self.flame_head.forward(flame_params)
            vertex_loss = nn.functional.mse_loss(pred_vertices, target_vertices)
            vertex_loss_l1 = nn.functional.l1_loss(pred_vertices, target_vertices)
            loss_dict['vertex_loss'] = vertex_loss
            loss_dict['vertex_loss_l1'] = vertex_loss_l1  # just for logging
            # loss = loss + vertex_loss*1e2
            loss = loss + vertex_loss_l1

        # Audio loss
        if self.use_audio:
            audio_loss = nn.functional.mse_loss(pred_audio_features, target_audio_features)
            loss_dict['audio_loss'] = audio_loss
            loss = loss + audio_loss*1e-4

        # Regularization loss
        regularization_loss = nn.functional.mse_loss(pred_latent, target_latent.detach())
        loss_dict['regularization_loss'] = regularization_loss
        loss = loss + regularization_loss*0.1  # self.reg_weight

        loss_dict['loss'] = loss
        return loss, loss_dict

    def forward(
        self,
        template: Float[torch.Tensor, 'batch n_vertices 3'],
        audio_features: Float[torch.Tensor, 'batch time audio_feature_dim'],
        vertices: Float[torch.Tensor, 'batch time n_vertices 3'],
        flame_params: FlameParams,
    ) -> tuple[Float[torch.Tensor, ""], dict]:
        """
        Args:
            template (torch.Tensor): template vertices of shape (batch, n_vertices, 3)
            audio_features (torch.Tensor): audio features of shape (batch, time, audio_feature_dim)
            vertices (torch.Tensor): ground truth vertices of shape (batch, time, n_vertices, 3)

        Returns:
            tuple: A tuple containing the loss, and a dictionary of the individual losses.
        """
        self.autoencoder.eval()
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1)  # (1,1,V,3)
        gt_vertices = vertices

        # audio feature processing
        audio_features_input = self.audio_feature_map(audio_features)
        batch_size, time = audio_features.shape[:2]

        # gt motion feature extraction
        # feat_q_gt, _ = self.autoencoder.get_quant(vertices - template)
        vertices = vertices - template
        flame_code = torch.cat([flame_params.expr, flame_params.jaw], dim=-1)
        feat_q_gt, _, _ = self.autoencoder.encode(
            vertices=vertices, flame_code=flame_code, audio=audio_features)
        feat_q_gt = feat_q_gt.permute(0, 2, 1)

        # auto-regressive facial motion prediction with teacher-forcing
        if self.flame_mode == 'flame':
            first_code = torch.zeros((batch_size, 1, 103), device=self.device)
            vertices_input = torch.cat((first_code, flame_code[:, :-1]), 1)  # shift one position
        else:
            vertices_input = torch.cat((template, vertices[:, :-1]), 1)  # shift one position
            vertices_input = vertices_input - template  # (batch, seq_len, v, 3)
            vertices_input = vertices_input.reshape(batch_size, time, -1)  # (batch, seq_len, V*3)

        vertices_input = self.vertices_map(vertices_input)
        vertices_input = self.PPE(vertices_input)
        tgt_mask = self.biased_mask[:, :vertices_input.shape[1], :vertices_input.shape[1]].clone(
        ).detach().to(device=self.device)
        memory_mask = enc_dec_mask(self.device, self.dataset, vertices_input.shape[1],
                                   audio_features_input.shape[1])
        feat_out = self.transformer_decoder(
            vertices_input, audio_features_input, tgt_mask=tgt_mask, memory_mask=memory_mask)
        feat_out = self.feat_map(feat_out)
        feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1] * self.face_quan_num, -1)

        feat_out = feat_out

        # feature quantization
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
        feat_out_q = feat_out_q.permute(0, 2, 1)

        # feature decoding
        vertices_out, flame_code_out, audio_out = self.autoencoder.decode(feat_out_q)
        if self.flame_mode == 'vertex':
            vertices_out = vertices_out + template  # (batch, seq_len, v, 3)

        return self.compute_loss(
            target_vertices=gt_vertices,
            target_flame_params=flame_params,
            target_audio_features=audio_features,
            target_latent=feat_q_gt,
            pred_vertices=vertices_out,
            pred_flame_code=flame_code_out,
            pred_audio_features=audio_out,
            pred_latent=feat_out_q,
        )

    @torch.no_grad()
    def predict(
        self,
        audio_features: Float[torch.Tensor, 'time audio_feature_dim'],
        template: Float[torch.Tensor, '1 n_vertices 3'],
    ) -> Float[torch.Tensor, 'time n_vertices 3'] | Float[torch.Tensor, 'time 103']:
        """
        Args:
            audio_features (torch.Tensor): audio features of shape (time, audio_feature_dim)
            template (torch.Tensor): template vertices of shape (1, n_vertices, 3)

        Returns:
            torch.Tensor: predicted vertices of shape (time, n_vertices, 3)
        """
        self.autoencoder.eval()
        audio_features = audio_features.unsqueeze(0)  # (1, time, audio_feature_dim)
        template = template.unsqueeze(1)  # (1,1,v,3)

        # audio feature processing
        audio_features_input = self.audio_feature_map(audio_features)  # (batch, time, feature_dim)
        batch, time = audio_features.shape[:2]
        frame_num = audio_features.shape[1]

        # auto-regressive facial motion prediction
        for i in tqdm(range(frame_num), desc="Predicting vertex positions"):
            if i == 0:
                # vertices_emb = obj_embedding  # (1,1,feature_dim)
                # style_emb = vertices_emb
                vertices_emb = torch.zeros((1, 1, self.feature_dim), device=self.device)
                vertices_input = self.PPE(vertices_emb)
            else:
                vertices_input = self.PPE(vertices_emb)

            tgt_mask = self.biased_mask[:, :vertices_input.shape[1], :vertices_input
                                        .shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertices_input.shape[1],
                                       audio_features_input.shape[1])
            feat_out = self.transformer_decoder(
                vertices_input, audio_features_input, tgt_mask=tgt_mask, memory_mask=memory_mask)
            feat_out = self.feat_map(feat_out)

            feat_out = feat_out.reshape(feat_out.shape[0], feat_out.shape[1] * self.face_quan_num,
                                        -1)
            # predicted feature to quantized one
            feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
            # quantized feature to vertices
            if i == 0:
                vertices_out_q, vertex_code_out_q, _ = self.autoencoder.decode(
                    torch.cat([feat_out_q, feat_out_q], dim=-1))
                if self.flame_mode == 'flame':
                    vertices_out_q = vertex_code_out_q
                vertices_out_q = vertices_out_q[:, 0].unsqueeze(1)  # (1, 1, v, 3)
                vertices_out_q = vertices_out_q.reshape(vertices_out_q.shape[0],
                                                        vertices_out_q.shape[1], -1)
            else:
                vertices_out_q, vertex_code_out_q, _ = self.autoencoder.decode(feat_out_q)
                if self.flame_mode == 'flame':
                    vertices_out_q = vertex_code_out_q
                vertices_out_q = vertices_out_q.reshape(vertices_out_q.shape[0],
                                                        vertices_out_q.shape[1], -1)

            if i != frame_num - 1:
                new_output = self.vertices_map(vertices_out_q[:, -1, :]).unsqueeze(1)
                new_output = new_output
                vertices_emb = torch.cat((vertices_emb, new_output), 1)

        # quantization and decoding
        feat_out_q, _, _ = self.autoencoder.quantize(feat_out)
        vertices_out, flame_code_out, _ = self.autoencoder.decode(feat_out_q)

        if self.flame_mode == 'vertex':
            vertices_out = vertices_out + template
            vertices_out = vertices_out.squeeze(0).reshape(frame_num, -1, 3)  # TODO: double check
        else:
            vertices_out = flame_code_out.squeeze(0)
        return vertices_out

    def analyze_gradients(
        self,
        log_grad_norm=True,
        print_max_grad=True,
        print_min_grad=True,
        print_zero_grad=True,
        verbose=False,
        limit=5,
    ):
        """Analyzes gradients of model parameters."""
        grad_norms = []
        max_grads = []
        zero_grad_params = []

        for name, param in self.named_parameters():
            if param.grad is None:
                zero_grad_params.append(name)
                continue

            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
            max_grad = param.grad.abs().max().item()
            max_grads.append((name, max_grad))

        if log_grad_norm:
            print("\nGradient Norms:")
            for name, norm in sorted(grad_norms, key=lambda x: x[1], reverse=True)[:limit]:
                print(f"{name}: {norm:.5f}")

        if print_max_grad:
            print("\nMax Gradients:")
            for name, max_g in sorted(max_grads, key=lambda x: x[1], reverse=True)[:limit]:
                print(f"{name}: {max_g:.5f}")

        if print_min_grad:
            print("\nMin Gradients:")
            for name, max_g in sorted(max_grads, key=lambda x: x[1], reverse=False)[:limit]:
                print(f"{name}: {max_g:.5e}")

        if print_zero_grad and zero_grad_params:
            print("\nParameters with Zero Gradients:")
            print(zero_grad_params)

        if verbose:
            print("\nAll Parameter Shapes:")
            for name, param in self.named_parameters():
                print(f"{name}: {param.shape}")
