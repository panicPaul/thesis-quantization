from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from vector_quantize_pytorch import FSQ

from thesis.code_talker.models.lib.base_models import (
    LinearEmbedding,
    PositionalEncoding,
    Transformer,
)
from thesis.code_talker.models.lib.quantizer import VectorQuantizer


class BottleNeck(nn.Module):
    """ BottleNeck for ablation and debugging """

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim) if dim is not None else nn.Identity()

    def forward(
        self, x: Float[torch.Tensor, "batch time c_in"]
    ) -> tuple[
            Float[torch.Tensor, "batch code_dim h"],
            Float[torch.Tensor, ""],
            tuple,
    ]:
        # embedding_loss = torch.tensor(0.0, requires_grad=True)  # dummy value
        # info = ()  # empty tuple
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        return x


class LFQWrapper(nn.Module):
    """ LFQ Wrapper """

    def __init__(
        self,
        dim: int = 8,  # codebook size will be 2**dim
        entropy_loss_weight=0.1,
        diversity_gamma=1.,
    ) -> None:
        pass

    def forward(
        self, x: Float[torch.Tensor, "batch time c_in"]
    ) -> tuple[
            Float[torch.Tensor, "batch code_dim h"],
            Float[torch.Tensor, ""],
            tuple,
    ]:
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch, time, c_in) where c_in is the flattened
                vertex coordinates, i.e. V*3

        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple]: tuple containing:
                - quant (torch.Tensor): quantized representation of the input tensor
                    h is time * n_face_regions
                - emb_loss (torch.Tensor): embedding loss
                - info (tuple): additional information
        """
        raise NotImplementedError


class FSQWrapper(nn.Module):
    """ FSQ Wrapper """

    def __init__(self, latent_dim: int, levels: list[int] = [8, 6, 5]) -> None:
        super().__init__()
        self.levels = levels
        self.latent_dim = latent_dim
        self.fsq = FSQ(levels=levels, dim=latent_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch time c_in"]
    ) -> tuple[
            Float[torch.Tensor, "batch code_dim h"],
            Float[torch.Tensor, ""],
            tuple,
    ]:
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch, time, c_in) where c_in is the flattened
                vertex coordinates, i.e. V*3

        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple]: tuple containing:
                - quant (torch.Tensor): quantized representation of the input tensor
                    h is time * n_face_regions
                - emb_loss (torch.Tensor): embedding loss
                - info (tuple): additional information
        """
        embedding_loss = torch.tensor(0.0, requires_grad=True)  # dummy value
        info = ()  # empty tuple

        batch_size, time, c_in = x.shape
        x, _ = self.fsq.forward(x)
        # move channel dimension to second position
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, self.latent_dim, -1)

        return x, embedding_loss, info


# TODO: also add magvit model


class VQAutoEncoder(nn.Module):
    """ VQ-GAN model """

    def __init__(
        self,
        n_vertices: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        relu_negative_slope: float,
        quant_factor: int,
        instance_normalization_affine: bool,
        n_embed: int,
        code_dim: int,
        face_quan_num: int,
        is_audio=False,
        quantization_mode: Literal['default', 'fsq', 'bottleneck'] = 'default',
        disable_neck: bool = False,
    ):
        super().__init__()
        self.disable_neck = disable_neck
        self.code_dim = code_dim
        self.face_quan_num = face_quan_num
        self.input_dim = n_vertices * 3
        self.n_vertices = n_vertices

        self.encoder = TransformerEncoder(
            input_dim=n_vertices * 3,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            relu_negative_slope=relu_negative_slope,
            quant_factor=quant_factor,
            instance_normalization_affine=instance_normalization_affine,
        )
        self.decoder = TransformerDecoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            relu_negative_slope=relu_negative_slope,
            quant_factor=quant_factor,
            instance_normalization_affine=instance_normalization_affine,
            out_dim=n_vertices * 3,
            is_audio=is_audio,
        )
        self.face_quan_num = face_quan_num  # h in the paper
        match quantization_mode:
            case 'default':
                self.quantize = VectorQuantizer(n_embed, code_dim, beta=0.25)
            case 'fsq':
                self.quantize = FSQWrapper(latent_dim=code_dim, levels=[face_quan_num])
            case 'bottleneck':
                self.quantize = BottleNeck(dim=code_dim)
            case _:
                raise ValueError(f"Invalid quantization mode: {quantization_mode}")

    def encode(
        self, x: Float[torch.Tensor, "batch time n_vertices 3"]
    ) -> tuple[
            Float[torch.Tensor, "batch code_dim th"],
            Float[torch.Tensor, ""],
            tuple,
    ]:
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch, time, n_vertices, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple]: tuple containing:
                - quant (torch.Tensor): quantized representation of the input tensor
                    shape is (batch, code_dim, time * n_face_regions)
                - emb_loss (torch.Tensor): embedding loss
                - info (tuple): additional information
        """
        batch_size, time, n_vertices, _ = x.shape
        x = x.view(batch_size, time, -1)
        h = self.encoder(x)  # x --> z'
        h = h.view(batch_size, time, self.face_quan_num, self.code_dim)  # (b, t, h, c)
        h = h.view(batch_size, time * self.face_quan_num, self.code_dim)  # (b, t*h, c)
        quant, emb_loss, info = self.quantize(h)  # finds nearest quantization
        return quant, emb_loss, info

    def decode(
        self,
        quant: Float[torch.Tensor, "batch code_dim th"],
    ) -> Float[torch.Tensor, "batch time n_vertices 3"]:
        """
        Args:
            quant (torch.Tensor): quantized representation of the input tensor.
                Shape is (batch, code_dim, time * n_face_regions)

        Returns:
            torch.Tensor: reconstructed tensor of shape (batch, time, n_vertices, 3)
        """
        # BCL
        batch_size, code_dim, ht = quant.shape
        quant = quant.permute(0, 2, 1)
        quant = quant.view(batch_size, -1, self.face_quan_num, self.code_dim).contiguous()
        quant = quant.view(batch_size, -1, self.face_quan_num * self.code_dim).contiguous()
        quant = quant.permute(0, 2, 1).contiguous()
        dec = self.decoder(quant)  # z' --> x
        dec = dec.view(batch_size, -1, self.n_vertices, 3)
        return dec * 1e-3

    def forward(
        self,
        x: Float[torch.Tensor, "batch time n_vertices 3"],
        template: Float[torch.Tensor, "batch n_vertices 3"],
    ) -> tuple[
            Float[torch.Tensor, "batch time n_vertices 3"],
            Float[torch.Tensor, ""],
            tuple,
    ]:
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch, time, c_in) where c_in is the flattened
                vertex coordinates, i.e. V*3
            template (torch.Tensor): template tensor of shape (batch, c_in) where c_in is the
                flattened vertex coordinates, i.e. V*3

        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple]: tuple containing:
                - dec (torch.Tensor): reconstructed tensor of shape (batch, time, c_in) where c_in
                    is the flattened vertex coordinates, i.e. V*3
                - emb_loss (torch.Tensor): embedding loss
                - info (tuple): additional information
        """
        template = template.unsqueeze(1)  # (b, 1, v, 3)
        x = x - template  # (b, t, v, 3)

        quant, emb_loss, info = self.encode(x)  # (b, c, t*h)
        dec = self.decode(quant)  # (b, t, v, 3)

        dec = dec + template
        return dec, emb_loss, info

    def sample_step(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check

    def get_quant(self, x):
        quant_z, _, info = self.encode(x)
        indices = info[2]
        return quant_z, indices

    def get_distances(self, x):
        h = self.encoder(x)  # x --> z'
        d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def entry_to_feature(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1), shape=None)
        quant_z = torch.reshape(quant_z, zshape)
        return quant_z

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1), shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0, 2, 1)  # B L 1 -> B L C -> B C L
        x = self.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
        else:
            ix = logits
        ix = torch.reshape(ix, (-1, 1))
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self,
                  logits,
                  sample=True,
                  filter_value=-float('Inf'),
                  temperature=0.7,
                  top_p=0.9,
                  sample_idx=None):
        """ function that samples the distribution of logits. (used in test)
        if sample_idx is None, we perform nucleus sampling
        """
        logits = logits / temperature
        sample_idx = 0

        probs = F.softmax(logits, dim=-1)  # B, N, embed_num
        if sample:
            # multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0] * shape[1], shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx + 1)
            probs = probs.reshape(shape[0], shape[1], shape[2])
            ix = ix.reshape(shape[0], shape[1])
        else:
            # top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs


class TransformerEncoder(nn.Module):
    """ Encoder class for VQ-VAE with Transformer backbone """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        relu_negative_slope: float,
        quant_factor: int,
        instance_normalization_affine: bool,
    ) -> None:
        super().__init__()

        size = input_dim
        dim = hidden_size
        self.vertices_mapping = nn.Sequential(
            nn.Linear(size, dim), nn.LeakyReLU(relu_negative_slope, True))
        if quant_factor == 0:
            layers = [
                nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(relu_negative_slope, True),
                    nn.InstanceNorm1d(dim, affine=instance_normalization_affine))
            ]
        else:
            layers = [
                nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=2, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(relu_negative_slope, True),
                    nn.InstanceNorm1d(dim, affine=instance_normalization_affine))
            ]
            for _ in range(1, quant_factor):
                layers += [
                    nn.Sequential(
                        nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                        nn.LeakyReLU(relu_negative_slope, True),
                        nn.InstanceNorm1d(dim, affine=instance_normalization_affine),
                        nn.MaxPool1d(2))
                ]
        self.squasher = nn.Sequential(*layers)
        self.encoder_transformer = Transformer(
            in_size=hidden_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )
        self.encoder_pos_embedding = PositionalEncoding(hidden_size)
        self.encoder_linear_embedding = LinearEmbedding(hidden_size, hidden_size)

    def forward(self, inputs):
        # downsample into path-wise length seq before passing into transformer
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = self.vertices_mapping(inputs)
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)  # [N L C]

        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, dummy_mask))

        return encoder_features


class TransformerDecoder(nn.Module):
    """ Decoder class for VQ-VAE with Transformer backbone """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        relu_negative_slope: float,
        quant_factor: int,
        instance_normalization_affine: bool,
        out_dim,
        is_audio=False,
    ) -> None:
        super().__init__()
        size = hidden_size
        dim = hidden_size
        self.expander = nn.ModuleList()
        if quant_factor == 0:
            self.expander.append(
                nn.Sequential(
                    nn.Conv1d(size, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(relu_negative_slope, True),
                    nn.InstanceNorm1d(dim, affine=instance_normalization_affine)))
        else:
            self.expander.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        size,
                        dim,
                        5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                        padding_mode='replicate'), nn.LeakyReLU(relu_negative_slope, True),
                    nn.InstanceNorm1d(dim, affine=instance_normalization_affine)))
            num_layers = quant_factor+2 \
                if is_audio else quant_factor

            for _ in range(1, num_layers):
                self.expander.append(
                    nn.Sequential(
                        nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                        nn.LeakyReLU(relu_negative_slope, True),
                        nn.InstanceNorm1d(dim, affine=instance_normalization_affine),
                    ))
        self.decoder_transformer = Transformer(
            in_size=hidden_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )
        self.decoder_pos_embedding = PositionalEncoding(hidden_size)
        self.decoder_linear_embedding = LinearEmbedding(hidden_size, hidden_size)

        self.vertices_map_reverse = nn.Linear(hidden_size, out_dim)

    def forward(self, inputs):
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        # upsample into original length seq before passing into transformer
        for i, module in enumerate(self.expander):
            inputs = module(inputs)
            if i > 0:
                inputs = inputs.repeat_interleave(2, dim=2)
        inputs = inputs.permute(0, 2, 1)  # BLC
        decoder_features = self.decoder_linear_embedding(inputs)
        decoder_features = self.decoder_pos_embedding(decoder_features)

        decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
        pred_recon = self.vertices_map_reverse(decoder_features)
        return pred_recon
