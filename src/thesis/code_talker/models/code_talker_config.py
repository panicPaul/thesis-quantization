""" Configuration for the models. """

from typing import NamedTuple


class QuantizerConfig(NamedTuple):
    """
    Args:
        input_dim (int):
        hidden_size (int):
        num_hidden_layers (int):
        num_attention_heads (int):
        intermediate_size (int):
        relu_negative_slope (float):
        quant_factor (int):
        instance_normalization_affine (bool):
        n_embed (int):
        zquant_dim (int): quanitzed embedding dimension
        face_quan_num (int): number of face components
        is_audio (bool):
    """
    input_dim: int = 5143 * 3  # 3 * num_vertices
    hidden_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1536
    relu_negative_slope: float = 0.2
    quant_factor: int = 0
    instance_normalization_affine: bool = False
    n_embed: int = 256
    zquant_dim: int = 64
    face_quan_num: int = 16
    is_audio: bool = False
    # window size is hardcoded to 1


class QuantizationTrainingConfig(NamedTuple):
    """
    Configuration for quantization training.

    Args:
        num_train_workers (int): Number of workers for training data loading.
        num_val_workers (int): Number of workers for validation data loading.
        batch_size (int): Batch size for training.
        base_lr (float): Base learning rate.
        StepLR (bool): Whether to use StepLR scheduler.
        warmup_steps (int): Number of warmup steps.
        factor (float): Factor by which the learning rate will be reduced.
        patience (int): Number of epochs with no improvement after which learning rate will be
            reduced.
        threshold (float): Threshold for measuring the new optimum, to only focus on significant
            changes.
        epochs (int): Total number of training epochs.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
        start_epoch (int): Starting epoch for training.
        power (float): Power factor for polynomial learning rate decay.
        momentum (float): Momentum factor for optimizer.
        weight_decay (float): Weight decay (L2 penalty) for optimizer.
    """
    num_train_workers: int = 10
    num_val_workers: int = 4
    batch_size: int = 1
    base_lr: float = 0.0001
    StepLR: bool = True
    warmup_steps: int = 1
    factor: float = 0.3
    patience: int = 3
    threshold: float = 0.0001
    epochs: int = 200
    step_size: int = 20
    gamma: float = 0.5
    start_epoch: int = 0
    power: float = 0.9
    momentum: float = 0.9
    weight_decay: float = 0.002
