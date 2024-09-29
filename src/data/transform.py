from argparse import Namespace

import torch
from torch import Tensor
from torch.nn import Module
from torchaudio.transforms import MelSpectrogram


def get_2d_transform(config: Namespace) -> Module:
    return MelSpectrogram(
        sample_rate=config.sample_rate,
        **config.mel_spec,
    )


def add_noise_to_waveform(waveform: Tensor, snr_db: float) -> Tensor:
    signal_power = torch.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise
