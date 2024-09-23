from argparse import Namespace
from typing import Tuple, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.dataset import AudioClassificationDataset


def collate_fn(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    waveforms, targets = zip(*batch)
    max_length = max(waveform.size(1) for waveform in waveforms)

    padded_waveforms = torch.stack([
        torch.nn.functional.pad(waveform, (0, max_length - waveform.size(1)), value=0.0)
        for waveform in waveforms
    ])
    targets = torch.stack(targets)

    return padded_waveforms, targets


def get_dataloader(config: Namespace, subset: Optional[str] = None) -> DataLoader:
    return DataLoader(
        AudioClassificationDataset(config, subset),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
