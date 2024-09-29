from argparse import Namespace
from functools import partial
from typing import Tuple, List, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from src.data.dataset import AudioClassificationDataset
from src.data.transform import get_2d_transform


def collate_fn(batch: List[Tensor], transform: Module) -> Tuple[Tensor, Tensor]:
    waveforms, targets = zip(*batch)
    max_length = max(waveform.size(1) for waveform in waveforms)

    padded_features = torch.stack([
        transform(torch.nn.functional.pad(waveform, (0, max_length - waveform.size(1)), value=0.0))
        for waveform in waveforms
    ])
    targets = torch.stack(targets)

    return padded_features, targets


def get_dataloader(
        config: Namespace,
        subset: Optional[str] = None,
        snr_db: Optional[float] = None,
        shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        AudioClassificationDataset(config, subset, snr_db),
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=partial(collate_fn, transform=get_2d_transform(config)),
    )
