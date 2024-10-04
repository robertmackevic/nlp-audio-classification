from argparse import Namespace
from typing import Tuple, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.dataset import AudioClassificationDataset


def collate_fn(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    _, features, targets = zip(*batch)
    return torch.stack(features), torch.stack(targets)


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
        collate_fn=collate_fn,
    )
