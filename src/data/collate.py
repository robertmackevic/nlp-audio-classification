from typing import Tuple, List

import torch
from torch import Tensor


def collate_fn(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    waveforms, targets = zip(*batch)
    max_length = max(waveform.size(1) for waveform in waveforms)

    padded_waveforms = torch.stack([
        torch.nn.functional.pad(waveform, (0, max_length - waveform.size(1)), value=0.0)
        for waveform in waveforms
    ])
    targets = torch.stack(targets)

    return padded_waveforms, targets
