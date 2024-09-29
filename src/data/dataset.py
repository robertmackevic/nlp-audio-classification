from argparse import Namespace
from typing import Optional, Tuple

import torch
import torchaudio
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample

from src.data.transform import add_noise_to_waveform
from src.paths import ROOT_DIR, DATASET_DIR


class AudioClassificationDataset(Dataset):
    def __init__(self, config: Namespace, subset: Optional[str] = None, snr_db: Optional[float] = None) -> None:
        super().__init__()
        original_dataset = SPEECHCOMMANDS(root=ROOT_DIR, download=True, subset=subset)

        self.sample_rate = config.sample_rate
        self.class_labels = config.classes
        self.snr_db = snr_db

        self.samples = [
            {
                "filepath": DATASET_DIR / metadata[0],
                "label": metadata[2],
                "speaker": metadata[3]
            }
            for i in range(len(original_dataset))
            if (metadata := original_dataset.get_metadata(n=i))[2] in self.class_labels
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def label_to_one_hot_tensor(self, label: str) -> Tensor:
        return one_hot(torch.tensor(self.class_labels.index(label)), num_classes=len(self.class_labels)).float()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sample = self.samples[index]
        target = self.label_to_one_hot_tensor(sample["label"])
        waveform, original_sample_rate = torchaudio.load(sample["filepath"], normalize=True)

        if original_sample_rate != self.sample_rate:
            waveform = Resample(original_sample_rate, self.sample_rate)(waveform)

        if self.snr_db is not None:
            waveform = add_noise_to_waveform(waveform, self.snr_db)

        return waveform, target
