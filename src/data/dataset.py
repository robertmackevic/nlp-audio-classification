from argparse import Namespace
from typing import Optional, Tuple

import torch
import torchaudio
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, Resample

from src.paths import ROOT_DIR, DATASET_DIR


class AudioClassificationDataset(Dataset):
    def __init__(self, config: Namespace, subset: Optional[str] = None, snr_db: Optional[float] = None) -> None:
        super().__init__()
        original_dataset = SPEECHCOMMANDS(root=ROOT_DIR, download=True, subset=subset)

        self.sample_rate = config.sample_rate
        self.max_samples_in_waveform = self.sample_rate
        self.class_labels = config.classes
        self.snr_db = snr_db
        self.transform = MelSpectrogram(sample_rate=config.sample_rate, **config.mel_spec)

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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        target = self.label_to_one_hot_tensor(sample["label"])
        waveform, original_sample_rate = torchaudio.load(sample["filepath"], normalize=True)

        if original_sample_rate != self.sample_rate:
            waveform = Resample(original_sample_rate, self.sample_rate)(waveform)

        waveform = torch.nn.functional.pad(
            input=waveform,
            pad=(0, self.max_samples_in_waveform - waveform.size(1)),
            value=0.0
        )

        if self.snr_db is not None:
            waveform = self._add_noise_to_waveform(waveform)

        feature = self.transform(waveform)

        return waveform, feature, target

    def label_to_one_hot_tensor(self, label: str) -> Tensor:
        return one_hot(torch.tensor(self.class_labels.index(label)), num_classes=len(self.class_labels)).float()

    def _add_noise_to_waveform(self, waveform: Tensor) -> Tensor:
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        noise = torch.normal(0, torch.sqrt(noise_power).item(), size=waveform.shape)
        return waveform + noise
