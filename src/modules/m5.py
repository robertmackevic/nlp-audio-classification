from argparse import Namespace

from torch import Tensor
from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, ReLU, MaxPool1d, Linear, AdaptiveAvgPool1d


class M5(Module):
    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.network = Sequential(
            Conv1d(1, 128, kernel_size=80, stride=4, bias=False),
            BatchNorm1d(128, ),
            ReLU(),
            MaxPool1d(4),

            Conv1d(128, 128, kernel_size=3, bias=False),
            BatchNorm1d(128),
            ReLU(),
            MaxPool1d(4),

            Conv1d(128, 256, kernel_size=3, bias=False),
            BatchNorm1d(256),
            ReLU(),
            MaxPool1d(4),

            Conv1d(256, 512, kernel_size=3, bias=False),
            BatchNorm1d(512),
            ReLU(),
            MaxPool1d(4)
        )

        self.avg_pool = AdaptiveAvgPool1d(1)
        self.fc = Linear(512, len(config.classes))

    def forward(self, x: Tensor) -> Tensor:
        x = self.network(x)
        x = self.avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x
