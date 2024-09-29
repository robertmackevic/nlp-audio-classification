from argparse import Namespace

from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, Linear, AdaptiveAvgPool2d, Softmax, MaxPool2d


class AudioClassifier(Module):

    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.network = Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(),

            Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(128),
            ReLU(),

            Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(512),
            ReLU(),
        )

        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc = Linear(512, len(config.classes))
        self.softmax = Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.network(x)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return self.softmax(x)
