from argparse import Namespace

from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    Linear,
    MaxPool2d,
    Flatten,
    AdaptiveMaxPool2d,
)


class AudioClassifier(Module):

    def __init__(self, config: Namespace) -> None:
        super().__init__()

        self.network = Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(),
            AdaptiveMaxPool2d(output_size=(16, 16)),

            Flatten(),

            Linear(64 * 16 * 16, 256),
            ReLU(),
            Linear(256, len(config.classes)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
