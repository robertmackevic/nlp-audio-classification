from argparse import Namespace
from os import listdir, makedirs

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src.paths import RUNS_DIR, CONFIG_FILE
from src.utils import get_available_device, get_logger, save_config


class Trainer:
    def __init__(self, config: Namespace) -> None:
        self.config = config
        self.device = get_available_device()
        self.logger = get_logger(__name__)

    def fit(self, train_dl: DataLoader, val_dl: DataLoader) -> None:
        RUNS_DIR.mkdir(exist_ok=True, parents=True)
        model_dir = RUNS_DIR / f"v{len(listdir(RUNS_DIR)) + 1}"

        summary_writer_train = SummaryWriter(log_dir=model_dir / "train")
        summary_writer_eval = SummaryWriter(log_dir=model_dir / "eval")

        makedirs(summary_writer_train.log_dir, exist_ok=True)
        makedirs(summary_writer_eval.log_dir, exist_ok=True)
        save_config(self.config, model_dir / CONFIG_FILE.name)

    def eval(self, dataloader: DataLoader) -> None:
        pass
