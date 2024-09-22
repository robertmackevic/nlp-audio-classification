from argparse import Namespace
from os import listdir, makedirs
from typing import Dict, Optional

from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from src.modules.m5 import M5
from src.paths import RUNS_DIR, CONFIG_FILE
from src.utils import get_available_device, get_logger, save_config, save_weights


class AverageMeter:
    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    def __init__(self, config: Namespace) -> None:
        self.config = config
        self.device = get_available_device()
        self.logger = get_logger(__name__)

        self.model = M5(config).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = CrossEntropyLoss().to(self.device)

    def fit(self, train_dl: DataLoader, val_dl: DataLoader) -> None:
        RUNS_DIR.mkdir(exist_ok=True, parents=True)
        model_dir = RUNS_DIR / f"v{len(listdir(RUNS_DIR)) + 1}"

        summary_writer_train = SummaryWriter(log_dir=model_dir / "train")
        summary_writer_eval = SummaryWriter(log_dir=model_dir / "eval")

        makedirs(summary_writer_train.log_dir, exist_ok=True)
        makedirs(summary_writer_eval.log_dir, exist_ok=True)
        save_config(self.config, model_dir / CONFIG_FILE.name)

        best_score = 0
        best_score_metric = self.config.best_score_metric

        for epoch in range(1, self.config.epochs + 1):
            metrics = self._train_for_epoch(train_dl)
            self.logger.info(f"[Epoch {epoch} / {self.config.epochs}]")
            self.log_metrics(metrics, summary_writer_train, epoch=epoch)

            if epoch % self.config.eval_interval == 0:
                metrics = self.eval(val_dl)
                self.log_metrics(metrics, summary_writer_eval, epoch=epoch)

                score = metrics[best_score_metric].avg

                if score > best_score:
                    best_score = score
                    self.logger.info(f"Saving best weights with {best_score_metric}: {score:.3f}")
                    save_weights(model_dir / "weights_best.pth", self.model)

            if epoch % self.config.save_interval == 0:
                self.logger.info(f"Saving model weights at epoch: {epoch}")
                save_weights(model_dir / f"weights_{epoch}.pth", self.model)

    def _train_for_epoch(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.train()
        metrics = {
            "loss": AverageMeter(),
        }
        return metrics

    def eval(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.eval()
        metrics = {
            "loss": AverageMeter(),
        }
        return metrics

    def _forward(self, batch: Tensor):
        pass

    def log_metrics(
            self,
            metrics: Dict[str, AverageMeter],
            summary_writer: Optional[SummaryWriter] = None,
            epoch: Optional[int] = None
    ) -> None:
        message = ""
        for metric, value in metrics.items():
            message += f"{metric}: {value.avg:.3f} | "

            if epoch is not None and summary_writer is not None:
                summary_writer.add_scalar(tag=metric, scalar_value=value.avg, global_step=epoch)

        self.logger.info(message[:-3])
