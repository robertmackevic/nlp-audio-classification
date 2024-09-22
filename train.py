from torch.utils.data import DataLoader

from src.data.collate import collate_fn
from src.data.dataset import AudioClassificationDataset
from src.trainer import Trainer
from src.utils import get_logger, load_config, seed_everything


def run() -> None:
    logger = get_logger()
    config = load_config()
    seed_everything(config.seed)

    logger.info("Preparing the data...")
    train_dataset = AudioClassificationDataset(subset="training")
    val_dataset = AudioClassificationDataset(subset="validation")

    train_dl = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    trainer = Trainer(config)

    try:
        logger.info("Starting training...")
        trainer.fit(train_dl, val_dl)
    except KeyboardInterrupt:
        logger.info("Training terminated.")


if __name__ == "__main__":
    run()
