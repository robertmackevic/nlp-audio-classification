from src.data.loader import get_dataloader
from src.trainer import Trainer
from src.utils import get_logger, load_config, seed_everything, count_parameters


def run() -> None:
    logger = get_logger()
    config = load_config()
    seed_everything(config.seed)

    logger.info("Preparing the data...")
    train_dl = get_dataloader(config, subset="training")
    val_dl = get_dataloader(config, subset="validation")

    trainer = Trainer(config)
    logger.info(f"Number of trainable parameters: {count_parameters(trainer.model):,}")

    try:
        logger.info("Starting training...")
        trainer.fit(train_dl, val_dl)
    except KeyboardInterrupt:
        logger.info("Training terminated.")


if __name__ == "__main__":
    run()
