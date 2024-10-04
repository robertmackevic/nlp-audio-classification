from argparse import Namespace, ArgumentParser

from src.data.loader import get_dataloader
from src.paths import RUNS_DIR, CONFIG_FILE
from src.trainer import Trainer
from src.utils import get_logger, load_config, seed_everything, load_weights, count_parameters


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", type=str, required=True, help="v1, v2, v3, etc.")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Name of the .pth file")
    parser.add_argument("-s", "--subset", type=str, required=True, choices=["training", "validation", "testing"])
    return parser.parse_args()


def run(version: str, weights: str, subset: str) -> None:
    logger = get_logger()
    model_dir = RUNS_DIR / version

    config = load_config(model_dir / CONFIG_FILE.name)
    seed_everything(config.seed)

    for snr_db in [None, 10, -10]:
        logger.info(f"Preparing the data... (SNR: {snr_db})")
        dataloader = get_dataloader(config, subset=subset, snr_db=snr_db)

        trainer = Trainer(config)
        trainer.model = load_weights(filepath=model_dir / weights, model=trainer.model)
        logger.info(f"Number of model parameters: {count_parameters(trainer.model):,}")

        try:
            trainer.log_metrics(trainer.eval(dataloader))
        except KeyboardInterrupt:
            logger.info("Evaluation terminated.")


if __name__ == "__main__":
    run(**vars(parse_args()))
