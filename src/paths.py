from pathlib import Path

from torchaudio.datasets.speechcommands import FOLDER_IN_ARCHIVE

ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = ROOT_DIR / FOLDER_IN_ARCHIVE

CONFIG_FILE = ROOT_DIR / "config.json"
