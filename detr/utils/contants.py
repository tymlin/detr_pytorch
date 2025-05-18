import logging
from datetime import datetime
from pathlib import Path

from .types import ModuleStage

ROOT = Path(__file__).parent.parent.parent

RUNS_DIRPATH = ROOT / "runs"
CONFIG_TRAINING_DIRPATH = ROOT / "config_training"
CONFIG_INFERENCE_DIRPATH = ROOT / "config_inference"
DATA_DIRPATH = ROOT / "data_temp"
NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LOGGER_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER_DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_FILE_NAME: str | None = "logs.log"
LOGGER_LEVEL = logging.INFO

MODULE_STAGES: list[ModuleStage] = ["train", "val"]
