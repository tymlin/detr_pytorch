"""Base logger class."""

import json
import logging
import uuid
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ..utils.contants import NOW
from ..utils.registry import Registry, create_register_decorator
from .pylogger import log

LOGGERS = Registry()
register_logger = create_register_decorator(LOGGERS)


class Status(Enum):
    """Based on MLFlow"""

    FINISHED = 0
    RUNNING = 1
    SCHEDULED = 2
    FAILED = 3
    KILLED = 4
    STOPPED = 5


class LoggerResults:
    """Storage for training results."""

    def __init__(self):
        self.steps: dict[str, list[int]] = defaultdict(lambda: [], {})
        self.metrics: dict[str, list[float]] = defaultdict(lambda: [], {})
        self.params: dict[str, Any] = {}

    def update_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Append new metrics."""
        for name, value in metrics.items():
            self.metrics[name].append(value)
            if step is not None:
                self.steps[name].append(step)

    def update_params(self, params: dict[str, Any]) -> None:
        """Update params dictionary."""
        self.params.update(params)

    def get_metrics(self) -> dict[str, dict[str, list[int | float]]]:
        """Return metrics for each split and each step."""
        metrics: dict[str, dict] = {name: {} for name in self.metrics}
        for name in self.metrics:
            metrics[name]["value"] = self.metrics[name]
            if name in self.steps:
                metrics[name]["step"] = self.steps[name]
        return metrics


class BaseLogger:
    """Base logger class."""

    log_dirs = [
        "logs",
        "checkpoints",
        "model",
        "eval_examples",
        "dataset",
        "other_plots",
        "epochs_metrics",
    ]

    name: str = "base"

    def __init__(
        self,
        log_path: str | Path,
        file_log: logging.Logger,
        experiment_name: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        description: str = "",
        params: dict | None = None,
        **kwargs,
    ):
        if config is None:
            config = {}
        self.config = config
        self._log_config_info()
        log_path = Path(log_path) if isinstance(log_path, str) else log_path
        self.log_path = log_path
        self.logs_dirpath = log_path / "logs"
        self.ckpt_dirpath = log_path / "checkpoints"
        self.model_dirpath = log_path / "model"
        self.eval_examples_dirpath = log_path / "eval_examples"
        self.other_plots_dirpath = log_path / "other_plots"
        self.epochs_metrics_dirpath = log_path / "epochs_metrics"

        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)
        self.model_dirpath.mkdir(parents=True, exist_ok=True)
        self.logs_dirpath.mkdir(exist_ok=True, parents=True)
        self.eval_examples_dirpath.mkdir(exist_ok=True, parents=True)
        self.other_plots_dirpath.mkdir(exist_ok=True, parents=True)
        self.epochs_metrics_dirpath.mkdir(exist_ok=True, parents=True)

        self.results = LoggerResults()
        self.timestamp = NOW
        self.history_artifacts_dir = f"history/{self.timestamp}"
        self._run_id = None

        self.file_log = file_log
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.description = description
        self.params = params

    def _log_config_info(self):
        cfg_repr = json.dumps(self.config, indent=4)
        log.info(f"Experiment config:\n{cfg_repr}")

    @property
    def run_id(self) -> str:
        if self._run_id is not None:
            return self._run_id
        else:
            self._run_id = str(uuid.uuid1())
            return self._run_id

    def start_run(self):
        msg = f"Starting {self.__class__.__name__}"
        if self.run_id:
            msg += f" with `run_id`: {self.run_id}"
        log.info(msg)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log single metric."""
        self.results.update_metrics({key: value}, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics."""
        self.results.update_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log params."""
        self.results.update_params(params)

    def log_dict(self, dct: dict[str, Any], filepath: str = "dct.yaml") -> None:
        """Log dict to yaml file."""
        with open(filepath, "w") as file:
            yaml.dump(dct, file, default_flow_style=False)

    def log_config(self) -> None:
        """Log config to yaml."""
        config_local_filepath = f"{self.log_path}/config.yaml"
        with open(config_local_filepath, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)
        # log to remote root directory
        self.log_artifact(config_local_filepath, "", artifact_name="config", artifact_type="config")
        # log to remote history directory
        self.log_artifact(
            config_local_filepath,
            self.history_artifacts_dir,
            artifact_name="config_history",
            artifact_type="config",
        )
        log.info("Config file logged to remote.")

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None, **kwargs) -> None:
        """Log directory artifacts."""
        pass

    def log_artifact(self, local_path: str, artifact_path: str | None = None, **kwargs) -> None:
        """Log artifact."""
        pass

    def log_image(self, image, **kwargs) -> None:
        """Log image."""
        pass

    def log_html(self, html_filepath: str, **kwargs) -> None:
        """Log html."""
        pass

    def log_confusion_matrix(self, targets: list, preds: list, classes_names: list[str], name: str) -> None:
        """Log confusion matrix."""
        pass

    def log_bar_plot(self, values: list, class_names: list[str], name: str) -> None:
        """Log bar plot."""
        pass

    def log_image_bbox(self, **kwargs):
        """Log image with bounding boxes."""
        pass

    def log_model(self, model, **kwargs) -> None:
        """Log model."""
        pass

    def finalize(self, status: Status) -> None:
        """Close logger"""
        # log to remote history directory
        logs_remote_dirpath = f"{self.history_artifacts_dir}/logs"
        self.log_artifacts(str(self.logs_dirpath), logs_remote_dirpath, artifact_name="logs", artifact_type="logs")
        log.warning(f"Experiment {status.value}. Closing {self.__class__.__name__}")
