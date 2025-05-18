"""MLFlow logger."""

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.client
import mlflow.entities

from .base import BaseLogger, Status, register_logger
from .pylogger import log

mlflow.enable_system_metrics_logging()

run_id_to_system_metrics_monitor = {}


@register_logger
class MLFlowLogger(BaseLogger):
    """Logger for logging with MLFlow."""

    client: mlflow.client.MlflowClient
    run: mlflow.entities.Run
    name: str = "mlflow"

    def __init__(
        self,
        log_path: str | Path,
        file_log: logging.Logger,
        config: dict[str, Any],
        experiment_name: str,
        run_name: str | None = None,
        host: str | None = None,
        port: str | None = None,
        run_id: str | None = None,
        resume: bool = True,
        description: str = "",
        log_system_metrics: bool = True,
        params: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            log_path=log_path,
            file_log=file_log,
            config=config,
            experiment_name=experiment_name,
            run_name=run_name,
            description=description,
            params=params,
        )
        if host is None:
            host = "localhost"  # 127.0.0.1
        if port is None:
            port = "5000"
        tracking_uri = f"http://{host}:{port}"
        self.tracking_uri = tracking_uri
        self.resume = resume
        self._run_id = run_id
        self.log_system_metrics = log_system_metrics
        self.run = None

    def start_run(self):
        super().start_run()
        client = mlflow.client.MlflowClient(self.tracking_uri)

        mlflow.set_tracking_uri(self.tracking_uri)

        run_name = self.run_name
        exp_name = self.experiment_name

        mlflow.get_experiment_by_name(exp_name)
        experiment = client.get_experiment_by_name(exp_name)
        if experiment is None:
            experiment_id = client.create_experiment(exp_name)
            experiment = client.get_experiment(experiment_id)
        experiment_id = experiment.experiment_id

        if not self.resume:
            log.info(f"-> Creating new run with {run_name} name")
            run = client.create_run(experiment_id, run_name=run_name)

        elif self._run_id is None:
            # get run by name
            runs = client.search_runs(
                experiment_ids=[str(experiment_id)],
                filter_string=f'tags.mlflow.runName = "{run_name}"',
            )
            num_runs = len(runs)
            if num_runs == 0:
                log.info(f"\tThere is no run with '{run_name}' name (for experiment '{exp_name}')")
                log.info(f"\t-> Creating new run with '{run_name}' name")
                run = client.create_run(experiment_id, run_name=run_name)
            if num_runs == 1:
                log.info(f"\tFound existing run with '{run_name}' name on mlflow server")
                run = runs[0]
                run = client.get_run(run.info.run_id)
                log.info(f"-> Resuming Run '{run.info.run_name}' (ID = {run.info.run_id})")
            elif num_runs > 1:
                log.info(
                    f"\tMore than one run with '{run_name}' name found on mlflow server." "Raising Exception",
                    logging.WARN,
                )
                raise ValueError()
        else:
            run = client.get_run(self._run_id)  # get run by id
            log.info(f"-> Resuming Run {run.info.run_name} (ID = {run.info.run_id})")
        self.client = client
        if self.log_system_metrics:
            log.info("-> Starting SystemMetricsMonitor")
            from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

            system_monitor = SystemMetricsMonitor(
                run.info.run_id,
                resume_logging=self._run_id is not None,
            )
            global run_id_to_system_metrics_monitor

            run_id_to_system_metrics_monitor[run.info.run_id] = system_monitor
            system_monitor.start()
        client.update_run(run_id=run.info.run_id, status=Status.RUNNING.name)
        self.run = run
        run_url = f"{self.tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
        log.info(f"Visit run at: {run_url}")

        if self.params is not None:
            self.log_params(self.params)
        self.set_tag("mlflow.note.content", self.description)

    @property
    def run_id(self) -> str:
        if self.run is None:
            return ""
        return self.run.info.run_id

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        super().log_metric(key, value, step)
        if not isinstance(value, (int, float)):
            # log.warning(f"Value of metric {key} is not a `float` or `int`. Skipping logging")
            return
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        super().log_metrics(metrics, step)
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_param(self, key: str, value: float | str | int) -> None:
        self.client.log_param(self.run_id, key, value)

    def set_tag(self, key: str, value: float | str | int) -> None:
        self.client.set_tag(self.run_id, key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            self.log_param(key, value)

    def log_dict(self, dct: dict[str, Any], filename: str = "config.yaml") -> None:
        self.client.log_dict(self.run_id, dct, filename)

    def log_artifacts(self, local_dir: str, artifact_dirpath: str | None = None, **kwargs) -> None:
        self.client.log_artifacts(self.run_id, local_dir, artifact_dirpath)

    def log_artifact(self, local_path: str, artifact_dirpath: str | None = None, **kwargs) -> None:
        self.client.log_artifact(self.run_id, local_path, artifact_dirpath)

    def download_artifact(self, artifact_path: str) -> str:
        """Download artifact from mlflow.

        Artifact is stored in dst_path (relative to log dir) directory
        Returns path to the downloaded artifact
        """
        dst_path = str(self.log_path / "loaded")
        log.info(f"-> Downloading {artifact_path} from mlflow run {self.run_id} to {dst_path}")
        return self.client.download_artifacts(
            run_id=self.run_id,
            path=artifact_path,
            dst_path=dst_path,
        )

    def finalize(self, status: Status) -> None:
        self.client.set_terminated(self.run_id, status=status.name)
        super().finalize(status)
