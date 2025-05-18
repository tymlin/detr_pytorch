import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from torch import nn

import wandb

from ..utils.contants import ROOT
from .base import BaseLogger, Status, register_logger
from .pylogger import log


@register_logger
class WandbLogger(BaseLogger):
    """Logger for logging with Weights and Biases."""

    name: str = "wandb"
    default_artifact_type = "default_type"
    default_artifact_name = "default_name"

    def __init__(
        self,
        log_path: str | Path,
        file_log: logging.Logger,
        experiment_name: str,
        run_name: str | None = None,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        description: str = "",
        params: dict | None = None,
        api_key: str = None,
        group: str = None,
        tags: list[str] | None = None,
        delete_old_artifacts_versions: bool = True,
        delete_old_run_media_files: bool = True,
        log_image_artifacts: bool = True,
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

        if api_key:
            wandb.login(key=api_key)

        self._run_id = run_id
        self.group = group
        self.tags = tags
        self.delete_old_artifacts_versions = delete_old_artifacts_versions
        self.delete_old_run_media_files = delete_old_run_media_files
        self.log_image_artifacts = log_image_artifacts

    def start_run(self):
        kwargs = {}
        if self._run_id:
            kwargs["id"] = self._run_id
            kwargs["resume"] = "must"
            log.info(f"Resuming Run with ID: {self._run_id}")
        wandb.init(
            project=self.experiment_name,
            name=self.run_name,
            # config=self.config,
            group=self.group,
            notes=self.description,
            tags=self.tags,
            dir=ROOT / "wandb",
            **kwargs,
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

        if self.params is not None:
            self.log_params(self.params)
        super().start_run()

    @property
    def run_id(self) -> str:
        return wandb.run.id

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        super().log_metric(key, value, step)
        value.tolist() if isinstance(value, np.ndarray) else value
        if step is not None:
            wandb.log({key: value, "step": step})
            # wandb.log({key: value}, step=step)
        else:
            wandb.log({key: value})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        super().log_metrics(metrics, step)
        metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
        if step is not None:
            wandb.log({**metrics, "step": step})
            # wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_param(self, key: str, value: float | str | int) -> None:
        """Log config key (aka param in MLFlow) to the logger."""
        wandb.config.update({key: value})

    def log_params(self, params: dict[str, Any]) -> None:
        """Log config (aka params in MLFlow) to the logger."""
        wandb.config.update(params)

    def log_artifact(self, local_path: str, artifact_path: str | None = None, **kwargs) -> None:
        artifact_name = kwargs.get("artifact_name", self.default_artifact_name)
        artifact_type = kwargs.get("artifact_type", self.default_artifact_type)
        artifact_name = f"{artifact_name}_{self.run_name}_{self.run_id}"
        if self.delete_old_artifacts_versions:
            self.delete_artifact(artifact_type, artifact_name)
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(local_path)
        wandb.log_artifact(artifact)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None, **kwargs) -> None:
        artifact_name = kwargs.get("artifact_name", self.default_artifact_name)
        artifact_type = kwargs.get("artifact_type", self.default_artifact_type)
        artifact_name = f"{artifact_name}_{self.run_name}_{self.run_id}"
        if self.delete_old_artifacts_versions:
            self.delete_artifact(artifact_type, artifact_name)
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_dir(local_dir)
        wandb.log_artifact(artifact)

    def delete_artifact(self, artifact_type: str, artifact_name: str) -> None:
        api = wandb.Api()
        try:
            collection = api.artifact_collection(
                artifact_type, f"{wandb.run.entity}/{self.experiment_name}/{artifact_name}"
            )
            collection.delete()
        except Exception:
            # log.warning(f"Error while deleting old versions of artifact {artifact_name}: {e}")
            pass

    def delete_run_files(self, artifact_name: str) -> None:
        api = wandb.Api()
        try:
            run = api.run(f"{wandb.run.entity}/{self.experiment_name}/{self.run_id}")
            files = run.files()
            for file in files:
                if artifact_name in file.name:
                    file.delete()
                    # log.info(f"Deleted file: {file.name}")
        except Exception:
            # log.error(f"Error deleting run files: {e}")
            pass

    def download_artifact(self, artifact_path: str) -> str:
        dst_path = str(self.log_path / "loaded")
        log.info(f"Downloading {artifact_path} from run {self.run_id} to {dst_path}")
        return wandb.use_artifact(artifact_path).download(dst_path)

    def log_image(self, image, **kwargs) -> None:
        """Logs an image to wandb.

        :param image: (numpy array, string, io) Accepts numpy array of image data, or a PIL image.
        """
        if self.log_image_artifacts:
            artifact_name = kwargs.get("artifact_name", self.default_artifact_name)
            # if self.delete_old_run_media_files:
            #     self.delete_run_files(artifact_name)
            caption = kwargs.get("caption")
            wandb.log({artifact_name: wandb.Image(image, caption=caption)})

    def log_image_bbox(
        self, image, preds: dict[str, list], targets: dict[str, list], classes_int2str: dict[int, str], **kwargs
    ) -> None:
        """Logs an image with bounding boxes to wandb.

        :param image: (numpy array, string, io) Accepts numpy array of image data, or a PIL image.
        :param preds: Dictionary containing predicted boxes, scores, and labels.
        :param targets: Dictionary containing target boxes and labels.
        :param classes_int2str: Dictionary mapping class indices to class names.
        :param kwargs: Additional arguments for logging.

        :return: None
        """
        if self.log_image_artifacts:
            artifact_name = kwargs.get("artifact_name", self.default_artifact_name)
            caption = kwargs.get("caption")

            pred_box_data = []
            for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
                pred_box_data.append(
                    {
                        "position": {"minX": box[0], "maxX": box[2], "minY": box[1], "maxY": box[3]},
                        "class_id": label,
                        "box_caption": f"{classes_int2str[label]} ({score:.2%})",
                        "scores": {"confidence": score},
                        "domain": "pixel",
                    }
                )
            target_data = []
            for box, label in zip(targets["boxes"], targets["labels"]):
                target_data.append(
                    {
                        "position": {"minX": box[0], "maxX": box[2], "minY": box[1], "maxY": box[3]},
                        "class_id": label,
                        "box_caption": classes_int2str[label],
                        "domain": "pixel",
                    }
                )
            boxes_dict = {
                "predictions": {
                    "box_data": pred_box_data,
                    "class_labels": classes_int2str,
                },
                "ground_truth": {
                    "box_data": target_data,
                    "class_labels": classes_int2str,
                },
            }
            img = wandb.Image(
                image,
                boxes=boxes_dict,
                caption=caption,
            )
            wandb.log({artifact_name: img})

    def log_html(self, html_filepath: str, **kwargs) -> None:
        """Logs an HTML string to wandb.

        :param html_filepath: HTML path to log.
        """
        if self.log_image_artifacts:
            artifact_name = kwargs.get("artifact_name", self.default_artifact_name)
            if self.delete_old_run_media_files:
                self.delete_run_files(artifact_name)
            with open(html_filepath) as file:
                html_content = file.read()
            wandb.log({artifact_name: wandb.Html(html_content)})

    def log_confusion_matrix(self, targets: list, preds: list, classes_names: list[str], name: str) -> None:
        wandb.log(
            {name: wandb.plot.confusion_matrix(probs=None, y_true=targets, preds=preds, class_names=classes_names)}
        )

    def log_bar_plot(self, values: list, class_names: list[str], name: str) -> None:
        metric_name = name.split("/")[-1]
        data = [[name, value] for (name, value) in zip(class_names, values)]
        table = wandb.Table(data=data, columns=["class_names", metric_name])
        wandb.log({name: wandb.plot.bar(table, "class_names", metric_name, title=metric_name)})

    def log_model(
        self,
        model: nn.Module,
        log_type: Literal["gradients", "parameters", "all"] | None = "gradients",
        log_freq: int = 1000,
    ) -> None:
        """Logs the model's architecture and parameters to wandb.

        Also logs the gradients and parameters of the model.

        :param model: PyTorch model
        :param log_type: What to log. Choose from 'gradients', 'parameters', 'all'
        :param log_freq: Frequency (in batches) to log gradients and parameters
        """
        try:
            wandb.watch(model, log=log_type, log_freq=log_freq)
            # log.info(f"Model {model_name} logged successfully.")
        except Exception as e:
            log.error(f"Error logging model: {e}")

    def finalize(self, status: Status) -> None:
        super().finalize(status)
        if status == Status.FINISHED:  # Status is based on MLFlow Status
            wandb.finish()
        wandb.finish(exit_code=status.value)
