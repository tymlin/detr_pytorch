import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from detr.loggers import Status
from detr.loggers.pylogger import log
from detr.metrics import MetricsStorage
from detr.modules import DETRModuleStepOut
from detr.utils.files import relpath, save_txt_to_file, save_yaml
from detr.utils.misc import add_tab_to_lines
from detr.utils.registry import Registry, create_register_decorator
from detr.utils.types import ModuleStage
from detr.utils.visualization import plot_metrics_matplotlib, plot_results

if TYPE_CHECKING:
    from detr.trainer import Trainer

CALLBACKS = Registry()
register_callback = create_register_decorator(CALLBACKS)


class BaseCallback:
    """Base class for all callbacks"""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def on_fit_start(self, trainer: "Trainer") -> None:
        """Called on fit method start; after the trainer checkpoint is loaded"""
        pass

    def on_epoch_start(self, trainer: "Trainer") -> None:
        """Called on epoch start."""
        pass

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Called on epoch end."""
        pass

    def on_failure(self, trainer: "Trainer", status: Status) -> None:
        """Called when exception is raised during training"""
        pass

    def on_step_end(self, trainer: "Trainer") -> None:
        """Called on training/validation step end."""
        pass

    def state_dict(self) -> dict:
        """Return Callback state dict"""
        return {}

    def load_state_dict(self, state_dict: dict):
        """Load Callback state dict."""
        pass


class Callbacks:
    """Wrapper used to store many callbacks"""

    def __init__(self, callbacks: list[BaseCallback] | None = None) -> None:
        """Initialize Callbacks.

        :param: callbacks (list[BaseCallback]): List of callbacks used during training process.
        """
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def on_step_end(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_step_end(trainer)

    def on_fit_start(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_fit_start(trainer)

    def on_epoch_start(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(trainer)

    def on_epoch_end(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def on_failure(self, trainer: "Trainer", status: Status) -> None:
        log.warning("Callbacks: Failure mode detected. Running callbacks `on_failure` methods")
        for callback in self.callbacks:
            callback.on_failure(trainer, status)

    def state_dict(self) -> dict[str, dict]:
        state_dict = {}
        for callback in self.callbacks:
            name = callback.__class__.__name__
            state_dict[name] = callback.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, dict]):
        for callback in self.callbacks:
            name = callback.__class__.__name__
            callback.load_state_dict(state_dict.get(name, {}))


@register_callback
class ModelCheckpoint(BaseCallback):
    def __init__(
        self,
        monitor: str = "loss",
        stage: ModuleStage = "val",
        mode: Literal["min", "max"] = "min",
        save_last: bool = False,
    ) -> None:
        """Initialize ModelCheckpoint callback.

        :param monitor: metric name to monitor
        :param stage: stage to monitor, either 'train' or 'val'
        :param mode: metric mode, either 'min' or 'max'
        :param save_last: whether to save last checkpoint or just the best one
        """
        self.monitor = monitor
        self.stage = stage
        self.mode = mode
        self.save_last = save_last

        self.best_metric_value = torch.inf if self.mode == "min" else -torch.inf
        if self.mode == "min":
            self.is_better = lambda x, y: x < y
        else:
            self.is_better = lambda x, y: x > y

        if self.stage not in ["train", "val"]:
            e = ValueError(f"Invalid stage {self.stage}, available: ['train', 'val']")
            log.error(str(e))
            raise e

        self.best_metric_name = f"best_{self.stage}_{self.monitor}"

    def on_epoch_end(self, trainer: "Trainer"):
        ckpt_dirpath = trainer.logger.ckpt_dirpath
        artifact_dirpath = ckpt_dirpath.name

        if self.save_last:
            fpath = str(ckpt_dirpath / "last.pt")
            trainer.save_checkpoint(fpath)
            trainer.logger.log_artifact(fpath, artifact_dirpath, artifact_name="last", artifact_type=artifact_dirpath)
            log.debug(f"{self.name}: Saving the `last` checkpoint to '{relpath(fpath)}'")

        metrics_storage = trainer.metrics_storage_epoch.aggregate_over_key(key="epoch")
        if not metrics_storage.metrics:
            e = ValueError(f"{self.name}: No metrics have been logged yet.")
            log.error(str(e))
            raise e

        metric_value = metrics_storage.metrics.get(self.monitor, {}).get(self.stage, [])
        if not metric_value:
            e = ValueError(f"{self.name}: No metrics value found for {self.monitor} in {self.stage}")
            log.error(str(e))
            raise e

        last_metric_value = metric_value[-1]["value"]
        if self.is_better(last_metric_value, self.best_metric_value):
            fpath = str(ckpt_dirpath / f"{self.best_metric_name}.pt")
            trainer.save_checkpoint(fpath)
            trainer.logger.log_artifact(
                fpath, artifact_dirpath, artifact_name=self.best_metric_name, artifact_type=artifact_dirpath
            )
            log.debug(f"{self.name}: Saving the `best` checkpoint to '{relpath(fpath)}'")
            log.info(
                f"{self.name}: Found new best value for `{self.stage}/{self.monitor}` metric: "
                f"current={last_metric_value:.4f}, best={self.best_metric_value:.4f}"
            )
            self.best_metric_value = last_metric_value

    def state_dict(self) -> dict:
        return {self.best_metric_name: self.best_metric_value}

    def load_state_dict(self, state_dict: dict):
        self.best_metric_value = state_dict.get(self.best_metric_name, self.best_metric_value)
        log.info(f"Loaded `{self.name}` state ({self.best_metric_name} = {self.best_metric_value})")


@register_callback
class ModelSummary(BaseCallback):
    def __init__(self, depth: int) -> None:
        self.depth = depth

    def on_fit_start(self, trainer: "Trainer") -> None:
        model_basic_info = trainer.module.model.summary_ops()
        optim_info = str(trainer.module.optimizer)
        loss_fn_info = str(trainer.module.loss_fn)
        lr_schedulers_info = str(trainer.module.lr_scheduler)
        metrics_info = trainer.module.metrics[trainer.module.stages[0]].summary()
        transforms_train_info = str(trainer.datamodule.train_ds.transforms) if trainer.datamodule.train_ds else "None"
        transforms_val_info = str(trainer.datamodule.val_ds.transforms) if trainer.datamodule.val_ds else "None"

        sep = ""
        module_info_dict = {
            f"{sep}Model": model_basic_info,
            f"{sep}Optimizer": optim_info,
            f"{sep}Loss function": loss_fn_info,
            f"{sep}LR scheduler": lr_schedulers_info,
            f"{sep}Metrics": metrics_info,
            f"{sep}Transforms train": transforms_train_info,
            f"{sep}Transforms val": transforms_val_info,
        }
        msg = f"`{trainer.module.name}` summary: \n"
        for name, info in module_info_dict.items():
            msg += add_tab_to_lines(info, name, sep * 2 if sep else "\t") + "\n"
        log.info(msg)
        model_summary = trainer.module.model.summary(depth=self.depth)
        model_dirpath = trainer.logger.model_dirpath
        artifact_dirpath = model_dirpath.name
        filepath = f"{model_dirpath}/model_summary.txt"
        save_txt_to_file(model_summary, filepath)
        trainer.logger.log_artifact(
            filepath, artifact_dirpath, artifact_name="model_summary", artifact_type=artifact_dirpath
        )


@register_callback
class MetricsSaverCallback(BaseCallback):
    """Plot per epoch metrics"""

    filename = "epoch_metrics"

    def __init__(self, log_every_n_epochs: int = 1):
        self.log_every_n_epochs = log_every_n_epochs

    @staticmethod
    def save(cls, metrics_dirpath: str | Path, metrics_storage: MetricsStorage) -> None:
        # yaml_filepath = f"{metrics_dirpath}/{cls.filename}.yaml"
        # metrics = metrics_storage.to_dict()
        # metrics_converted = convert_arrays_to_lists(metrics)
        # save_yaml(metrics_converted, yaml_filepath)

        csv_filepath = f"{metrics_dirpath}/{cls.filename}.csv"
        metrics_df = metrics_storage.to_df()
        metrics_df.to_csv(csv_filepath, index=True)

        txt_filepath = f"{metrics_dirpath}/{cls.filename}.txt"
        formatted_df = metrics_df.round(5).to_string(index=True, col_space=10, justify="right")
        with open(txt_filepath, "w") as file:
            file.write(formatted_df)

    @staticmethod
    def plot(cls, metrics_dirpath: str | Path, metrics_storage: MetricsStorage, classes_name: list[str]) -> None:
        mpl_filepath = f"{metrics_dirpath}/{cls.filename}.png"
        plot_metrics_matplotlib(metrics_storage, "epoch", filepath=mpl_filepath, classes_names=classes_name)

    def on_epoch_end(self, trainer: "Trainer") -> None:
        metrics_dirpath = trainer.logger.epochs_metrics_dirpath
        metrics_storage = trainer.metrics_storage_epoch.aggregate_over_key(key="epoch")
        if not metrics_storage.metrics:
            e = ValueError(f"{self.name}: No metrics have been logged yet.")
            log.error(str(e))
            raise e

        classes_names = trainer.datamodule.get_classes_names_dataset()
        self.save(self, metrics_dirpath, metrics_storage)
        self.plot(self, metrics_dirpath, metrics_storage, classes_names)

        artifact_dirpath = metrics_dirpath.name
        trainer.logger.log_artifacts(
            metrics_dirpath, artifact_dirpath, artifact_name=artifact_dirpath, artifact_type=artifact_dirpath
        )


@register_callback
class ResultsPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(
        self,
        ncols: int = 5,
        max_n: int = 25,
    ):
        self.ncols = ncols
        self.max_n = max_n

    def log_image_with_bboxes(
        self, trainer: "Trainer", results: list[DETRModuleStepOut], classes_int2str: dict[int, str], name: str
    ) -> None:
        images = [r.inputs_inv for r in results]
        targets_bboxes = [r.targets_inv["boxes"] for r in results]
        targets_labels = [r.targets_inv["labels"] for r in results]
        outputs_bboxes = [r.outputs_postprocessed["boxes"] for r in results]
        outputs_labels = [r.outputs_postprocessed["labels"] for r in results]
        outputs_scores = [r.outputs_postprocessed["scores"] for r in results]

        for i, image in enumerate(images):
            output_bboxes = outputs_bboxes[i]
            output_bboxes = [o.tolist() for o in output_bboxes]
            output_labels = outputs_labels[i]
            output_labels = [o.tolist() for o in output_labels]
            output_scores = outputs_scores[i] if outputs_scores[i] is not None else None
            if output_scores is not None:
                output_scores = [o.tolist() for o in output_scores]
            target_bboxes = targets_bboxes[i]
            target_bboxes = [t.tolist() for t in target_bboxes]
            target_labels = targets_labels[i]
            target_labels = [t.tolist() for t in target_labels]

            preds = {
                "boxes": output_bboxes,
                "labels": output_labels,
                "scores": output_scores,
            }
            targets = {
                "boxes": target_bboxes,
                "labels": target_labels,
            }
            trainer.logger.log_image_bbox(
                image=image, preds=preds, targets=targets, classes_int2str=classes_int2str, artifact_name=name
            )

    def on_epoch_end(self, trainer: "Trainer") -> None:
        eval_examples_dir = trainer.logger.eval_examples_dirpath
        artifact_dirpath = eval_examples_dir.name
        val_plot_results = trainer.random_val_results
        k_sampels = min(self.max_n, len(val_plot_results))
        val_plot_results = random.sample(val_plot_results, k=k_sampels)
        if len(val_plot_results) > 0:
            filepath = str(eval_examples_dir / f"epoch_{trainer.current_epoch}.jpg")
            classes_int2str = trainer.datamodule.get_classes_int2str_dataset()
            colors = trainer.datamodule.get_classes_int2color_dataset()
            plot_results(val_plot_results, classes_int2str, colors=colors, ncols=self.ncols, filepath=filepath)
            self.log_image_with_bboxes(
                trainer, val_plot_results, classes_int2str, name=f"{artifact_dirpath}-img_bboxes"
            )
            trainer.logger.log_artifacts(
                eval_examples_dir, artifact_dirpath, artifact_name=artifact_dirpath, artifact_type=artifact_dirpath
            )
        else:
            log.warning(f"{self.name}: No results to visualize")
