import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from detr.callbacks import Callbacks
from detr.data import DataModule
from detr.loggers import BaseLogger, Status, log, log_breaking_point, remove_last_logged_line
from detr.metrics import MetricsStorage
from detr.modules import ObjectDetectionModule
from detr.utils.files import relpath
from detr.utils.torch_utils import get_device_and_id
from detr.utils.types import Accelerator


class Trainer:
    def __init__(
        self,
        module: ObjectDetectionModule,
        datamodule: DataModule,
        callbacks: Callbacks,
        logger: BaseLogger,
        accelerator: Accelerator,
        max_epochs: int,
        limit_batches: int,
        default_device_id: int,
        ckpt_path: str | None,
    ) -> None:
        self.module = module
        self.datamodule = datamodule
        self.callbacks = callbacks
        self.logger = logger

        self.accelerator = accelerator
        self.max_epochs = max_epochs
        self.limit_batches = limit_batches
        self.default_device_id = default_device_id

        self.device, self.device_id = get_device_and_id(accelerator, default_device_id)

        self.ckpt_path = ckpt_path

        self.current_epoch = 0
        self.current_step = -1
        self.metrics_storage_epoch = MetricsStorage(name="epoch")

        self.MAX_RANDOM_VAL_RESULTS = 30  # max number results to store, higher number more memory usage
        self.random_val_results = []
        self.val_targets = {}
        self.val_preds = {}

        self.num_char = 150

    def fit(self) -> None:
        self.datamodule.setup_dataloaders()
        train_dataloader = self.datamodule.train_dataloader
        val_dataloader = self.datamodule.val_dataloader
        log.info(
            "Dataloader batches:\n"
            f"Train: {self.get_number_of_batches(train_dataloader)}/{len(train_dataloader)} batches\n"
            f"Val  : {self.get_number_of_batches(val_dataloader)}/{len(val_dataloader)}  batches"
        )
        self.module.setup(self.device, self.device_id, inv_image_transform=self.datamodule.inverse_transforms)
        if isinstance(self.module.lr_scheduler.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            # important check for OneCycleLR
            num_batches = self.get_number_of_batches(train_dataloader)
            assert self.module.lr_scheduler.total_steps == num_batches * self.max_epochs, (
                f"Total steps in OneCycleLR scheduler should be equal to "
                f"`num_batches * max_epochs = {num_batches} * {self.max_epochs} = {num_batches * self.max_epochs}`, "
                f"provided `total_steps`={self.module.lr_scheduler.total_steps}"
            )
        self.load_checkpoint(self.ckpt_path)
        self.on_fit_start()
        self._log_starting_msg()

        start_epoch = self.current_epoch
        try:
            for epoch in range(start_epoch, self.max_epochs):
                self.current_epoch = epoch
                self.on_epoch_start()
                self.training_epoch(train_dataloader)
                self.validation_epoch(val_dataloader)
                self._update_metrics_storage()
                self.on_epoch_end()
        except KeyboardInterrupt as e:
            log.exception(str(e) + "KeyboardInterrupt")
            self.callbacks.on_failure(self, Status.KILLED)
            self.logger.finalize(status=Status.KILLED)
            torch.cuda.empty_cache()
            raise e
        self.logger.finalize(status=Status.FINISHED)
        torch.cuda.empty_cache()

    def training_epoch(self, dataloader: DataLoader) -> None:
        self.module.model.train()
        self.module.loss_fn.train()
        num_batches = self.get_number_of_batches(dataloader)
        tqdm_dataloader = tqdm(
            dataloader, desc="Training", total=num_batches, unit=" batch", leave=True, ncols=self.num_char
        )
        for _, batch in enumerate(tqdm_dataloader):
            if num_batches == 0:
                break
            self.module.training_step(batch)
            self.callbacks.on_step_end(self)

            num_batches -= 1
            self.current_step += 1
            # calculate metrics only for visu purposes
            metrics = self.module.compute_metrics(stage="train")
            metrics_visu = {name: f"{value:.4f}" for name, value in metrics.items() if "loss" in name}
            map = metrics.get("meanaverageprecision_avg-macro--map")
            if map is not None:
                metrics_visu["map"] = f"{map:.4f}"
            tqdm_dataloader.set_postfix(metrics_visu)
            self.logger.file_log.info(str(tqdm_dataloader))
            remove_last_logged_line(log)  # remove last line of log -> tqdm progress bar

        self.logger.file_log.info(str(tqdm_dataloader))

    def validation_epoch(self, dataloader: DataLoader) -> None:
        self.module.model.eval()
        self.module.loss_fn.eval()
        num_batches = self.get_number_of_batches(dataloader)
        random_idx = random.choice(list(range(dataloader.batch_size - 1)))

        tqdm_dataloader = tqdm(
            dataloader, desc="Validating", total=num_batches, unit=" batch", leave=True, ncols=self.num_char
        )
        for _, batch in enumerate(tqdm_dataloader):
            if num_batches == 0:
                break
            val_out = self.module.validation_step(batch)
            val_out.to_numpy()
            random_idx = min(random_idx, len(val_out) - 1)
            if num_batches < self.MAX_RANDOM_VAL_RESULTS or (
                len(self.random_val_results) < self.MAX_RANDOM_VAL_RESULTS and random.uniform(0, 1) > 0.5
            ):
                self.random_val_results.append(val_out[random_idx])

            targets_bbox = [t["boxes"] for t in val_out.targets_inv]
            targets_labels = [t["labels"] for t in val_out.targets_inv]
            outputs_bbox = [t["boxes"] for t in val_out.outputs_postprocessed]
            outputs_labels = [t["labels"] for t in val_out.outputs_postprocessed]
            outputs_scores = [t["scores"] for t in val_out.outputs_postprocessed]
            self.val_targets.setdefault("boxes", []).extend(targets_bbox)
            self.val_targets.setdefault("labels", []).extend(targets_labels)
            self.val_preds.setdefault("boxes", []).extend(outputs_bbox)
            self.val_preds.setdefault("labels", []).extend(outputs_labels)
            self.val_preds.setdefault("scores", []).extend(outputs_scores)
            self.callbacks.on_step_end(self)

            num_batches -= 1
            # calculate metrics only for visu purposes
            metrics = self.module.compute_metrics(stage="val")
            metrics_visu = {name: f"{value:.4f}" for name, value in metrics.items() if "loss" in name}
            map = metrics.get("meanaverageprecision_avg-macro--map")
            if map is not None:
                metrics_visu["map"] = f"{map:.4f}"
            tqdm_dataloader.set_postfix(metrics_visu)
            self.logger.file_log.info(str(tqdm_dataloader))
            remove_last_logged_line(log)  # remove last line of log -> tqdm progress bar
        self.logger.file_log.info(str(tqdm_dataloader))

    def get_number_of_batches(self, dataloader: DataLoader) -> int:
        if self.limit_batches <= 0:
            return len(dataloader)
        return int(self.limit_batches)

    def _update_metrics_storage(self):
        num_optim_params_groups = len(self.module.optimizer.param_groups)
        optim_pg_metrics = [
            {"epoch": self.current_epoch, "step": self.current_step} for _ in range(num_optim_params_groups)
        ]
        for i, param_group in enumerate(self.module.optimizer.param_groups):
            optim_pg_metrics[i]["LR"] = param_group["lr"]
            if "momentum" in param_group:
                optim_pg_metrics[i]["momentum"] = param_group["momentum"]

        for i, metrics in enumerate(optim_pg_metrics):
            self.logger.log_metrics(metrics, self.current_epoch)
            self.metrics_storage_epoch.append(metrics, self.current_step, self.current_epoch, split=f"pg_{i}")

        stages = self.module.stages
        for stage in stages:
            metrics = self.module.compute_metrics(stage=stage)
            self.metrics_storage_epoch.append(metrics, self.current_step, self.current_epoch, stage)
            stage_metrics = {f"{stage}/{name}": value for name, value in metrics.items()}
            self.logger.log_metrics(stage_metrics, self.current_epoch)

    def _log_starting_msg(self) -> None:
        if self.ckpt_path is None:
            msg = f" Starting the training of `{self.module.name}` from scratch "
        else:
            msg = (
                f" Resuming training of `{self.module.name}`: " f"epoch={self.current_epoch}, step={self.current_step} "
            )
        log_breaking_point(msg, num_chars=self.num_char, fill_char="=", n_bottom=1, n_top=1)

    def on_fit_start(self) -> None:
        log.info("Running on_fit_start in trainer")
        self.callbacks.on_fit_start(self)

    def on_epoch_start(self) -> None:
        log_breaking_point(
            f" Starting epoch {self.current_epoch + 1}/{self.max_epochs} ", num_chars=self.num_char, fill_char="="
        )
        self.module.on_epoch_start()
        self.callbacks.on_epoch_start(self)

    def on_epoch_end(self) -> None:
        log_breaking_point(
            f"Finished epoch {self.current_epoch + 1}/{self.max_epochs}",
            num_chars=self.num_char,
            fill_char="=",
            n_bottom=1,
        )
        self.module.on_epoch_end()
        self.callbacks.on_epoch_end(self)
        self.random_val_results.clear()
        self.val_targets.clear()
        self.val_preds.clear()

    def load_checkpoint(self, ckpt_path: str | None) -> None:
        if ckpt_path is None:
            log.warning("No checkpoint found to load in the `Trainer` class, `ckpt_path` is None")
            return

        log.info(f"Loading checkpoint from '{ckpt_path}'")
        ckpt_state = torch.load(ckpt_path)

        # check if the checkpoint is compatible with the current module
        assert ckpt_state.get("module") is not None, "Checkpoint does not contain `module` state dict"
        assert ckpt_state.get("datamodule") is not None, "Checkpoint does not contain `datamodule` state dict"
        assert ckpt_state.get("callbacks") is not None, "Checkpoint does not contain `callbacks` state dict"
        assert ckpt_state.get("metrics") is not None, "Checkpoint does not contain `metrics` state dict"
        assert ckpt_state.get("epoch") is not None, "Checkpoint does not contain `epoch`"
        assert ckpt_state.get("step") is not None, "Checkpoint does not contain `step`"

        self.module.load_state_dict(ckpt_state["module"])
        self.datamodule.load_state_dict(ckpt_state["datamodule"])
        self.callbacks.load_state_dict(ckpt_state["callbacks"])
        self.metrics_storage_epoch.load_state_dict(ckpt_state["metrics"]["metrics_storage_epoch"])
        self.current_epoch = ckpt_state["epoch"] + 1
        self.current_step = ckpt_state["step"]
        log.info(f"Loaded checkpoint in the `Trainer` class (epoch={self.current_epoch}, step={self.current_step})")

    def save_checkpoint(self, ckpt_path: str) -> None:
        self.module.model.eval()
        epoch, step = self.current_epoch, self.current_step
        ckpt_name = Path(ckpt_path).stem
        log.info(f"Saving the `{ckpt_name}` checkpoint (epoch={epoch}, step={step}) to '{relpath(ckpt_path)}'")
        module_state = self.module.state_dict()
        datamodule_state = self.datamodule.state_dict()
        callbacks_state = self.callbacks.state_dict()
        metrics_state = {
            "metrics_storage_epoch": self.metrics_storage_epoch.state_dict(),
        }

        ckpt_state = {
            "module": module_state,
            "datamodule": datamodule_state,
            "callbacks": callbacks_state,
            "metrics": metrics_state,
            "epoch": epoch,
            "step": step,
        }

        torch.save(ckpt_state, ckpt_path)
