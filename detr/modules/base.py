import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchvision.transforms import Compose

from detr.loggers import log
from detr.metrics import Metrics
from detr.schedulers import LRSchedulerWrapper
from detr.utils.contants import MODULE_STAGES
from detr.utils.types import ModuleStage


@dataclass
class ModuleStepOut:
    inputs: Tensor
    inputs_inv: Tensor
    targets: Tensor
    targets_inv: Tensor
    outputs: Tensor
    outputs_postprocessed: Tensor
    loss: Tensor

    def to_cpu(self):
        self.inputs = self.inputs.detach().cpu()
        self.inputs_inv = self.inputs_inv.detach().cpu()
        self.targets = self.targets.detach().cpu()
        self.outputs = self.outputs.detach().cpu()
        self.outputs_postprocessed = self.outputs_postprocessed.detach().cpu()
        self.loss = self.loss.detach().cpu()

    def to_numpy(self):
        self.inputs = self.inputs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.inputs_inv = self.inputs_inv.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.targets = self.targets.detach().cpu().numpy()
        self.outputs = self.outputs.detach().cpu().numpy()
        self.outputs_postprocessed = self.outputs_postprocessed.detach().cpu().numpy()
        self.loss = self.loss.detach().cpu().numpy()

    def __getitem__(self, index: int) -> "ModuleStepOut":
        return ModuleStepOut(
            inputs=self.inputs[index],
            inputs_inv=self.inputs_inv[index],
            targets=self.targets[index],
            targets_inv=self.targets_inv[index],
            outputs=self.outputs[index],
            outputs_postprocessed=self.outputs_postprocessed[index],
            loss=self.loss if self.loss.ndim == 0 else self.loss[index],
        )

    def __len__(self) -> int:
        assert (
            self.inputs.shape[0] == self.inputs_inv.shape[0] == self.targets.shape[0] == self.outputs.shape[0]
        ), "Inputs, targets and outputs must have the same length"
        return self.inputs.shape[0]


class ObjectDetectionModule:
    model: nn.Module
    optimizer: Optimizer
    loss_fn: nn.Module
    lr_scheduler: LRSchedulerWrapper
    metrics: dict[str, Metrics]

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None,
        loss_fn: nn.Module,
        lr_scheduler: LRSchedulerWrapper | None,
        metrics: Metrics,
        stages: list[ModuleStage] | None = None,
    ) -> None:
        self.name = self.__class__.__name__
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.stages: list[ModuleStage] = MODULE_STAGES if stages is None else stages
        self.metrics = {stage: copy.deepcopy(metrics) for stage in self.stages}  # deepcopy object for each stage

        self.device = None
        self.device_id = None
        self.inv_transforms: Compose | None = None
        self.postprocessor: nn.Module | None = None

    def setup(
        self,
        device: str,
        device_id: int,
        ckpt_state: dict[str, Any] | None = None,
        inv_image_transform: Compose | None = None,
    ) -> None:
        """Setup the module with the device and device_id, and load the checkpoint state if provided.

        :param device: (str) device to use, e.g. "cuda" or "cpu"
        :param device_id: (int) device id to use, e.g. 0
        :param ckpt_state: (dict) checkpoint state to load, default is None
                            Note that checkpoint for the module can also be loaded in Trainer class
        :param inv_image_transform: (Compose) inverse image (input) transform to use, default is None
        :return: None
        """
        self.device = device
        self.device_id = device_id
        self.inv_transforms = inv_image_transform

        if "cuda" in self.device:
            log.info(f"Moving model, loss and metrics to CUDA device ({self.device})")
            self.model.cuda(self.device_id)
            self.loss_fn.cuda(self.device_id)
            for stage in self.stages:
                self.metrics[stage].to(self.device)
        else:
            log.info(f"Moving model, loss and metrics to {self.device.upper()} device")
            self.model.to(self.device)
            self.loss_fn.to(self.device)
            for stage in self.stages:
                self.metrics[stage].to(self.device)

        if ckpt_state is not None:
            self.load_state_dict(ckpt_state)

    def _reset_metrics(self) -> None:
        log.debug(f"Resetting metrics in {self.name} module for all stages: {self.stages}")
        for stage in self.stages:
            self.metrics[stage].reset()

    def compute_metrics(self, stage: ModuleStage) -> dict[str, float | np.ndarray]:
        computed_metrics = self.metrics[stage].compute(to_cpu=True, as_tensor=False)

        # check if any of the metrics is a dict, if so, flatten it
        metric_names_to_remove = []
        metrics_flatten = {}
        for name, value in computed_metrics.items():
            if isinstance(value, dict):
                metric_names_to_remove.append(name)
                for k, v in value.items():
                    if isinstance(v, float):  # supporting only float values, not lists, arrays, etc.
                        metrics_flatten[f"{name}--{k}"] = v
        if len(metrics_flatten) > 0:
            computed_metrics.update(metrics_flatten)
            # remove the original metric names with dict values
            for name in metric_names_to_remove:
                del computed_metrics[name]

        return computed_metrics

    def training_step(self, batch) -> ModuleStepOut:
        stage: ModuleStage = "train"
        out = self._common_step(batch, stage=stage)
        metrics_kwargs = {"output_full": out}
        self.metrics[stage].update(out.outputs, out.targets, **metrics_kwargs)
        if self.lr_scheduler is not None and self.lr_scheduler.is_step_interval:
            self.lr_scheduler.step()
        return out

    def validation_step(self, batch) -> ModuleStepOut:
        stage: ModuleStage = "val"
        with torch.no_grad():
            out = self._common_step(batch, stage=stage)
        metrics_kwargs = {"output_full": out}
        self.metrics[stage].update(out.outputs, out.targets, **metrics_kwargs)
        return out

    def _common_step(self, batch, stage: ModuleStage) -> ModuleStepOut:
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        if stage == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        inputs_inv, targets_inv = self.inv_transforms(inputs) if self.inv_transforms is not None else inputs
        outputs_postprocessed = self.postprocessor(outputs) if self.postprocessor is not None else outputs

        step_out = ModuleStepOut(
            inputs=inputs,
            inputs_inv=inputs_inv,
            targets=targets,
            targets_inv=targets_inv,
            outputs=outputs,
            outputs_postprocessed=outputs_postprocessed,
            loss=loss,
        )
        return step_out

    def on_epoch_start(self) -> None:
        self._reset_metrics()

    def on_epoch_end(self) -> None:
        if self.lr_scheduler is not None and self.lr_scheduler.is_epoch_interval:
            self.lr_scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        lr_scheduler_state = self.lr_scheduler.state_dict()
        module_state = {
            "model": model_state,
            "optimizer": optimizer_state,
            "lr_scheduler": lr_scheduler_state,
        }
        return module_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        log.info(f"Loading `{self.name}` state dict.")

        self.model.load_state_dict(state_dict["model"])
        log.info(f"\tLoaded `{self.model.name}` state dict")

        self.optimizer.load_state_dict(state_dict["optimizer"])
        lr = self.optimizer.param_groups[0]["lr"]
        log.info(f"\tLoaded optimizer state with lr = {lr}")

        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        last_epoch = self.lr_scheduler.last_epoch
        log.info(f"\tLoaded lr_scheduler state with last_epoch = {last_epoch}")

        log.info(f"Loaded `{self.name}` state dict")
