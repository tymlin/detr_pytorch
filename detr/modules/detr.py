from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from detr.data.utils import NestedTensor
from detr.metrics import Metrics
from detr.model.postprocess import PostProcess
from detr.schedulers import LRSchedulerWrapper
from detr.utils.types import ModuleStage

from .base import ModuleStepOut, ObjectDetectionModule


@dataclass
class DETRModuleStepOut(ModuleStepOut):
    inputs: NestedTensor
    targets: list[dict[str, Tensor]]
    targets_inv: list[dict[str, Tensor]]
    outputs: dict[str, Tensor | list[dict[str, Tensor]]]
    outputs_postprocessed: dict[str, Tensor]

    def to_cpu(self) -> None:
        self.inputs = self.inputs.to("cpu")
        self.inputs_inv = self.inputs_inv.detach().cpu()
        self.targets = [{k: v.cpu() for k, v in t.items()} for t in self.targets]
        self.targets_inv = [{k: v.cpu() for k, v in t.items()} for t in self.targets_inv]
        outputs_ = {}
        for k, v in self.outputs.items():
            if isinstance(v, Tensor):
                outputs_[k] = v.cpu()
            else:
                outputs_[k] = [{kk: vv.cpu() for kk, vv in v.items()} for v in v]
        self.outputs = outputs_
        self.outputs_postprocessed = {k: v.cpu() for k, v in self.outputs_postprocessed.items()}
        self.loss = self.loss.detach().cpu()

    def to_numpy(self) -> None:
        self.inputs = self.inputs.to_numpy()
        if isinstance(self.inputs_inv, list):
            self.inputs_inv = [i.detach().cpu().numpy().transpose(1, 2, 0) for i in self.inputs_inv]
        else:
            self.inputs_inv = self.inputs_inv.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.targets = [{k: v.cpu().numpy() for k, v in t.items()} for t in self.targets]
        self.targets_inv = [{k: v.cpu().numpy() for k, v in t.items()} for t in self.targets_inv]
        outputs_ = {}
        for k, v in self.outputs.items():
            if isinstance(v, Tensor):
                outputs_[k] = v.cpu().numpy()
            else:
                outputs_[k] = [{kk: vv.cpu().numpy() for kk, vv in v.items()} for v in v]
        self.outputs = outputs_
        self.outputs_postprocessed = [{k: v.cpu().numpy() for k, v in o.items()} for o in self.outputs_postprocessed]
        self.loss = self.loss.detach().cpu().numpy()

    def __len__(self) -> int:
        assert len(self.targets) == self.inputs.tensors.shape[0], "Inputs and targets must have the same length"
        return len(self.targets)

    def __getitem__(self, index: int) -> "DETRModuleStepOut":
        output_ = {}
        for k, v in self.outputs.items():
            if isinstance(v, (Tensor, np.ndarray)):
                output_[k] = v[index]
            else:
                output_[k] = [{kk: vv[index] for kk, vv in v.items()} for v in v]
        return DETRModuleStepOut(
            inputs=self.inputs.tensors[index],
            inputs_inv=self.inputs_inv[index],
            targets=self.targets[index],
            targets_inv=self.targets_inv[index],
            outputs=output_,
            outputs_postprocessed=self.outputs_postprocessed[index],
            loss=self.loss if self.loss.ndim == 0 else self.loss[index],
        )


class DETRModule(ObjectDetectionModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None,
        loss_fn: nn.Module,
        lr_scheduler: LRSchedulerWrapper | None,
        metrics: Metrics,
        stages: list[ModuleStage] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            stages=stages,
        )
        self.clip_max_norm: float | None = 0.1
        self.postprocessors = PostProcess()

    def _common_step(
        self, batch: tuple[NestedTensor, list[dict[str, Tensor]]], stage: ModuleStage
    ) -> DETRModuleStepOut:
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        outputs = self.model(inputs)
        loss_dict = self.loss_fn(outputs, targets)
        weight_dict = self.loss_fn.weight_dict
        losses_weighted = [loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict]
        loss = torch.stack(losses_weighted).sum()

        if stage == "train":
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_max_norm is not None and self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
            self.optimizer.step()

        inputs_t = []
        for img, m in zip(inputs.tensors, inputs.mask):
            coords = torch.nonzero(~m, as_tuple=False)
            if coords.numel() == 0:
                inputs_t.append(torch.zeros((img.size(0), 0, 0), dtype=img.dtype))
                continue
            rows, cols = coords[:, 0], coords[:, 1]
            y0, y1 = rows.min().item(), rows.max().item()
            x0, x1 = cols.min().item(), cols.max().item()
            patch = img[:, y0 : (y1 + 1), x0 : (x1 + 1)]
            inputs_t.append(patch)

        inputs_inv, targets_inv = (
            self.inv_transforms(inputs_t, targets) if self.inv_transforms is not None else (inputs.tensors, targets)
        )
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        input_inv_sizes = [
            torch.tensor(inp_inv.shape[1:], dtype=torch.int64, device=self.device) for inp_inv in inputs_inv
        ]
        input_inv_sizes = torch.stack(input_inv_sizes, dim=0)
        outputs_postprocessed = self.postprocessors(outputs, input_inv_sizes)

        step_out = DETRModuleStepOut(
            inputs=inputs,
            inputs_inv=inputs_inv,
            targets=targets,
            targets_inv=targets_inv,
            outputs=outputs,
            outputs_postprocessed=outputs_postprocessed,
            loss=loss,
        )
        return step_out
