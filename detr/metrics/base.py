import copy
from typing import Any, Type

import numpy as np
import torch
import torchmetrics
from torch import Tensor

from detr.losses import LOSSES
from detr.utils.registry import Registry, create_register_decorator

METRICS = Registry()
register_metric = create_register_decorator(METRICS)


def extend_metrics_registry_with_default_namespace():
    from torchmetrics.detection import __all__ as torch_detection_metrics

    for metric_name in torch_detection_metrics:
        metric_cls = getattr(torchmetrics.detection, metric_name)
        register_metric(metric_cls)


def find_metric_in_default_namespace(metric_name: str) -> Type:
    metric_cls = getattr(torchmetrics.detection, metric_name)
    return metric_cls


extend_metrics_registry_with_default_namespace()


class Metrics:
    def __init__(self, metrics: list[torchmetrics.Metric] | None = None) -> None:
        if metrics is None:
            metrics = {}
        self.metrics = {self.create_metric_name(metric): metric for metric in metrics}

    def __getitem__(self, metric: str) -> torchmetrics.Metric:
        return self.metrics[metric]

    def __len__(self) -> int:
        return len(self.metrics)

    def reset(self) -> None:
        for name in self.metrics:
            self.metrics[name].reset()

    def compute(self, to_cpu: bool = False, as_tensor: bool = True) -> dict[str, Tensor | float | np.ndarray]:
        """Compute metrics and return them as dictionary.

        :param: to_cpu (bool): whether to move tensors to cpu
        :param: as_tensor (bool): whether to return metrics as tensors, float or numpy arrays
                                  if `as_tensor` is False, then metrics will be moved to cpu
                                  and converted to float or numpy array
        :return (dict): dictionary with computed metrics
        """
        computed_metrics = {}
        for name in self.metrics:
            metric_value = self.metrics[name].compute()
            if to_cpu:
                if isinstance(metric_value, (list, tuple)):
                    metric_value = [m.cpu() for m in metric_value]
                elif isinstance(metric_value, dict):
                    metric_value = {k: v.cpu() for k, v in metric_value.items()}
                else:
                    metric_value = metric_value.cpu()
            if not as_tensor:
                # make sure that tensor was moved to cpu, not necessary fot .item()
                if isinstance(metric_value, (list, tuple)):
                    metric_value = [m.item() if m.ndim == 0 else m.cpu().numpy() for m in metric_value]
                elif isinstance(metric_value, dict):
                    metric_value = {k: v.item() if v.ndim == 0 else v.cpu().numpy() for k, v in metric_value.items()}
                else:
                    metric_value = metric_value.item() if metric_value.ndim == 0 else metric_value.cpu().numpy()
            computed_metrics[name] = metric_value
        return computed_metrics

    def update(self, preds: Tensor, targets: Tensor, **kwargs) -> None:
        for name in self.metrics:
            self.metrics[name].update(preds, targets, **kwargs)

    def to(self, device: str) -> None:
        for name in self.metrics:
            self.metrics[name].to(device)

    def state_dict(self) -> dict[str, Any]:
        state_dict = {}
        for name in self.metrics:
            state_dict[name] = self.metrics[name].state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for name in self.metrics:
            self.metrics[name].load_state_dict(state_dict[name])

    @staticmethod
    def create_metric_name(metric: torchmetrics.Metric) -> str:
        names_to_remove = ["Binary", "Multiclass", "MultiLabel", "Metric"]
        name = getattr(metric, "name", metric.__class__.__name__)
        for name_to_remove in names_to_remove:
            name = name.replace(name_to_remove, "")
        name = name.lower()
        average = getattr(metric, "average", "")
        if average:
            name = f"{name}_avg-{average}"
        return name

    def __str__(self) -> str:
        metrics_summary = ""
        for name in self.metrics:
            metrics_summary += str(self.metrics[name]) + "\n"
        return metrics_summary

    def summary(self) -> str:
        return self.__str__()


# Custom metric, e.g.:
#
# @register_metric
# class CustomMetric:
#     ...
# https://lightning.ai/docs/torchmetrics/stable/pages/implement.html


@register_metric
class LossMetric(torchmetrics.Metric):
    """Computes the average loss over a batches."""

    name: str = "Loss"

    def __init__(self, loss_fn: str, loss_fn_params: dict[Any], **kwargs):
        super().__init__(**kwargs)
        loss_cls = LOSSES[loss_fn]
        self.loss_fn = loss_cls(**loss_fn_params)  # default params
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, **kwargs) -> None:
        # loss for batch of data
        batch_loss = self.loss_fn(preds, target)
        batch_size = target.size(0)

        self.sum_loss += batch_loss * batch_size
        self.total += batch_size

    def compute(self) -> Tensor:
        return self.sum_loss / self.total


@register_metric
class DETRLossMetric(LossMetric):
    name: str = "DETRLoss"

    def update(
        self, preds: dict[str, Tensor | list[dict[str, Tensor]]], target: list[dict[str, Tensor]], **kwargs
    ) -> None:
        batch_size = preds[(list(preds.keys())[0])].shape[0]
        batch_losses_dict = self.loss_fn(preds, target)
        weight_dict = self.loss_fn.weight_dict
        batch_loss = sum(batch_losses_dict[k] * weight_dict[k] for k in batch_losses_dict if k in weight_dict)

        self.sum_loss += batch_loss * batch_size
        self.total += batch_size


@register_metric
class DETRMeanAveragePrecision(torchmetrics.detection.MeanAveragePrecision):
    """Computes the mean average precision (mAP) for object detection tasks.

    Based on torchmetrics.detection.MeanAveragePrecision.
    """

    name: str = "MeanAveragePrecision"

    def update(self, preds, target, **kwargs) -> None:
        out = kwargs.get("output_full")
        if out is None:
            raise ValueError("Output full is None")
        preds_ = copy.deepcopy(out.outputs_postprocessed)
        target_ = copy.deepcopy(out.targets_inv)
        super().update(preds_, target_)
