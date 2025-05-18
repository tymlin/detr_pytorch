from typing import Any, Type

import torch
from torch.optim.lr_scheduler import LRScheduler

from detr.utils.registry import Registry, create_register_decorator
from detr.utils.types import LRSchedulerInterval

SCHEDULERS = Registry()
register_scheduler = create_register_decorator(SCHEDULERS)


def extend_schedulers_registry_with_default_namespace():
    from torch.optim.lr_scheduler import __all__ as torch_schedulers

    for scheduler_name in torch_schedulers:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        register_scheduler(scheduler_cls)


def find_scheduler_in_default_namespace(scheduler_name: str) -> Type:
    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
    return scheduler_cls


extend_schedulers_registry_with_default_namespace()


class LRSchedulerWrapper:
    def __init__(self, scheduler: LRScheduler, interval: LRSchedulerInterval) -> None:
        """Wraps a PyTorch LR scheduler to add an `interval` attribute.

        This attribute specifies whether to step every "step" or "epoch".

        :param: scheduler: The PyTorch LR scheduler instance to wrap.
        :param: interval (str): Specifies whether to step every "step" or "epoch".
        """
        self.scheduler = scheduler
        self.interval = interval

    def __getattr__(self, name):
        return getattr(self.scheduler, name)

    def step(self, *args, **kwargs):
        return self.scheduler.step(*args, **kwargs)

    @property
    def is_step_interval(self):
        return self.interval == "step"

    @property
    def is_epoch_interval(self):
        return self.interval == "epoch"

    def state_dict(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.scheduler.load_state_dict(state_dict)

    def __str__(self) -> str:
        format_string = self.scheduler.__class__.__name__ + " (\n"
        for key, value in self.scheduler.__dict__.items():
            if key != "optimizer" and not key.startswith("_"):
                format_string += f"\t{key}: {value}\n"
        format_string += f"\tinterval: {self.interval}\n"
        format_string += ")"
        print(format_string)
        return format_string

    def summary(self) -> str:
        return self.__str__()


# Custom scheduler, e.g.:
#
# @register_scheduler
# class CustomScheduler:
#     ...
