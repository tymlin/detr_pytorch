from typing import Type

import torch

from detr.utils.registry import Registry, create_register_decorator

OPTIMIZERS = Registry()
register_optimizer = create_register_decorator(OPTIMIZERS)


def extend_optimizers_registry_with_default_namespace():
    from torch.optim import __all__ as torch_optimizers

    for optimizer_name in torch_optimizers:
        optimizer_cls = getattr(torch.optim, optimizer_name)
        register_optimizer(optimizer_cls)


def find_optimizer_in_default_namespace(optimizer_name: str) -> Type:
    optimizer_cls = getattr(torch.optim, optimizer_name)
    return optimizer_cls


extend_optimizers_registry_with_default_namespace()

# Custom optimizer, e.g.:
#
# @register_optimizer
# class CustomOptimizer:
#     ...
