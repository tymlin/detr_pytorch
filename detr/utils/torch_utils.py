import random
from copy import deepcopy

import numpy as np
import thop
import torch
from ptflops import get_model_complexity_info
from torch import nn

from detr.loggers import log
from detr.utils.types import Accelerator


def seed_everything(seed: int) -> None:
    log.info(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device_and_id(accelerator: Accelerator, default_device_id: int = 0) -> tuple[str, int]:
    available_devices = check_torch_devices_availability()
    log.info(
        "Available devices: "
        + ", ".join(f"`{d}`" for d in available_devices)
        + f"; specified device: `{accelerator}` with id: `{default_device_id}`"
    )
    if accelerator not in available_devices:
        msg = f"Specified device `{accelerator}` is not available"
        log.error(msg)
        raise ValueError(msg)

    if accelerator == "cuda" and torch.cuda.is_available():
        device_id = int(default_device_id)
        device = f"cuda:{device_id}"
    elif accelerator == "mps" and torch.backends.mps.is_available():
        device = "mps"
        device_id = 0
    else:
        device = "cpu"
        device_id = 0

    return device, device_id


def check_torch_devices_availability(as_dict: bool = False) -> dict[str, bool] | list[str]:
    """Checks the availability of CPU, CUDA, and MPS devices in PyTorch.

    as_dict: bool: Whether to return the availability status as a dictionary.

    :return dict: A dictionary with the availability status of each device.
    """
    available_devices = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if not as_dict:
        available_devices = [device for device, status in available_devices.items() if status]
    return available_devices


def get_flops_thop(model: nn.Module, input_sizes: list[tuple[int, ...]]) -> tuple[float, float]:
    """Return models GFLOPs."""
    p = next(model.parameters())
    inputs = model.dummy_inputs(input_sizes, p.device, p.dtype)
    macs, params = thop.profile(deepcopy(model), inputs=inputs, verbose=False)

    # roughly 2 FLOPs per MAC (MAC operation typically consists of one multiplication and one addition)
    flops = macs * 2

    # follow convention from literature where
    # Floating Point Operations = Multiply-Add Cumulation Multiply-Accumulate Operations
    # flops = macs
    return flops, macs


def get_flops_ptflops(model: nn.Module, input_sizes: list[tuple[int, ...]]) -> tuple[float, float]:
    input_size = input_sizes[0]  # only one input size is supported
    if len(input_size) > 3:
        input_size = input_size[1:]
    macs, params = get_model_complexity_info(
        model, input_size, as_strings=False, backend="aten", print_per_layer_stat=False, verbose=False
    )
    flops = macs * 2
    return flops, macs


def get_num_params(model: nn.Module) -> int:
    """Return the total number of models parameters."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model: nn.Module) -> int:
    """Return the number of models parameters which require grad."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_model_size(model: nn.Module, as_megabytes: bool = True) -> float:
    """Return the total size of models parameters."""
    size_model = 0.0
    for param in model.parameters():
        if param.is_floating_point():
            size_model += param.numel() * torch.finfo(param.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.dtype).bits
    if as_megabytes:
        size_model = size_model / 8e6
    return size_model


def model_info(
    model: nn.Module,
    input_sizes: list[tuple[int, ...]],
    detailed: bool = False,
    as_dict: bool = False,
) -> tuple[str, int, int, int, str, str] | dict[str, int | str]:
    num_params = get_num_params(model)
    num_gradients = get_num_gradients(model)
    model_size = get_model_size(model, as_megabytes=True)
    num_layers = len(list(model.modules()))
    info = ""
    if detailed:
        info += (
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} " f"{'shape':>20} {'mu':>10} {'sigma':>10}\n"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            info += "%5g %40s %9s %12g %20s %10.3g %10.3g %10s \n" % (
                i,
                name,
                p.requires_grad,
                p.numel(),
                list(p.shape),
                p.mean(),
                p.std(),
                p.dtype,
            )
    flops, macs = get_flops_thop(model, input_sizes)
    # flops, macs = get_flops_ptflops(model, input_sizes)
    giga_flops = flops / 1e9
    giga_macs = macs / 1e9
    name = f"{model.__class__.__name__}"
    layers = f"{num_layers} layers"
    params = f"{num_params} parameters"
    gradients = f"{num_gradients} gradients"
    flops_str = f"{giga_flops:.4f} GFLOPs (thop)" if giga_flops else ""
    macs_str = f"{giga_macs:.4f} GMACs (thop)" if giga_macs else ""
    model_size_str = f"{model_size:.2f} MB"
    info += f"{name}, summary: {layers}, {params}, {gradients}, {flops_str}, {macs_str}, {model_size_str}"

    if as_dict:
        output = {
            "info": info,
            "num_layers": num_layers,
            "num_params": num_params,
            "num_gradients": num_gradients,
            "model_size_str": model_size_str,
            "model_size": model_size,
            "flops_str": flops_str,
            "macs_str": macs_str,
            "flops": flops,
            "macs": macs,
        }
    else:
        output = (info, num_layers, num_params, num_gradients, flops_str, model_size_str)
    return output
