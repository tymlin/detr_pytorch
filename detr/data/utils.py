from typing import Optional

import numpy as np
import torch
import torchvision
from torch import Tensor


def _empty_like(x: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if isinstance(x, np.ndarray):
        return np.empty_like(x)
    else:
        return torch.empty_like(x)


def box_xyxy2cxcywh(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    boxes_ = _empty_like(boxes)
    xmin, ymin = boxes[..., 0], boxes[..., 1]
    xmax, ymax = boxes[..., 2], boxes[..., 3]
    boxes_[..., 0] = (xmin + xmax) / 2  # xc
    boxes_[..., 1] = (ymin + ymax) / 2  # yc
    boxes_[..., 2] = xmax - xmin  # w
    boxes_[..., 3] = ymax - ymin  # h
    return boxes_


def box_cxcywh2xyxy(boxes: np.ndarray | Tensor) -> np.ndarray | Tensor:
    boxes_ = _empty_like(boxes)
    xy = boxes[..., :2]  # centers
    wh = boxes[..., 2:] / 2  # half wh
    boxes_[..., :2] = xy - wh  # xmin, ymin
    boxes_[..., 2:] = xy + wh  # xmax, ymax
    return boxes_


def interpolate(
    input,
    size: Optional[list[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """Equivalent to nn.functional.interpolate, but with support for empty batch sizes.

    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


class NestedTensor:
    def __init__(self, tensors, mask: Tensor | np.ndarray | None):
        self.tensors: Tensor | np.ndarray = tensors
        self.mask: Tensor | np.ndarray = mask

    def to(self, device: str) -> "NestedTensor":
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        cast_mask = mask.to(device) if mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def to_numpy(self) -> "NestedTensor":
        assert self.tensors.ndim == 4, "tensors must be 4D"
        tensors = self.tensors.cpu().numpy().transpose(0, 2, 3, 1)
        mask = self.mask.cpu().numpy() if self.mask is not None else None
        return NestedTensor(tensors, mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list: list[list[int]]) -> list[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: list[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError(f"not supported, input: {tensor_list}, type: {type(tensor_list)}")
    return NestedTensor(tensor, mask)
