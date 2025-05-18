# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""DETR model and criterion classes."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from detr.data.utils import NestedTensor, nested_tensor_from_tensor_list

from .base import BaseModel


class DETR(BaseModel):
    """DETR module that performs object detection"""

    input_names: list[str] = ["images"]
    example_input_shapes: list[tuple] = [(1, 3, 768, 1151)]

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """Initializes the model.

        :param backbone: backbone network
        :param transformer: transformer architecture
        :param num_classes: number of object classes
        :param num_queries: number of object queries
        :param aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """The forward pass.

        Expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": (Tensor) the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": (Tensor) The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": (list[dict[str, Tensor]]) Optional, only returned when auxilary losses are activated.
                            It is a list of dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def example_inputs(cls, device: str | torch.device, dtype: torch.dtype) -> list[Tensor]:
        shapes = cls.example_input_shapes
        inputs = []
        for shape in shapes:
            bs = shape[0]
            img_size = shape[1:]
            input_ = []
            for _ in range(bs):
                input_.append(torch.randn(*img_size, device=device, dtype=dtype))
            inputs.append(input_)
        return inputs

    def dummy_inputs(
        cls, shapes: tuple[int, int, int, int], device: str | torch.device, dtype: torch.dtype
    ) -> list[Tensor]:
        inputs = []
        for shape in shapes:
            bs = shape[0]
            img_size = shape[1:]
            input_ = []
            for _ in range(bs):
                input_.append(torch.randn(*img_size, device=device, dtype=dtype))
            inputs.append(input_)
        return inputs


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
