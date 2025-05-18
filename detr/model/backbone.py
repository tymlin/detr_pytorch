import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from detr.data.utils import NestedTensor

from .position_encoding import PositionEmbeddingSine
from .utils import FrozenBatchNorm2d


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> dict[str, NestedTensor]:
        xs = self.body(tensor_list.tensors)
        out: dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        num_channels: int,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        pretrained: bool = True,
    ):
        backbone_cls = getattr(torchvision.models, name)
        backbone = backbone_cls(
            replace_stride_with_dilation=[False, False, dilation], pretrained=pretrained, norm_layer=FrozenBatchNorm2d
        )
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone: Backbone, position_embedding: PositionEmbeddingSine):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: list[NestedTensor] = []
        pos = []
        for _name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
