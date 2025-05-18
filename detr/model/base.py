from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torchinfo import summary

from detr.utils.torch_utils import model_info


class BaseModel(nn.Module, ABC):
    input_names: list[str]
    example_input_shapes: list[tuple[int, ...]]

    @property
    def name(self):
        return self.__class__.__name__

    def summary_tab(self, depth: int = 4) -> str:
        col_names = ["input_size", "output_size", "num_params", "params_percent"]
        model_summary = str(
            summary(
                self,
                input_size=self.example_input_shapes,
                depth=depth,
                col_names=col_names,
                verbose=0,
                device=self.device,
            )
        )
        return model_summary

    def summary_ops(self) -> str:
        txt_summary, *_ = model_info(self, input_sizes=self.example_input_shapes)
        return txt_summary

    @classmethod
    def example_input(
        cls, name: str, shape: tuple[int, ...], device: str | torch.device, dtype: torch.dtype
    ) -> dict[str, Tensor]:
        return {name: torch.randn(*shape, device=device, dtype=dtype)}

    @classmethod
    @abstractmethod
    def example_inputs(cls, device: str | torch.device) -> dict[str, Tensor]:
        pass

    def summary(self, depth: int = 4) -> str:
        summary_tab = self.summary_tab(depth=depth)
        summary_ops = self.summary_ops()
        summary_net_pt = f"{self.name}: {str(self)}"
        sep = "\n\n"
        summary = f"{summary_ops}{sep}{summary_tab}{sep}{summary_net_pt}"
        return summary

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
