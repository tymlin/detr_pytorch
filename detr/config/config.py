from dataclasses import dataclass, fields
from typing import Any

from detr.utils.contants import NOW
from detr.utils.types import (
    Accelerator,
    DatasetType,
    LRSchedulerInterval,
    Split,
)

from .utils import AbstractConfig


@dataclass
class CustomParamsConfig(AbstractConfig):
    name: str
    params: dict

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    def create_log_parameters(self, prefix: str | None) -> dict[str, Any]:
        prefix = f"{prefix}_" if prefix is not None else ""
        name_param = {f"{prefix}name": self.name}
        return {f"{prefix}{k}": v for k, v in self.params.items()} | name_param


@dataclass
class SetupConfig(AbstractConfig):
    seed: int
    experiment_name: str
    run_name: str
    ckpt_path: str | None

    def __post_init__(self):
        self.timestamp = NOW
        self.run_name = f"{self.timestamp}_{self.run_name}"


@dataclass
class LoggerConfig(CustomParamsConfig):
    description: str


@dataclass
class TrainerConfig(AbstractConfig):
    accelerator: Accelerator
    max_epochs: int
    limit_batches: int
    default_device_id: int = 0


@dataclass
class EvaluatorConfig(AbstractConfig):
    accelerator: Accelerator
    limit_batches: int
    default_device_id: int = 0


@dataclass
class BackboneConfig(AbstractConfig):
    name: str
    num_channels: int
    train_backbone: bool
    pretrained: bool
    return_interm_layers: bool
    dilation: bool

    def create_log_parameters(self, prefix: str | None) -> dict[str, Any]:
        prefix = f"{prefix}_" if prefix is not None else ""
        params = {f"{prefix}{f.name}": getattr(self, f.name) for f in fields(self)}
        return params


@dataclass
class TransformerConfig(AbstractConfig):
    hidden_dim: int
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    pre_norm: bool
    return_intermediate_dec: bool

    def create_log_parameters(self, prefix: str | None) -> dict[str, Any]:
        prefix = f"{prefix}_" if prefix is not None else ""
        params = {f"{prefix}{f.name}": getattr(self, f.name) for f in fields(self)}
        return params


@dataclass
class ModelConfig(AbstractConfig):
    name: str
    num_classes: int
    num_queries: int
    aux_loss: bool
    backbone: BackboneConfig
    transformer: TransformerConfig

    def create_log_parameters(self, prefix: str | None) -> dict[str, Any]:
        prefix = f"{prefix}_" if prefix is not None else ""
        prefix_backbone = f"{prefix}backbone" if prefix is not None else "backbone"
        prefix_transformer = f"{prefix}transformer" if prefix is not None else "transformer"
        backbone_params = self.backbone.create_log_parameters(prefix_backbone)
        transformer_params = self.transformer.create_log_parameters(prefix_transformer)
        params_ = {
            f"{prefix}{f.name}": getattr(self, f.name)
            for f in fields(self)
            if f.name not in ["backbone", "transformer"]
        }
        params = {
            **params_,
            **backbone_params,
            **transformer_params,
        }
        return params


@dataclass
class DataloaderConfig(AbstractConfig):
    dataset: DatasetType
    split: Split | None
    dataset_args: dict[str, Any]
    batch_size: int
    pin_memory: bool
    num_workers: int
    shuffle: bool
    drop_last: bool


@dataclass
class OptimizerConfig(CustomParamsConfig):
    # params_additional: dict[str, Any]
    lr_backbone: float | None

    def create_log_parameters(self, prefix: str | None) -> dict[str, Any]:
        params = super().create_log_parameters(prefix)
        prefix = f"{prefix}_" if prefix is not None else ""
        # params_add = {f"{prefix}{k}": v for k, v in self.params_additional.items()}
        params_add = {
            f"{prefix}lr_backbone": self.lr_backbone,
        }
        return params | params_add


@dataclass
class LossConfig(CustomParamsConfig):
    pass


@dataclass
class LRSchedulerConfig(CustomParamsConfig):
    interval: LRSchedulerInterval


MetricsConfig = list[dict]

CallbacksConfig = list[dict]


@dataclass
class Config(AbstractConfig):
    setup: SetupConfig
    logger: LoggerConfig
    trainer: TrainerConfig
    model: ModelConfig
    dataloader: DataloaderConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    lr_scheduler: LRSchedulerConfig
    metrics: MetricsConfig
    callbacks: CallbacksConfig

    def create_log_parameters(self) -> dict[str, Any]:
        log_params = {
            "seed": self.setup.seed,
            "max_epochs": self.trainer.max_epochs,
            "batch_size": self.dataloader.batch_size,
            "num_workers": self.dataloader.num_workers,
            "pin_memory": self.dataloader.pin_memory,
            **self.model.create_log_parameters("model"),
            **self.optimizer.create_log_parameters("optimizer"),
            **self.loss.create_log_parameters("loss"),
            **self.lr_scheduler.create_log_parameters("lr_scheduler"),
        }
        return log_params


@dataclass
class ConfigEval(AbstractConfig):
    setup: SetupConfig
    logger: LoggerConfig
    evaluator: EvaluatorConfig
    model: ModelConfig
    dataloader: DataloaderConfig
    loss: LossConfig
    metrics: MetricsConfig
    callbacks: CallbacksConfig

    def create_log_parameters(self) -> dict[str, Any]:
        log_params = {
            "seed": self.setup.seed,
            "batch_size": self.dataloader.batch_size,
            "num_workers": self.dataloader.num_workers,
            "pin_memory": self.dataloader.pin_memory,
            **self.loss.create_log_parameters("loss"),
            **self.model.create_log_parameters("model"),
        }
        return log_params
