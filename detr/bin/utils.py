import json
from pathlib import Path

from torch import nn

from detr.callbacks import CALLBACKS, Callbacks
from detr.config import (
    CallbacksConfig,
    Config,
    ConfigEval,
    DataloaderConfig,
    EvaluatorConfig,
    LossConfig,
    LRSchedulerConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
    parse_list_dct_cfgs,
)
from detr.data import (
    DATASETS,
    Compose,
    DataModule,
    create_inverse_normalize_transforms,
    create_train_transforms,
    create_val_transforms,
)
from detr.evaluator import Evaluator
from detr.loggers import LOGGERS, BaseLogger, add_file_handler_to_logger, log
from detr.losses import LOSSES
from detr.metrics import METRICS, Metrics
from detr.model import DETR, Backbone, Joiner, PositionEmbeddingSine, Transformer
from detr.modules import DETRModule, ObjectDetectionModule
from detr.optimizers import OPTIMIZERS
from detr.schedulers import SCHEDULERS, LRSchedulerWrapper
from detr.trainer import Trainer
from detr.utils.contants import LOGGER_FILE_NAME, RUNS_DIRPATH
from detr.utils.types import Split


def create_logger(config: Config | ConfigEval) -> BaseLogger:
    log_path = RUNS_DIRPATH / config.setup.experiment_name / config.setup.run_name
    logger_filepath = log_path / "logs" / LOGGER_FILE_NAME
    Path(logger_filepath.parent).mkdir(exist_ok=True, parents=True)

    if hasattr(config, "trainer"):
        device = config.trainer.accelerator
        device_id = config.trainer.default_device_id
    elif hasattr(config, "evaluator"):
        device = config.evaluator.accelerator
        device_id = config.evaluator.default_device_id
    else:
        raise ValueError("Config must have either 'trainer' or 'evaluator' attribute.")

    file_log = add_file_handler_to_logger(logger_filepath, device=device, device_id=device_id)
    logger_cls = LOGGERS[config.logger.name]

    params = {
        **config.create_log_parameters(),
    }
    logger = logger_cls(
        log_path=log_path,
        file_log=file_log,
        experiment_name=config.setup.experiment_name,
        run_name=config.setup.run_name,
        config=config.to_dict(),
        description=config.logger.description,
        params=params,
        **config.logger.params,
    )
    logger.start_run()
    logger.log_config()

    return logger


def create_transforms() -> tuple[Compose, Compose, Compose]:
    train_transforms = create_train_transforms()
    log.info(f"Initialized train transforms:\n{train_transforms}")
    val_transforms = create_val_transforms()
    log.info(f"Initialized val transforms:\n{val_transforms}")
    inverse_norm_transforms = create_inverse_normalize_transforms()
    return train_transforms, val_transforms, inverse_norm_transforms


def create_datamodule(dataloader_cfg: DataloaderConfig) -> DataModule:
    config_repr = json.dumps(dataloader_cfg.to_dict(), indent=4)
    log.info(f"Initializing datamodule with parameters: \n {config_repr}")

    train_transform, val_transform, inverse_norm_transforms = create_transforms()

    dataset_cls = DATASETS[dataloader_cfg.dataset]
    log.info(
        f"Building dataset `{dataset_cls.__name__}` with parameters:\n"
        f"{json.dumps(dataloader_cfg.dataset_args, indent=4)}"
    )
    if dataloader_cfg.split is not None:
        dataset = dataset_cls(split=dataloader_cfg.split, transforms=val_transform, **dataloader_cfg.dataset_args)
        datamodule = DataModule(
            train_ds=dataset if dataloader_cfg.split == "train" else None,
            val_ds=dataset if dataloader_cfg.split == "val" else None,
            test_ds=dataset if dataloader_cfg.split == "test" else None,
            inverse_transforms=inverse_norm_transforms,
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            shuffle=dataloader_cfg.shuffle,
            drop_last=dataloader_cfg.drop_last,
            collate_fn_train=dataset.__class__.collate_fn if dataloader_cfg.split == "train" else None,
            collate_fn_val=dataset.__class__.collate_fn if dataloader_cfg.split == "val" else None,
            collate_fn_test=dataset.__class__.collate_fn if dataloader_cfg.split == "test" else None,
        )
    else:
        train_ds = dataset_cls(split="train", transforms=train_transform, **dataloader_cfg.dataset_args)
        val_ds = dataset_cls(split="val", transforms=val_transform, **dataloader_cfg.dataset_args)
        datamodule = DataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=None,
            inverse_transforms=inverse_norm_transforms,
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            shuffle=dataloader_cfg.shuffle,
            drop_last=dataloader_cfg.drop_last,
            collate_fn_train=train_ds.__class__.collate_fn,
            collate_fn_val=val_ds.__class__.collate_fn,
        )
    return datamodule


def create_backbone(
    model_cfg: ModelConfig,
) -> Joiner:
    n_steps = model_cfg.transformer.hidden_dim // 2
    positional_enc = PositionEmbeddingSine(n_steps, normalize=True)
    backbone_ = Backbone(
        name=model_cfg.backbone.name,
        num_channels=model_cfg.backbone.num_channels,
        train_backbone=model_cfg.backbone.train_backbone,
        pretrained=model_cfg.backbone.pretrained,
        return_interm_layers=model_cfg.backbone.return_interm_layers,
        dilation=model_cfg.backbone.dilation,
    )
    backbone = Joiner(backbone_, positional_enc)
    return backbone


def create_transformer(
    model_cfg: ModelConfig,
) -> Transformer:
    transformer = Transformer(
        d_model=model_cfg.transformer.hidden_dim,
        nhead=model_cfg.transformer.num_heads,
        num_encoder_layers=model_cfg.transformer.num_encoder_layers,
        num_decoder_layers=model_cfg.transformer.num_decoder_layers,
        dim_feedforward=model_cfg.transformer.dim_feedforward,
        dropout=model_cfg.transformer.dropout,
        normalize_before=model_cfg.transformer.pre_norm,
        return_intermediate_dec=model_cfg.transformer.return_intermediate_dec,
    )
    return transformer


def create_model(
    model_cfg: ModelConfig,
) -> DETR:  # now only supporting DETR model
    backbone = create_backbone(model_cfg)
    transformer = create_transformer(model_cfg)
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=model_cfg.num_classes,
        num_queries=model_cfg.num_queries,
        aux_loss=model_cfg.aux_loss,
    )
    return model


def create_module(
    model_cfg: ModelConfig,
    optimizer_cfg: OptimizerConfig,
    loss_cfg: LossConfig,
    lr_scheduler_cfg: LRSchedulerConfig,
    metrics: Metrics,
) -> ObjectDetectionModule:
    model = create_model(model_cfg)

    optimizer_cls = OPTIMIZERS[optimizer_cfg.name]
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": optimizer_cfg.lr_backbone,
        },
    ]
    optimizer = optimizer_cls(param_dicts, **optimizer_cfg.params)

    loss_cls = LOSSES[loss_cfg.name]
    loss_fn = loss_cls(**loss_cfg.params)

    lr_scheduler_cls = SCHEDULERS[lr_scheduler_cfg.name]
    lr_scheduler = lr_scheduler_cls(optimizer, **lr_scheduler_cfg.params)
    lr_scheduler = LRSchedulerWrapper(lr_scheduler, lr_scheduler_cfg.interval)

    # only supports DETRModule
    module = DETRModule(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
    )

    return module


def create_module_evaluation(
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    metrics: Metrics,
) -> ObjectDetectionModule:
    model = create_model(model_cfg)

    loss_cls = LOSSES[loss_cfg.name]
    loss_fn = loss_cls(**loss_cfg.params)

    # only supports DETRModule
    module = DETRModule(
        model=model, optimizer=None, loss_fn=loss_fn, metrics=metrics, lr_scheduler=None, stages=["val"]
    )
    return module


def create_metrics(metrics_cfg: MetricsConfig) -> Metrics:
    log.info("Initializing metrics:")
    metrics_list = parse_list_dct_cfgs(metrics_cfg, METRICS)
    metrics = Metrics(metrics_list)
    return metrics


def create_callbacks(callbacks_cfg: CallbacksConfig) -> Callbacks:
    log.info("Initializing callbacks:")
    callbacks_list = parse_list_dct_cfgs(callbacks_cfg, CALLBACKS)
    callbacks = Callbacks(callbacks_list)
    return callbacks


def create_trainer(
    trainer_cfg: TrainerConfig,
    module: ObjectDetectionModule,
    datamodule: DataModule,
    callbacks: Callbacks,
    logger: BaseLogger,
    ckpt_path: str,
) -> Trainer:
    trainer = Trainer(
        module=module,
        datamodule=datamodule,
        callbacks=callbacks,
        logger=logger,
        ckpt_path=ckpt_path,
        **trainer_cfg.to_dict(),
    )
    return trainer


def create_evaluator(
    evaluator_cfg: EvaluatorConfig,
    module: ObjectDetectionModule,
    datamodule: DataModule,
    callbacks: Callbacks,
    logger: BaseLogger,
    ckpt_path: str,
    split: Split,
) -> Evaluator:
    evaluator = Evaluator(
        module=module,
        datamodule=datamodule,
        callbacks=callbacks,
        logger=logger,
        ckpt_path=ckpt_path,
        split=split,
        **evaluator_cfg.to_dict(),
    )
    return evaluator
