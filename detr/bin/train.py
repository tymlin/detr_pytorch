import sys
import time

from detr.config import Config
from detr.loggers import Status, log
from detr.utils.contants import CONFIG_TRAINING_DIRPATH
from detr.utils.files import relpath
from detr.utils.torch_utils import seed_everything

from .utils import create_callbacks, create_datamodule, create_logger, create_metrics, create_module, create_trainer


def train() -> None:
    if len(sys.argv) < 2:
        log.warning("No experiment config provided. Using default config from 'default.yaml' file.")
        experiment_name_yaml = "default.yaml"
    else:
        experiment_name_yaml = sys.argv[1]

    config_path = CONFIG_TRAINING_DIRPATH / experiment_name_yaml
    log.critical(f"Loading experiment config from '{relpath(config_path)}'.")
    time.sleep(2)  # wait so that the user can see the log message
    cfg = Config.from_yaml(config_path)

    seed_everything(cfg.setup.seed)

    logger = create_logger(cfg)
    datamodule = create_datamodule(cfg.dataloader)
    metrics = create_metrics(cfg.metrics)
    callbacks = create_callbacks(cfg.callbacks)
    module = create_module(
        model_cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        loss_cfg=cfg.loss,
        lr_scheduler_cfg=cfg.lr_scheduler,
        metrics=metrics,
    )
    trainer = create_trainer(
        trainer_cfg=cfg.trainer,
        module=module,
        datamodule=datamodule,
        callbacks=callbacks,
        logger=logger,
        ckpt_path=cfg.setup.ckpt_path,
    )

    try:
        trainer.fit()
    except Exception as e:
        log.exception(e)
        logger.finalize(Status.FAILED)
        raise e
