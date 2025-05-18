import sys
import time

from detr.config import ConfigEval
from detr.loggers import Status, log
from detr.utils.contants import CONFIG_INFERENCE_DIRPATH
from detr.utils.files import relpath
from detr.utils.torch_utils import seed_everything

from .utils import (
    create_callbacks,
    create_datamodule,
    create_evaluator,
    create_logger,
    create_metrics,
    create_module_evaluation,
)


def eval() -> None:
    if len(sys.argv) < 2:
        log.warning("No experiment config provided. Using default config from 'evaluation.yaml' file.")
        experiment_name_yaml = "evaluation.yaml"
    else:
        experiment_name_yaml = sys.argv[1]

    config_path = CONFIG_INFERENCE_DIRPATH / experiment_name_yaml
    log.critical(f"Loading experiment config from '{relpath(config_path)}'.")
    time.sleep(2)  # wait so that the user can see the log message
    cfg = ConfigEval.from_yaml(config_path)

    seed_everything(cfg.setup.seed)

    logger = create_logger(cfg)
    datamodule = create_datamodule(cfg.dataloader)
    metrics = create_metrics(cfg.metrics)
    callbacks = create_callbacks(cfg.callbacks)
    module = create_module_evaluation(
        model_cfg=cfg.model,
        loss_cfg=cfg.loss,
        metrics=metrics,
    )
    evaluator = create_evaluator(
        evaluator_cfg=cfg.evaluator,
        module=module,
        datamodule=datamodule,
        callbacks=callbacks,
        logger=logger,
        ckpt_path=cfg.setup.ckpt_path,
        split=cfg.dataloader.split,
    )

    try:
        evaluator.eval()
    except Exception as e:
        log.exception(e)
        logger.finalize(Status.FAILED)
        raise e
