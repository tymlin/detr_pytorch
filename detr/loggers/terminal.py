"""Terminal Logger."""

from typing import Any

from .base import BaseLogger, register_logger
from .pylogger import log


@register_logger
class TerminalLogger(BaseLogger):
    """Terminal Logger (only prints info and saves locally)."""

    name: str = "terminal"

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        super().log_metric(key, value, step)
        log.info(f"Step {step}, {key}: {value}")

    def log_params(self, params: dict[str, Any]) -> None:
        super().log_params(params)
        log.info(f"Params: {params}")
