from collections import defaultdict
from itertools import groupby

import pandas as pd
import torch

from detr.loggers import log

MetricType = dict[str, list[dict[str, int | float]]]
MetricsType = dict[str, MetricType]


class MetricsStorage:
    metrics: MetricsType

    def __init__(self, name: str = "", metrics: MetricsType | None = None) -> None:
        if metrics is None:
            metrics = defaultdict(lambda: defaultdict(lambda: [], {}))
        self.metrics = metrics
        self.name = name

    @property
    def logged_metrics(self) -> list[str]:
        _logged_metrics = []
        for metric_name, splits_metrics in self.metrics.items():
            for _, values in splits_metrics.items():
                if len(values) > 0:
                    _logged_metrics.append(metric_name)
                    break
        return _logged_metrics

    def clear(self):
        self.metrics = defaultdict(lambda: defaultdict(lambda: [], {}))

    def get(self, metric_name: str, stage: str) -> list[dict]:
        return self.metrics[metric_name][stage]

    def aggregate_over_key(self, key: str) -> "MetricsStorage":
        epochs_metrics = {}
        for metric_name, splits_metrics in self.metrics.items():
            epochs_metrics[metric_name] = {}
            for split_name, logged_values in splits_metrics.items():

                def key_fn(v):
                    return v[key]

                logged_values = sorted(logged_values, key=key_fn)
                key_aggregated_values = []
                for key_step, grouped in groupby(logged_values, key_fn):
                    grouped = list(grouped)
                    values = [el["value"] for el in grouped]
                    avg_value = sum(values) / len(values)
                    agg_values = {key: key_step, "value": avg_value}
                    if key != "epoch":
                        agg_values["epoch"] = grouped[0]["epoch"]
                    key_aggregated_values.append(agg_values)
                epochs_metrics[metric_name][split_name] = key_aggregated_values
        return MetricsStorage(name=key, metrics=epochs_metrics)

    def inverse_nest(self) -> MetricsType:
        inverse_metrics = defaultdict(lambda: defaultdict(lambda: [], {}))
        for metric_name, splits_metrics in self.metrics.items():
            for split_name, values in splits_metrics.items():
                inverse_metrics[split_name][metric_name] = values
        return inverse_metrics

    def __getitem__(self, metric_name: str) -> MetricType:
        return self.metrics[metric_name]

    def append(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            values = {"step": step, "value": value, "epoch": epoch}
            if name not in self.metrics:
                self.metrics[name] = {}
            if split not in self.metrics[name]:
                self.metrics[name][split] = []
            self.metrics[name][split].append(values)

    def to_dict(self) -> dict:
        """For state saving"""
        metrics = {}
        for metric_name, splits_metrics in self.metrics.items():
            metrics[metric_name] = {}
            for split_name, values in splits_metrics.items():
                metrics[metric_name][split_name] = values
        return metrics

    def to_df(self) -> pd.DataFrame:
        """For state saving"""
        metrics = {}
        epochs = []
        for metric_name, splits_metrics in self.metrics.items():
            for split_name, values in splits_metrics.items():
                if metric_name == "epoch":
                    epochs = [v["epoch"] for v in values]
                else:
                    metrics[f"{split_name}/{metric_name}"] = [v["value"] for v in values]
        metrics["epoch"] = epochs
        metrics_df = pd.DataFrame.from_dict(metrics)
        metrics_df.set_index("epoch", inplace=True)
        return metrics_df

    def state_dict(self) -> dict:
        return {"metrics": self.to_dict()}

    def load_state_dict(self, state_dict: dict):
        self.metrics = state_dict["metrics"]
        log.info(f"Loaded `{self.name}` metrics state")
