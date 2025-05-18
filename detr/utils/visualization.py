import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from detr.metrics import MetricsStorage
from detr.modules import ModuleStepOut
from detr.utils.contants import MODULE_STAGES

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

COLORS = [
    "cornflowerblue",
    "limegreen",
    "mediumpurple",
    "navy",
    "orange",
    "firebrick",
    "royalblue",
    "gold",
    "lightcoral",
]


def colors_custom(i: int, reversed_: bool = False) -> str:
    if reversed_:
        return COLORS[::-1][i % len(COLORS)]
    return COLORS[i % len(COLORS)]


def colors_custom_reversed(i: int) -> str:
    return colors_custom(i, reversed_=True)


def plot_metrics_matplotlib(
    metrics_storage: MetricsStorage,
    step_name: Literal["step", "epoch"],
    filepath: str | None,
    classes_names: list[str] | None = None,
    # ncols: int = 4,
) -> None:
    """Plot metrics for each split and step"""
    sns.set_style("whitegrid")
    _metrics = metrics_storage.metrics
    metrics = _metrics.copy()
    metrics.pop("epoch")
    metrics.pop("step")

    palette = colors_custom  # plt.get_cmap("Set2")
    palette_2 = colors_custom_reversed  # plt.get_cmap("Paired")

    colors = {stage: palette(i) for i, stage in enumerate(MODULE_STAGES)}
    _metrics = {}
    for name, split_values in metrics.items():
        for stage, values in split_values.items():
            if "momentum" in name:
                continue
            key = str(name)
            nested_key = str(stage)
            if key not in _metrics:
                _metrics[key] = {nested_key: values}
            else:
                _metrics[key][nested_key] = values

    # num_splits = 2 # assume train and val splits # TODO auto calculate
    fig, axes = plt.subplots(1, len(_metrics), figsize=(10 * len(_metrics), 7 * 1))
    for (metric_name, split_values), ax in zip(_metrics.items(), axes.flatten()):
        title = f"{metric_name}"
        for i, (split, logged_values) in enumerate(split_values.items()):
            steps = [logged[step_name] for logged in logged_values]
            values = [logged["value"] for logged in logged_values]
            color = colors[split] if split in colors else palette_2(i)
            if split == "train":
                line_style = "dashdot"
                line_width = 1.5
            else:
                line_style = "solid"
                line_width = 2
            ax.plot(steps, values, label=split, color=color, linestyle=line_style, linewidth=line_width)
            if len(values) < 2:
                ax.scatter(steps, values, color=color)
            ax.legend()
            last_value = f"{values[-1]:.4f}"
            title += f" ({split}: {last_value})"
            ax.set_title(title, fontsize=20)
            ax.set_xlim(left=steps[0], right=steps[-1])
            ax.set_xlabel(metrics_storage.name, fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=12)

    if filepath is not None:
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        fig.savefig(filepath, dpi=200)
    plt.close()


def plot_images_bbox(
    image: np.ndarray,
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None,
    classes_int2str: dict[int, str],
    colors: dict[int, str] | None = None,
    keep_prob: float = 0.9,
) -> None:
    keep = scores > keep_prob if scores is not None else np.ones(len(bboxes), dtype=bool)
    bboxes_ = bboxes[keep]
    labels_ = labels[keep]
    scores_ = scores[keep] if scores is not None else None
    plt.imshow(image)
    plt.axis("off")
    plt.grid(False)
    for ix, (bbox, label) in enumerate(zip(bboxes_, labels_)):
        color = colors[label] if colors is not None else "blue"
        x0, y0, x1, y1 = bbox
        if scores_ is not None:
            score = scores_[ix]
            score_txt = f" {score:.2%}"
        else:
            score_txt = ""
        plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, linewidth=2))
        plt.text(
            x0,
            y0,
            f"{classes_int2str[label]}{score_txt}",
            fontsize=12,
            color=color,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor=color, boxstyle="round,pad=0.3"),
        )


def plot_results(
    results: list[ModuleStepOut],
    classes_int2str: dict[int, str],
    colors: dict[int, str] | None = None,
    filepath: str | None = None,
    ncols: int = 5,
) -> None:
    nrows = math.ceil(len(results) / ncols)
    plt.figure(figsize=(14 * ncols, 6 * nrows))
    num_images = len(results)
    keep_prob = 0.9
    for i in range(num_images):
        image = results[i].inputs_inv
        bboxes_pred = results[i].outputs_postprocessed["boxes"]
        labels_pred = results[i].outputs_postprocessed["labels"]
        scores_pred = results[i].outputs_postprocessed["scores"]
        bboxes_target = results[i].targets_inv["boxes"]
        labels_target = results[i].targets_inv["labels"]
        plt.subplot(nrows, 2 * ncols, 2 * i + 1)
        plot_images_bbox(image, bboxes_pred, labels_pred, scores_pred, classes_int2str, colors, keep_prob=keep_prob)
        plt.title(f"Predictions (keep_prob={keep_prob:.0%})")
        plt.subplot(nrows, 2 * ncols, 2 * i + 2)
        plot_images_bbox(image, bboxes_target, labels_target, None, classes_int2str, colors, keep_prob=keep_prob)
        plt.title("Targets")
    plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, dpi=200)
    plt.close()
