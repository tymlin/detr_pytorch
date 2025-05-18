import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from distinctipy import distinctipy
from PIL import Image
from pycocotools import mask as coco_mask
from torch import Tensor
from torchvision.datasets import VisionDataset

from detr.data.utils import box_cxcywh2xyxy, nested_tensor_from_tensor_list
from detr.utils.contants import DATA_DIRPATH
from detr.utils.registry import Registry, create_register_decorator
from detr.utils.types import Split

DATASETS = Registry()
register_dataset = create_register_decorator(DATASETS)


class Dataset(VisionDataset, ABC):
    name: str
    dataset: VisionDataset
    classes_names = list[str]
    classes_ids = list[int]
    classes_int2str: dict[int, str]
    classes_str2int: dict[str, int]
    num_classes: int
    collate_fn: Optional[Callable] = None

    def __init__(self, transforms: Optional[Callable] = None) -> None:
        super().__init__()
        self.transforms = transforms
        if not DATA_DIRPATH.exists():
            raise FileNotFoundError(
                f"Dataset directory '{DATA_DIRPATH}' does not exist. " f"Please prepare the dataset before usage."
            )

    @abstractmethod
    def _get_image(self, index: int) -> np.ndarray:
        pass

    @abstractmethod
    def _get_target(self, index: int) -> int:
        pass

    @abstractmethod
    def _get_class_names(self) -> list[str]:
        pass

    @abstractmethod
    def _get_class_ids(self) -> list[int]:
        pass

    @abstractmethod
    def _get_dataset_len(self) -> int:
        pass

    def __len__(self) -> int:
        return self._get_dataset_len()

    @abstractmethod
    def __getitem__(self, index: int) -> tuple:
        pass

    def _initialize_classes_dicts(self) -> None:
        self.classes_names = self._get_class_names()
        self.classes_ids = self._get_class_ids()
        self.classes_int2str = {k: v for k, v in zip(self.classes_ids, self.classes_names)}
        self.classes_str2int = {k: v for k, v in zip(self.classes_names, self.classes_ids)}
        self.num_classes = len(self.classes_names)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    masks = torch.stack(masks, dim=0) if masks else torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            num_keypoints = keypoints.shape[0]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes  # this can be transformed
        target["boxes_coco_format"] = torch.as_tensor([obj["bbox"] for obj in anno], dtype=torch.float32).reshape(-1, 4)
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])  # this can be changed in transforms

        return image, target


def coco_ds_collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


class COCODatasetABC(Dataset, ABC):
    dataset_dirpath: Path
    name: str
    annotation_json_filename: str
    collate_fn = coco_ds_collate_fn

    def __init__(self, split: Split, transforms: Optional[Callable] = None, return_masks: bool = False) -> None:
        super().__init__(transforms=transforms)

        if not self.dataset_dirpath.exists():
            raise FileNotFoundError(f"`{self.dataset_dirpath.name}` folder not found at '{DATA_DIRPATH}'")
        images_dirpath = str(self.dataset_dirpath / "images" / f"{split}2017")
        annotation_filename = f"instances_{split}2017.json"
        if self.name == "COCOMiniDetection" and split == "train":
            annotation_filename = "instances_minitrain2017.json"
        annotation_json_filepath = str(self.dataset_dirpath / "annotations" / annotation_filename)
        self.dataset: torchvision.datasets.CocoDetection = torchvision.datasets.CocoDetection(
            root=images_dirpath,
            annFile=annotation_json_filepath,
        )

        self.prepare = ConvertCocoPolysToMask(return_masks=return_masks)
        self._initialize_classes_dicts()

        self.class_colors: list[tuple[float, float, float]] = distinctipy.get_colors(self.num_classes, rng=42)
        self.classes_int2color = {k: v for k, v in zip(self.classes_ids, self.class_colors)}

    def _get_image(self, index: int) -> np.ndarray:
        i, _ = self.dataset[index]
        i = np.asarray(i)
        return i

    def _get_target(self, index: int) -> list[dict[str, Any]]:
        _, t = self.dataset[index]
        return t

    def _get_class_names(self) -> list[str]:
        classes = [cats["name"] for cats in self.dataset.coco.cats.values()]
        return classes

    def _get_class_ids(self) -> list[int]:
        class_ids = [cats["id"] for cats in self.dataset.coco.cats.values()]
        return class_ids

    def _get_dataset_len(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image, target = self._get_image(index), self._get_target(index)
        image = Image.fromarray(image)
        image_id = self.dataset.ids[index]
        target = {"image_id": image_id, "annotations": target}
        image, target = self.prepare(image, target)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def plot_images_bbox(self, idxs: list[int], after_transforms: bool = False) -> plt.Figure:
        """Visualize images with their annotations (bounding boxes and class labels)."""
        n = len(idxs)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for i, idx in enumerate(idxs):
            if after_transforms:
                img, target = self.__getitem__(idx)
                img = img.permute(1, 2, 0).cpu().numpy() if isinstance(img, Tensor) else np.asarray(img)
                h, w = img.shape[:2]

                axes[i].imshow(img)
                axes[i].set_title(f"Image {idx}, H: {h}, W: {w}")
                axes[i].axis("off")

                if "boxes" in target:
                    boxes = target["boxes"]
                    labels = target.get("labels", None)
                    if boxes.shape[-1] == 4:
                        if boxes.max() <= 1.0:
                            boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
                            boxes = box_cxcywh2xyxy(boxes)
                        for j, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.tolist()
                            class_id = labels[j].item()
                            color = self.classes_int2color[class_id]
                            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
                            axes[i].add_patch(rect)
                            class_name = self.classes_int2str.get(class_id, f"Class {class_id}")
                            axes[i].text(x1, y1 - 5, class_name, color="white", bbox=dict(facecolor=color, alpha=0.7))
            else:
                img = self._get_image(idx)
                target = self._get_target(idx)
                h, w = img.shape[:2]

                axes[i].imshow(img)
                axes[i].set_title(f"Image {idx}, H: {h}, W: {w}")
                axes[i].axis("off")

                for anno in target:
                    # COCO format: [x_min, y_min, width, height]
                    x, y, w, h = anno["bbox"]
                    category_id = anno["category_id"]
                    color = self.classes_int2color[category_id]
                    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
                    axes[i].add_patch(rect)
                    class_name = self.classes_int2str.get(category_id, f"Class {category_id}")
                    axes[i].text(x, y, class_name, color="white", bbox=dict(facecolor=color, alpha=0.7))

        plt.tight_layout()
        return fig

    def plot_images_seg(
        self,
        idxs: list[int],
        after_transforms: bool = False,
    ) -> plt.Figure:
        """Visualize images with their annotations (segmentation masks and class labels)."""
        n = len(idxs)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for i, idx in enumerate(idxs):
            if after_transforms:
                img, target = self.__getitem__(idx)
                img = img.permute(1, 2, 0).cpu().numpy() if isinstance(img, Tensor) else np.asarray(img)
                h, w = img.shape[:2]

                axes[i].imshow(img)
                axes[i].set_title(f"Image {idx}, H: {h}, W: {w}")
                axes[i].axis("off")

                if "masks" in target:
                    masks = target["masks"].cpu().numpy()
                    labels = target.get("labels", None)

                    for j, mask in enumerate(masks):
                        class_id = labels[j].item() if labels is not None else j
                        color = self.classes_int2color[class_id]

                        colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)  # 4 channels for RGBA
                        colored_mask[:, :, :3] = np.array(color)[None, None, :]
                        colored_mask[:, :, 3] = mask * 0.5  # alpha = 0.5
                        axes[i].imshow(colored_mask, interpolation="nearest")
                        x, y = np.mean(np.where(mask > 0), axis=1).round().astype("int")
                        axes[i].text(
                            y,
                            x,
                            self.classes_int2str.get(class_id, f"Class {class_id}"),
                            color="black",
                            fontsize=8,
                            ha="center",
                            va="center",
                            bbox=dict(boxstyle="round", ec=color, fc=(1.0, 1.0, 1.0), alpha=0.7),
                        )
            else:
                img = self._get_image(idx)
                target = self._get_target(idx)
                h, w = img.shape[:2]

                axes[i].imshow(img)
                axes[i].set_title(f"Image {idx}, H: {h}, W: {w}")
                axes[i].axis("off")

                for anno in target:
                    if "segmentation" in anno:
                        category_id = anno["category_id"]
                        color = self.classes_int2color[category_id]

                        mask = coco_mask.decode(coco_mask.frPyObjects(anno["segmentation"], h, w))
                        if mask.ndim > 2:
                            mask = mask.sum(axis=2) > 0

                        colored_mask = np.zeros((*mask.shape, 4), dtype=np.float32)  # 4 channels for RGBA
                        colored_mask[:, :, :3] = np.array(color)[None, None, :]
                        colored_mask[:, :, 3] = mask * 0.5  # alpha = 0.5
                        axes[i].imshow(colored_mask, interpolation="nearest")
                        x, y = np.mean(np.where(mask > 0), axis=1).round().astype("int")
                        axes[i].text(
                            y,
                            x,
                            self.classes_int2str.get(category_id, f"Class {category_id}"),
                            color="black",
                            fontsize=8,
                            ha="center",
                            va="center",
                            bbox=dict(boxstyle="round", ec=color, fc=(1.0, 1.0, 1.0), alpha=0.7),
                        )

        plt.tight_layout()
        return fig


@register_dataset
class COCODataset(COCODatasetABC):
    dataset_dirpath = DATA_DIRPATH / "COCO"
    name = "COCODetection"


@register_dataset
class COCOMiniDataset(COCODatasetABC):
    dataset_dirpath = DATA_DIRPATH / "COCOminitrain"
    name = "COCOMiniDetection"


if __name__ == "__main__":
    coco_dataset = COCODataset("train")
    img, target = coco_dataset.dataset[0]
