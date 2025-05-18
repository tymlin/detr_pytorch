import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from detr.loggers import log
from detr.utils.types import Split

from .dataset import COCODatasetABC  # avoids circular import


class DataModule:
    def __init__(
        self,
        train_ds: COCODatasetABC | None,
        val_ds: COCODatasetABC | None,
        test_ds: COCODatasetABC | None,
        inverse_transforms: Compose | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool = True,
        collate_fn_train: Callable | None = None,
        collate_fn_val: Callable | None = None,
        collate_fn_test: Callable | None = None,
    ):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.inverse_transforms = inverse_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.collate_fn = collate_fn_train
        self.collate_fn_val = collate_fn_val
        self.collate_fn_test = collate_fn_test

        self._log_info()

        self.total_batches = {}
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.classes_names = None
        self.classes_int2str = None
        self.classes_int2color = None

    def setup_dataloaders(self) -> None:
        setup_one = False
        if self.train_ds is not None:
            self.train_dataloader = self._dataloader("train")
            self.classes_names = self.train_ds.classes_names
            self.classes_int2str = self.train_ds.classes_int2str
            self.classes_int2color = self.train_ds.classes_int2color
            setup_one = True
        if self.val_ds is not None:
            self.val_dataloader = self._dataloader("val")
            self.classes_names = self.val_ds.classes_names
            self.classes_int2str = self.val_ds.classes_int2str
            self.classes_int2color = self.val_ds.classes_int2color
            setup_one = True
        if self.test_ds is not None:
            self.test_dataloader = self._dataloader("test")
            self.classes_names = self.test_ds.classes_names
            self.classes_int2str = self.test_ds.classes_int2str
            self.classes_int2color = self.test_ds.classes_int2color
            setup_one = True
        if not setup_one:
            raise ValueError("No datasets provided for dataloaders")

    def get_dataset(self, split: Split) -> COCODatasetABC:
        if split == "train":
            return self.train_ds
        elif split == "val":
            return self.val_ds
        elif split == "test":
            return self.test_ds
        else:
            raise ValueError("Wrong split type passed")

    def get_dataloader(self, split: Split) -> DataLoader:
        if split == "train":
            return self.train_dataloader
        elif split == "val":
            return self.val_dataloader
        elif split == "test":
            return self.test_dataloader
        else:
            raise ValueError("Wrong split type passed")

    def get_classes_names_dataset(self):
        if self.classes_names is not None:
            return self.classes_names
        else:
            raise ValueError("No datasets provided for dataloaders")

    def get_classes_int2str_dataset(self):
        if self.classes_int2str is not None:
            return self.classes_int2str
        else:
            raise ValueError("No datasets provided for dataloaders")

    def get_classes_int2color_dataset(self):
        if self.classes_int2color is not None:
            return self.classes_int2color
        else:
            raise ValueError("No datasets provided for dataloaders")

    def _dataloader(self, split: Split) -> DataLoader:
        shuffle = split == "train" and self.shuffle  # shuffle only for train
        drop_last = self.drop_last if split == "train" else False  # don't drop last for val/test

        dataset = self.get_dataset(split)
        if dataset is None:
            raise ValueError(f"Dataset for {split} split is `None`")

        collate_fn = getattr(self, f"collate_fn_{split}", self.collate_fn)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

        self.total_batches[split] = len(dataloader)
        return dataloader

    def _log_info(self) -> None:
        datamodule_info = []
        datasets = {"train": self.train_ds, "val": self.val_ds, "test": self.test_ds}
        for split, ds in datasets.items():
            ds_name = "None" if ds is None else ds.__class__.__name__
            num_samples = 0 if ds is None else len(ds)
            ds_info = f"{split.rjust(6)}: {num_samples} ({ds_name})"
            datamodule_info.append(ds_info)
        datamodule_info = "\n".join(datamodule_info)
        log.info(f"DataModule stats: \n{datamodule_info}")

    @staticmethod
    def state_dict() -> dict:
        state = {
            "random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            cuda_state = {
                "torch_cuda_random_state_all": torch.cuda.get_rng_state_all(),
                "torch_cuda_random_state": torch.cuda.get_rng_state(),
            }
            state.update(cuda_state)
        return state

    @staticmethod
    def load_state_dict(state_dict: dict) -> None:
        random.setstate(state_dict["random_state"])
        torch.random.set_rng_state(state_dict["torch_random_state"].cpu())
        np.random.set_state(state_dict["numpy_random_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state_dict.get("torch_cuda_random_state_all", []))
        log.info("Loaded datamodule state")
