from typing import Literal

DatasetType = Literal["COCODataset", "COCOMiniDataset"]
Split = Literal["train", "val", "test"]
Accelerator = Literal["cpu", "cuda"]
LRSchedulerInterval = Literal["step", "epoch"]
ModuleStage = Literal["train", "val"]
