import detr.data.transforms as T
from detr.data import COCODataset, DataModule, create_train_transforms, create_val_transforms

if __name__ == "__main__":  # needs to be for multiprocessing
    train_transforms, val_transforms = create_train_transforms(), create_val_transforms()
    train_ds = COCODataset("train", transforms=train_transforms)
    val_ds = COCODataset("val", transforms=val_transforms)

    img_train, targets_train = train_ds[0]

    datamodule = DataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=None,
        batch_size=4,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        collate_fn_train=train_ds.__class__.collate_fn,
        collate_fn_val=val_ds.__class__.collate_fn,
    )

    datamodule.setup_dataloaders()
    train_dataloader = datamodule.train_dataloader

    images, labels = next(iter(train_dataloader))
    print(f"images shape: {images.tensors.shape}, labels shape: {len(labels)}")
