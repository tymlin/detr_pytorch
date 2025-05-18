import random

import matplotlib.pyplot as plt

import detr.data.transforms as T
from detr.data import COCODataset

normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
train_transform = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose(
                [
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ]
            ),
        ),
        normalize,
    ]
)

basic_transforms = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        normalize,
    ]
)
transforms = None  # train_transform, basic_transforms, None

train_ds = COCODataset("train", transforms=transforms, return_masks=True)
val_ds = COCODataset("val", transforms=transforms, return_masks=True)
len_ = len(train_ds)
print(f"Length of train dataset: {len_}")
img, label = train_ds[0]
print(f"Image shape: {img.size} (type: {type(img)}), Label: {label} (type: {type(label)})")

idxs = list(range(10))
# idxs = random.sample(range(len_), 10)
fig = train_ds.plot_images_bbox(idxs, after_transforms=True)
fig.show()
plt.close("all")
fig = train_ds.plot_images_seg(idxs, after_transforms=True)
fig.show()
plt.close("all")
