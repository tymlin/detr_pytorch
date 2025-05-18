import torch

from detr.bin.utils import create_model
from detr.config import Config
from detr.utils.contants import CONFIG_TRAINING_DIRPATH

config_path = CONFIG_TRAINING_DIRPATH / "default.yaml"
cfg = Config.from_yaml(config_path)

model = create_model(cfg.model)
# model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

# checkpoint_path ="./detr-r50-e632da11.pth" # https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
# checkpoint = torch.load(checkpoint_path, map_location="cpu")
checkpoint = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", map_location="cpu", check_hash=True
)
model.load_state_dict(checkpoint["model"])
