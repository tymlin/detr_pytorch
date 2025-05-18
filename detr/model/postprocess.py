import torch
import torch.nn.functional as F
from torch import nn

from detr.data.utils import box_cxcywh2xyxy


class PostProcess(nn.Module):
    """Converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Forward pass.

        :param outputs: the raw outputs of the model
        :param target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                             For evaluation, this must be the original image size (before any data augmentation)
                             For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh2xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": _s, "labels": _l, "boxes": _b} for _s, _l, _b in zip(scores, labels, boxes)]

        return results
