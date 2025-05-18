from detr.utils.registry import Registry, create_register_decorator

from .detr_loss import DETRCriterion
from .matcher import HungarianMatcher

LOSSES = Registry()
register_loss = create_register_decorator(LOSSES)


# Custom loss, e.g.:
#
# @register_loss
# class CustomLoss:
#     ...


@register_loss
class DETRLoss(DETRCriterion):
    def __init__(
        self,
        num_classes: int,
        num_decoder_layers: int,
        loss_ce: float,
        loss_bbox: float,
        loss_giou: float,
        eos_coef: float,
        aux_loss: bool,
        losses: list,
        matcher_params: dict,
    ) -> None:
        """Wrapper for the DETRCriterion.

        :param num_classes (int): Number of object classes.
        :param num_decoder_layers (int): Number of decoder layers.
        :param loss_ce (float): Weight for the classification loss.
        :param loss_bbox (float): Weight for the bounding box loss.
        :param loss_giou (float): Weight for the generalized IoU loss.
        :param eos_coef (float): Relative weight for the no-object class.
        :param aux_loss (bool): Whether to use auxiliary loss.
        :param losses (list): List of loss types to compute.
        :param matcher_params (dict): Parameters for the matcher.

        :return: None
        """
        matcher = HungarianMatcher(**matcher_params)
        weight_dict = {"loss_ce": loss_ce, "loss_bbox": loss_bbox}
        weight_dict["loss_giou"] = loss_giou
        if aux_loss:
            aux_weight_dict = {}
            for i in range(num_decoder_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses)
