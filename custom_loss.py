import torch
import monai
from torch.nn.modules.loss import _Loss
import math


class CoshDiceLoss(_Loss):
    def __init__(self):
        super(CoshDiceLoss, self).__init__()

    def forward(self, output, target):
        dice = monai.losses.DiceLoss(sigmoid=True, batch=True)

        dice_loss = dice(output, target)
        cosh = math.cosh(dice_loss.item())
        ln = math.log(cosh, math.e)
        return torch.as_tensor(ln)
