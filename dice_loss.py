import torch
from torch import nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, inputs, targets, epsilon=1e-6):
        inputs = F.softmax(inputs, dim=1)  

        inputs = inputs.view(inputs.size(0), -1)  
        targets = targets.view(targets.size(0), -1)  

        dice = 0
        num_classes = inputs.size(1)

        for c in range(num_classes):
            intersection = (inputs[:, c] * targets[:, c]).sum()
            dice += (2. * intersection + epsilon) / (inputs[:, c].sum() + targets[:, c].sum() + epsilon)

        dice_loss = 1 - (dice / num_classes)
        return dice_loss
