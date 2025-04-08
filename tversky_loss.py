import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.5
BETA = 0.5

class MultiClassTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MultiClassTverskyLoss, self).__init__()

    def forward(self, inputs, targets, epsilon=1e-7, alpha=ALPHA, beta=BETA):
        inputs = F.softmax(inputs, dim=1)  

        inputs = inputs.view(inputs.size(0), -1)  
        targets = targets.view(targets.size(0), -1)  

      
        tversky = 0
        num_classes = inputs.size(1)

        for c in range(num_classes):
            
            TP = (inputs[:, c] * targets[:, c]).sum()
            FP = ((1 - targets[:, c]) * inputs[:, c]).sum()
            FN = (targets[:, c] * (1 - inputs[:, c])).sum()

            
            tversky_class = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
            tversky += tversky_class

    
        tversky_loss = 1 - (tversky / num_classes)
        return tversky_loss
