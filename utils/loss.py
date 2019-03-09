# %% libraries
import torch.nn as nn
import torch


class EdgeLoss(nn.Module):
    def __init__(self):
        """
        A weighted sum of pixel-wise L1 loss and sum of L2 loss of Gram matrices.

        :param w1: weight of L1  (pixel-wise)
        :param w2: weight of L2 loss (Gram matrix)
        """
        super(EdgeLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, y, y_pred):
        loss = self.cross_entropy(y, y_pred)
        return loss
