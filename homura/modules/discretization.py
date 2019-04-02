import torch
from torch import nn

from .functional import (gumbel_sigmoid, straight_through_estimator,
                         semantic_hashing, gumbel_softmax, to_onehot)

__all__ = ["GumbelSigmoid", "GumbelSoftmax", "StraightThroughEstimator", "SemanticHashing"]


class GumbelSigmoid(nn.Module):
    """ This module outputs `gumbel_sigmoid` while training and `input.sigmoid() >= threshold` while evaluation
    """

    def __init__(self,
                 temp: float = 0.1,
                 threshold: float = 0.5):
        super(GumbelSigmoid, self).__init__()
        self.temp = temp
        self.threshold = threshold

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return gumbel_sigmoid(input, self.temp)
        else:
            return (input.sigmoid() >= self.threshold).float()


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, input: torch.Tensor):
        return straight_through_estimator(input)


class SemanticHashing(nn.Module):
    def __init__(self):
        super(SemanticHashing, self).__init__()

    def forward(self, input: torch.Tensor):
        return semantic_hashing(input, self.training)


class GumbelSoftmax(nn.Module):
    """ This module outputs `gumbel_softmax` while training and argmax in onehot fashion while evaluation
    """

    def __init__(self,
                 dim: int,
                 temp: float = 0.1):
        super(GumbelSoftmax, self).__init__()
        self.dim = dim
        self.temp = temp

    def forward(self, input: torch.Tensor):
        if self.training:
            return gumbel_softmax(input, self.dim, self.temp)
        else:
            return to_onehot(input.argmax(dim=self.dim), input.size(self.dim))
