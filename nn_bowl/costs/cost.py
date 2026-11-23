
from torch import nn
from abc import ABC, abstractmethod


"""
Every cost function (errors, space, data, energy, time) has access to the model and configuration.
"""
class Cost(nn.Module, ABC):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.metrics = {}

    @abstractmethod
    def forward(self, out, labels):
        pass

