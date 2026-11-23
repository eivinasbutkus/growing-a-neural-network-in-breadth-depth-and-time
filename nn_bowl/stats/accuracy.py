
from .stat import Stat


class Accuracy(Stat):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, model, logits, activations, y):
        _, predicted = logits[:, -1].max(dim=1)
        self.correct += (predicted == y).sum().item()
        self.total += y.size(0)