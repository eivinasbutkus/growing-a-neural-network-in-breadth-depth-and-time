
from abc import ABC, abstractmethod

class Stat(ABC):
    @abstractmethod
    def update(self, model, logits, activations, y):
        pass

    @abstractmethod
    def get_value(self):
        pass


class Stats:
    def __init__(self, stats=[]):
        self.n = 0
        self.stats = stats


    def update(self, model, logits, activations, y):
        self.n += y.shape[0]
        for stat in self.stats:
            stat.update(model, logits, activations, y)


    def get_stats(self):
        stats = {}
        for stat in self.stats:
            stats[stat.__class__.__name__] = stat.get_value()