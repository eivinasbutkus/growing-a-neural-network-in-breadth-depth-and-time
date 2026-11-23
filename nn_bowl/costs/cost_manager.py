

import torch
import torch.nn as nn

from .error_cost import CrossEntropyErrorCost
from .space_cost import SpaceCost
from .time_cost import TimeCost
from nn_bowl.utils.train_utils import linear_anneal


COST_CLASSES = {
    # errors
    "cross_entropy_error_cost": CrossEntropyErrorCost,

    # space
    "space_cost": SpaceCost,

    # time
    "time_cost": TimeCost,
}


# class CostManager:
class CostManager(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()

        self.cfg = dict(cfg)
        self.model = model

        self.costs = nn.ModuleDict()
        self.anneal_params = {}

        self.metrics = {}
        self.metrics["loss"] = [] # tracking total loss

        for cost_name in ["errors", "space", "time", "energy", "data"]:
            cost_config = cfg["costs"][cost_name]
            if cost_config["enabled"]:
                cost_class_key = cost_config["cost"]
                self.costs[cost_name] = COST_CLASSES[cost_class_key](cfg, model)

                # adding linear annealing params if there
                if "anneal" in cost_config:
                    self.anneal_params[cost_name] = dict(cost_config["anneal"])

            self.metrics[f"loss_{cost_name}"] = [] # tracking individual loss components

    # def __call__(self, logits, labels, activations, epoch):
    def forward(self, out, labels, epoch):
        loss = 0.0
        for cost_name, cost in self.costs.items():
            cost_anneal = linear_anneal(epoch, **self.anneal_params[cost_name]) if cost_name in self.anneal_params else 1.0
            loss_cost = cost_anneal * cost(out, labels)
            self.metrics[f"loss_{cost_name}"].append(loss_cost.detach())
            loss += loss_cost

        self.metrics["loss"].append(loss.detach())
        return loss

    def get_postfix(self):
        postfix = {}
        for key, metric in self.metrics.items():
            if key not in self.cfg["train"].get("include_in_postfix", []):
                continue

            avg = torch.stack(metric).mean().item() if metric else 0
            postfix[key] = avg

        for cost_name, cost in self.costs.items():
            for key, metric in cost.metrics.items():
                if key not in self.cfg["train"].get("include_in_postfix", []):
                    continue
                avg = torch.stack(metric).mean().item() if metric else 0
                postfix[key] = avg

        return postfix


    def write_to_tensorboard(self, writer, split, epoch):
        # writing metric from the cost manager
        for key, metric in self.metrics.items():
            # avg = sum(metric) / len(metric) if metric else 0
            avg = torch.stack(metric).mean().item() if metric else 0
            writer.add_scalar(f"{split}/{key}", avg, epoch)

        # writing metrics from individual costs
        for cost_name, cost in self.costs.items():
            for key, metric in cost.metrics.items():
                avg = torch.stack(metric).mean().item() if metric else 0
                writer.add_scalar(f"{cost_name}/{split}/{key}", avg, epoch)
