

import math
import torch
from nn_bowl.costs.cost import Cost


class TimeCost(Cost):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.lam = cfg.costs.time.lam
        # self.entropy_beta = cfg.costs.time.get('entropy_beta', 0.0)
        self.n_timesteps = model.n_timesteps

        # normalize so max multiplier is 1.0
        # position_power > 1 creates a steeper cost curve, penalizing later timesteps more sharply
        position_power = cfg.costs.time.get('position_power', 1)
        multipliers = torch.arange(1, self.n_timesteps + 1).float() / max(1, self.n_timesteps - 1)
        self.register_buffer('time_step_multipliers', multipliers ** position_power)

        for t in range(self.n_timesteps):
            self.metrics[f"tw_mean_t{t}"] = []
            self.metrics[f"tw_std_t{t}"] = []
        self.metrics["time_entropy"] = []

    def get_time_cost(self, out, labels):
        time_selection = out["time_selection"]
        for t in range(self.n_timesteps):
            self.metrics[f"tw_mean_t{t}"].append(time_selection[:, t].mean().detach())
            self.metrics[f"tw_std_t{t}"].append(time_selection[:, t].std().detach())

        time_cost = (time_selection @ self.time_step_multipliers)
        return time_cost


    def forward(self, out, labels):
        time_cost = self.get_time_cost(out, labels)
        loss_time = self.lam * time_cost.mean()

        return loss_time

