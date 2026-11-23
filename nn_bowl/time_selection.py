
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSelection(nn.Module):
    def __init__(self, cfg, model):
        super(TimeSelection, self).__init__()

        self.cfg = cfg
        self.n_timesteps = cfg.model.n_timesteps

        if self.cfg.model.time_selection.scheme == 'adaptive':
            self.timestep_logits_adaptive = nn.Sequential(
                nn.Linear(model.n_layers * model.n_channels, cfg.model.time_selection.mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.model.time_selection.mlp_hidden_dim, 1)
            )

        elif self.cfg.model.time_selection.scheme == 'fixed':
            self.timestep_logits_fixed = nn.Parameter(torch.zeros(self.n_timesteps), requires_grad=True)

        else:
            raise ValueError(f"Unknown time selection type: {self.cfg.model.time_selection.scheme}")


    def get_time_selection(self, logits_time_selection, hard=False):

        # normalize so that logits are on unit sphere
        logits_normalized = logits_time_selection / (logits_time_selection.norm(dim=1, keepdim=True) + 1e-8)

        # scaling to increase dynamic range
        logits_scaled = logits_normalized * self.cfg.costs.time.logit_scale

        # simple softmax
        time_selection = F.softmax(logits_scaled, dim=1)

        return time_selection

    def forward(self, layer_outputs, t):
        if self.cfg.model.time_selection.scheme == 'adaptive':
            logit_output = self.timestep_logits_adaptive(layer_outputs).squeeze(-1)
        elif self.cfg.model.time_selection.scheme == 'fixed':
            logit_output = self.timestep_logits_fixed[t]

        return logit_output
