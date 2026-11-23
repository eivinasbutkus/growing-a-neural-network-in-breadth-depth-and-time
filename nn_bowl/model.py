

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_bowl

import hydra


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.cfg = cfg

        self.n_layers = cfg.model.n_layers
        self.n_channels = cfg.model.n_channels
        # self.n_input_channels = cfg.dataset.n_input_channels  # number of input channels (e.g. 3 for RGB)
        self.n_input_channels = 3 # hard code to 3 for RGB since we're only using image datasets for now
        self.n_classes = cfg.dataset.n_classes

        self.n_timesteps = cfg.model.n_timesteps

        # conv layers
        self.conv_layers = nn.ModuleDict()
        self.conv_layers["input"] = nn.Conv2d(self.n_input_channels, self.n_channels, kernel_size=3, padding=1) # initial conv to match input channels to n_channels and scale the input

        for i in range(self.n_layers):
            self.conv_layers[f"bottom_up_{i}"] = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, padding=1, bias=False)
            self.conv_layers[f"lateral_{i}"] = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, padding=1, bias=False)
            if i < self.n_layers - 1:
                self.conv_layers[f"top_down_{i}"] = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, padding=1, bias=False)

        # for hooking attribution methods
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(self.n_layers)])

        # learnt init state
        self.register_buffer('h0', torch.zeros(1, self.n_layers, self.n_channels, cfg.model.spatial_size, cfg.model.spatial_size))

        # read out across layers
        self.read_out = nn.Linear(self.n_layers * self.n_channels, self.n_classes)

        # time selection
        self.time_selection = nn_bowl.TimeSelection(cfg, self)

        # initialize weights
        nn_bowl.utils.init_weights(cfg, self)


    def forward_t(self, x, h, t, noise_t=None, noise_outputs_t=None):
        outputs = torch.zeros(x.shape[0], self.n_layers, self.n_channels, device=x.device, dtype=x.dtype) # to store average pooled outputs from each layer

        # h: internal hidden state private to each neuron (~membrane potential, although probably at a coarser time scale)
        # h_activated: what other neurons hear (~spikes)
        h_activated = torch.relu(h)
        h_activated = h_activated / (1.0 + h_activated.mean(dim=(2,3,4), keepdim=True)) # divisive normalization (lateral and top-down)

        for i in range(self.n_layers):
            x = x / (1.0 + x.mean(dim=(1,2,3), keepdim=True)) # divisive normalization (bottom-up)

            combined = 0.0
            combined += self.conv_layers[f"bottom_up_{i}"](x)  # bottom-up input from the layer below
            combined += self.conv_layers[f"lateral_{i}"](h_activated[:, i])  # lateral input from the same layer
            if i < self.n_layers - 1:
                combined += self.conv_layers[f"top_down_{i}"](h_activated[:, i + 1])  # top-down input from the layer above

            x = combined + h[:, i]  # no leak

            # noise (pre-generated in forward() to minimize CUDA kernel launches)
            if noise_t is not None:
                x = x + noise_t[:, i]

            h[:, i] = x  # update hidden state for the current layer

            # non-linearity (spikes to the next layer)
            # x = F.relu(x)
            x = self.relus[i](x)

            outputs[:, i] = x.mean(dim=(2, 3)) # average pooling

        outputs_flat = outputs.view(x.shape[0], -1)  # flattening layer and channel dimensions

        # SNR is way higher after spatial pooling, so adding noise to pull back SNR
        # ties weight magnitude to the noise scale more closely
        if noise_outputs_t is not None:
            outputs_flat = outputs_flat + noise_outputs_t

        logits_t = self.read_out(outputs_flat)  # readout logits
        logit_time_selection = self.time_selection(outputs_flat, t)  # get logit for time selection at this timestep

        return logits_t, logit_time_selection, h


    def forward(self, x, noise_anneal=1.0):
        x = self.conv_layers["input"](x)  # initial convolution to match input channels to n_channels
        batch_size = x.shape[0]

        # initial hidden state size (batch_size, n_layers, n_channels, height, width)
        h = self.h0.repeat(batch_size, 1, 1, 1, 1)
        logits_all = torch.zeros(batch_size, self.n_timesteps, self.n_classes, device=x.device, dtype=x.dtype)
        logits_time_selection = torch.zeros(batch_size, self.n_timesteps, device=x.device, dtype=x.dtype)

        # Pre-generate all noise in one kernel launch instead of one per (timestep, layer)
        if self.cfg.model.noise_std > 0:
            noise_all = (noise_anneal * self.cfg.model.noise_std) * torch.randn(
                batch_size, self.n_timesteps, self.n_layers, *x.shape[1:],
                device=x.device, dtype=x.dtype
            )
            noise_outputs_all = (noise_anneal * self.cfg.model.noise_std) * torch.randn(
                batch_size, self.n_timesteps, self.n_layers * self.n_channels,
                device=x.device, dtype=x.dtype
            )
        else:
            noise_all = None
            noise_outputs_all = None

        for t in range(self.n_timesteps):
            noise_t = noise_all[:, t] if noise_all is not None else None
            noise_outputs_t = noise_outputs_all[:, t] if noise_outputs_all is not None else None
            logits_t, logit_time_selection, h = self.forward_t(x, h, t, noise_t=noise_t, noise_outputs_t=noise_outputs_t)
            logits_all[:, t] = logits_t
            logits_time_selection[:, t] = logit_time_selection

        time_selection = self.time_selection.get_time_selection(logits_time_selection)  # (batch_size, n_timesteps)

        # final answer is a mixture of probability distributions from different timesteps
        probs_all = F.softmax(logits_all, dim=-1)
        probs = (probs_all * time_selection.unsqueeze(-1)).sum(dim=1)

        out = {
            "probs": probs,
            "probs_all": probs_all,
            "time_selection": time_selection,
            # "time_selection_soft": time_selection_soft,
            "activations": None
        }

        return out


@hydra.main(version_base=None, config_name='config', config_path=nn_bowl.paths.CONFIG_PATH)
def test_model(cfg):
    model = Model(cfg)
    print(model)

    x = torch.randn(128, 3, 32, 32)
    out = model(x)

    print(out["probs"].shape)
    print(out["time_selection"].shape)
    print(out["time_selection"][:3].detach().cpu().numpy())


if __name__ == '__main__':
    test_model()