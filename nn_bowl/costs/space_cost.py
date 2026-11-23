
import torch
from nn_bowl.costs.cost import Cost


class SpaceCost(Cost):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

        self.lams = {}
        self.lams["depth"] = cfg.costs.space.lam_depth
        self.lams["breadth"] = cfg.costs.space.lam_breadth

        self.norm_power = cfg.costs.space.norm_power

        self.metrics["cost_space_depth"] = []
        self.metrics["cost_space_breadth"] = []
        self.metrics["cost_space_breadth_lammed"] = []
        self.metrics["cost_space_depth_lammed"] = []
        self.metrics["weight_norm"] = []

        position_power = cfg.costs.space.get('position_power', 1)

        raw_depth = (torch.arange(1, model.n_layers + 1).float() / model.n_layers) ** position_power
        self.register_buffer('depth_weight_multipliers', raw_depth / raw_depth.mean()) # normalize so mean multiplier is 1.0, preventing cost scale from changing with position power

        raw_breadth = (torch.arange(1, model.n_channels + 1).float() / model.n_channels) ** position_power
        self.register_buffer('breadth_weight_multipliers', raw_breadth / raw_breadth.mean()) # normalize so mean multiplier is 1.0, preventing cost scale from changing with position power


    def get_space_costs(self):
        # print("Calculating space costs...")

        # getting layer x channel weights norms (shape: [n_layers, n_channels])
        layer_channel_norms = []
        layer_channel_indices = []
        for i in range(self.model.n_layers):
            # extract weights
            bottom_up_weights = self.model.conv_layers[f"bottom_up_{i}"].weight
            lateral_weights = self.model.conv_layers[f"lateral_{i}"].weight
            if i < self.model.n_layers - 1:
                top_down_weights = self.model.conv_layers[f"top_down_{i}"].weight

            if self.norm_power == 1:
                layer_channel_norm = bottom_up_weights.abs() + lateral_weights.abs()
                if i < self.model.n_layers - 1:
                    layer_channel_norm += top_down_weights.abs()

            elif self.norm_power == 2:
                layer_channel_norm = bottom_up_weights.pow(2) + lateral_weights.pow(2)
                if i < self.model.n_layers - 1:
                    layer_channel_norm += top_down_weights.pow(2)

            else:
                raise ValueError(f"Unsupported norm power: {self.norm_power!r}")

            layer_channel_norm_sum = layer_channel_norm.mean(dim=(1, 2, 3))

            # sorting according to channel magnitude: this is to align the channel cost with the most important channels
            # and enforce monotonicity of channel magnitudes (at least in the sorted space)
            layer_channel_norm_sum_sorted, layer_channel_sorted_indices = torch.sort(layer_channel_norm_sum, dim=0, descending=True)
            layer_channel_norms.append(layer_channel_norm_sum_sorted)
            layer_channel_indices.append(layer_channel_sorted_indices)

        layer_channel_norms = torch.stack(layer_channel_norms)  # shape: [n_layers, n_channels]
        layer_channel_indices = torch.stack(layer_channel_indices)  # shape: [n_layers, n_channels]

        space_costs = {}
        for cost_type in ["depth", "breadth"]:
            # selecting layers (index 0) or channels (index 1)
            if cost_type == "depth":
                space_cost_weights = self.depth_weight_multipliers.view(-1, 1)
            else:
                space_cost_weights = self.breadth_weight_multipliers.view(1, -1)

            # print device
            space_costs[cost_type] = torch.mean(layer_channel_norms * space_cost_weights)

        return space_costs, layer_channel_norms, layer_channel_indices


    def forward(self, out, labels):
        total_cost = 0.0

        space_costs, _, _ = self.get_space_costs()
        for space_cost_type, space_cost in space_costs.items():
            space_cost_lammed = self.lams[space_cost_type] * space_cost
            self.metrics[f"cost_space_{space_cost_type}_lammed"].append(space_cost_lammed.detach())
            self.metrics[f"cost_space_{space_cost_type}"].append(space_cost.detach())
            total_cost += space_cost_lammed

        # further metrics
        weight_norm_sq = sum(p.data.norm(2) ** 2 for p in self.model.parameters() if p.requires_grad)
        weight_norm = weight_norm_sq.sqrt()
        self.metrics["weight_norm"].append(weight_norm.detach())
        return total_cost
    
