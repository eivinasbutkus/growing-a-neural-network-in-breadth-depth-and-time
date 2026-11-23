
import torch
import torch.nn as nn



def init_weights(cfg, model):
    if cfg.model.init.scheme == "kaiming":
        # Initialize conv layers with Kaiming (He) initialization
        for layer in model.conv_layers.values():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    elif cfg.model.init.scheme == "xavier":
        # Initialize conv layers with Xavier initialization
        for layer in model.conv_layers.values():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight, gain=cfg.model.init.scale)  # Increased gain
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)  # Small positive bias


        # Initialize readout with even smaller scale
        nn.init.xavier_normal_(model.read_out.weight, gain=0.01)  # Very small!
        nn.init.constant_(model.read_out.bias, 0.0)  # Zero bias for classification
