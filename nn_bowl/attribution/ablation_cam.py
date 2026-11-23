import torch
import torch.nn.functional as F
import numpy as np

class AblationCAM:
    def __init__(self, model, target_layer_index=-1, target_timestep_index=-1):
        """
        Args:
            model: nn.Module model.
            target_layer_index: Index of the target layer.
            target_timestep_index: Index of the target timestep.
        """
        self.model = model
        # self.target_layer = model.conv_layers[f"bottom_up_{target_layer_index}"] if target_layer_index is not -1 else model.conv_layers[f"bottom_up_{model.n_layers - 1}"]
        self.target_layer = model.relus[target_layer_index]
        self.target_timestep_index = target_timestep_index
        self.activations = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Store activations for all timesteps
            self.activations.append(output.detach())

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))

    def generate_heatmap(self, input_image, input_class):
        """
        Generates Ablation-CAM heatmaps by zeroing out individual activation channels.
        Args:
            input_image: Tensor of shape (1, C, H, W)
            input_class: Index of the input class. 
        Returns:
            Heatmap: Numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        self.activations = [] # Reset for this call
        
        # 1. Get baseline output
        with torch.no_grad():
            outputs = self.model(input_image)
            probs = outputs["probs"]
            if input_class is None:
                input_class = torch.argmax(probs, dim=1).item()
            baseline_score = probs[0, input_class].item()
        
        # Get activations for the requested timestep
        layer_activations = self.activations[self.target_timestep_index]
        n_channels = layer_activations.shape[1]
        
        weights = []
        
        for i in range(n_channels):
            def ablation_hook(module, input, output):
                modified_output = output.clone()
                modified_output[:, i, :, :] = 0
                return modified_output
            
            temp_hook = self.target_layer.register_forward_hook(ablation_hook)
            
            with torch.no_grad():
                modified_outputs = self.model(input_image)
                modified_score = modified_outputs["probs"][0, input_class].item()
            
            temp_hook.remove()
            weight = (baseline_score - modified_score) / (baseline_score + 1e-8)
            weights.append(weight)
        
        # 3. Compute weighted sum of activations
        weights = torch.tensor(weights, device=input_image.device).view(1, -1, 1, 1)
        cam = torch.mul(layer_activations, weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam) 
        
        # 4. Normalize between 0 and 1, for visualization
        cam = cam.cpu().numpy()[0, 0]
        # if cam.max() > 0:
        #     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
