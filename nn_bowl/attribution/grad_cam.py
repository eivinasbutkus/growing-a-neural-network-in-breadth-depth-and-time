import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer_index=-1, target_timestep_index=-1):
        """
        Args:
            model: nn.Module model.
            target_layer_index: Index of the target layer.
            target_timestep_index: Index of the target timestep.
        """
        self.model = model
        # self.target_layer = model.conv_layers[f"bottom_up_{target_layer_index}"] if target_layer_index != -1 else model.conv_layers[f"bottom_up_{model.n_layers - 1}"]
        self.target_layer = model.relus[target_layer_index]
        self.target_timestep_index = target_timestep_index
        self.gradients = []
        self.activations = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Store activations for all timesteps
            self.activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            # Store gradients for all timesteps (order will be reversed during backprop, we'll handle that)
            self.gradients.append(grad_output[0].detach())

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_image, input_class):
        """
        Generates Grad-CAM heatmaps.
        Args:
            input_image: Tensor of shape (1, C, H, W)
            input_class: Index of the input class. If None, uses the highest scoring class.
        Returns:
            Heatmap: Numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        self.activations = [] # Reset for this call
        self.gradients = [] # Reset for this call
        
        # Forward pass
        outputs = self.model(input_image)
        
        # Use target timestep for logic
        logits = outputs["probs_all"][:, self.target_timestep_index, :]

        # Backward pass
        score = logits[:, input_class]
        score.backward(retain_graph=True)

        # Grad-CAM logic
        # Activations were stored forward (0 to T)
        # Gradients are stored during backward pass (T down to 0). 
        # So we reverse the gradients list so indices match timesteps.
        self.gradients.reverse()
        
        grads = self.gradients[self.target_timestep_index]      # shape (B, C, H, W)
        activations = self.activations[self.target_timestep_index] # shape (B, C, H, W)

        # Weight the channels by the gradients (Global Average Pooling)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU to keep only positive influence
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()[0, 0]
        # if cam.max() > 0:
        #      cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
             
        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
