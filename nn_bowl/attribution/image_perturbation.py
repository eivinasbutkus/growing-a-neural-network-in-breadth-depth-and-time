import torch
import torch.nn.functional as F
import numpy as np

class ImagePerturbation:
    def __init__(self, model, kernel_size=5, stride=1, noise_std=0.1):
        """
        Args:
            model: nn.Module, the model to evaluate.
            kernel_size: int or tuple, size of the sliding window.
            stride: int or tuple, stride of the sliding window.
            noise_std: float, standard deviation of the random noise.
        """
        self.model = model
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.noise_std = noise_std
        self.device = next(model.parameters()).device

    def _random_noise(self, patch):
        # Replace patch with random noise of same shape
        return torch.rand_like(patch)
    
    def _gaussian_noise(self, patch):
        # Add Gaussian noise to the existing patch and clamp to [0, 1]
        noise = torch.randn_like(patch) * self.noise_std
        return torch.clamp(patch + noise, 0, 1)


    def generate_heatmap(self, input_image, input_class, batch_size=64):
        self.model.eval()
        input_image = input_image.to(self.device) if self.device else input_image

        with torch.no_grad():
            output = self.model(input_image)
            probs = output["probs"]
            if input_class is None:
                input_class = torch.argmax(probs, dim=1).item()
            baseline_score = probs[0, input_class].item()

        _, _, H, W = input_image.shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)

        # collect all patch positions
        positions = []
        for y in range(0, H - self.kernel_size[0] + 1, self.stride[0]):
            for x in range(0, W - self.kernel_size[1] + 1, self.stride[1]):
                positions.append((y, x))

        # process in batches
        for batch_start in range(0, len(positions), batch_size):
            batch_positions = positions[batch_start:batch_start + batch_size]
            batch_images = input_image.repeat(len(batch_positions), 1, 1, 1)

            for i, (y, x) in enumerate(batch_positions):
                patch = batch_images[i:i+1, :, y:y+self.kernel_size[0], x:x+self.kernel_size[1]]
                batch_images[i:i+1, :, y:y+self.kernel_size[0], x:x+self.kernel_size[1]] = self._gaussian_noise(patch)

            with torch.no_grad():
                out = self.model(batch_images)
                scores = out["probs"][:, input_class]

            for i, (y, x) in enumerate(batch_positions):
                diff = baseline_score - scores[i].item()
                heatmap[y:y+self.kernel_size[0], x:x+self.kernel_size[1]] += diff
                counts[y:y+self.kernel_size[0], x:x+self.kernel_size[1]] += 1

        # average overlapping regions
        heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts != 0)

        # normalize to [0, 1]
        # if heatmap.max() > 0:
        #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap