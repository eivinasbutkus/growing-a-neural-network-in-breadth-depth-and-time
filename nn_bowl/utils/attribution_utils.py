import base64
import io

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from nn_bowl.attribution.grad_cam import GradCAM
from nn_bowl.attribution.ablation_cam import AblationCAM
from nn_bowl.attribution.image_perturbation import ImagePerturbation
# from nn_bowl.attribution.visualization import overlay_heatmap
from scipy.stats import entropy
from scipy.ndimage import center_of_mass

# gets raw heatmaps for test images in config.yaml — for use in evaluation.py
# returns a dict of dicts: {img_index: {method_name: heatmap_array,...},...}
def get_attribution_heatmaps_raw_for_test_images(cfg, model, test_loader, device):

    # for visualization
    image_indices = cfg.analysis.attribution.cifar_10_test_selected_img_inds

    # for stats, add 
    random_state = np.random.RandomState(cfg.analysis.attribution.cifar_10_test_stats_seed)
    randomly_draw_indices = random_state.choice(len(test_loader.dataset), size=cfg.analysis.attribution.cifar_10_test_stats_n_imgs, replace=False)

    # image_indices = sorted(set(image_indices) | set(randomly_draw_indices))
    image_indices = sorted(set(int(i) for i in image_indices) | set(int(i) for i in randomly_draw_indices))

    heatmaps_all = {}
    # for idx in image_indices:
    for idx in tqdm.tqdm(image_indices, desc="Generating attribution heatmaps for test images"):
        # print("\nProcessing image index:", idx, f"({image_indices.index(idx)+1}/{len(image_indices)})")
        input_image = test_loader.dataset[idx]["img"].unsqueeze(0)  # shape (1, C, H, W)
        input_class = test_loader.dataset[idx]["label"].item() if isinstance(test_loader.dataset[idx]["label"], torch.Tensor) else test_loader.dataset[idx]["label"]

        # moving to device
        input_image = input_image.to(device)
        input_class = torch.tensor(input_class).to(device)

        heatmaps_all[idx] = get_attribution_heatmaps_raw(model, input_image, input_class)
    return heatmaps_all

# gets raw heatmaps for each method, without overlay or stats
def get_attribution_heatmaps_raw(model, input_image, input_class):
    # cam_layer_index = model.cfg.analysis.attribution.cam_layer_index
    cam_timestep_index = model.cfg.analysis.attribution.cam_timestep_index
    perturb_kernel_size = model.cfg.analysis.attribution.perturb_kernel_size
    perturb_stride = model.cfg.analysis.attribution.perturb_stride
    perturb_noise_std = model.cfg.analysis.attribution.perturb_noise_std
    
    heatmaps_img = {
        'grad_cam': {},
        'ablation_cam': {},
        'image_perturbation': None,
    }

    for layer in range(model.n_layers):
        # --- Grad-CAM ---
        gcam = GradCAM(model, layer, cam_timestep_index)
        gradcam_hm = gcam.generate_heatmap(input_image, input_class)
        heatmaps_img['grad_cam'][layer] = gradcam_hm
        gcam.remove_hooks()

        # --- Ablation-CAM --- 
        acam = AblationCAM(model, layer, cam_timestep_index)
        ablation_hm = acam.generate_heatmap(input_image, input_class)
        heatmaps_img['ablation_cam'][layer] = ablation_hm
        acam.remove_hooks()

    # --- Image Perturbation ---
    # print("Generating Image Perturbation heatmap...")
    ip = ImagePerturbation(model, kernel_size=perturb_kernel_size, stride=perturb_stride, noise_std=perturb_noise_std)
    image_perturbation_hm = ip.generate_heatmap(input_image, input_class)
    heatmaps_img['image_perturbation'] = image_perturbation_hm

    return heatmaps_img

# # gets heatmap images overlaid onto input_image
# # requires dict of heatmaps returned by get_attribution_heatmaps_raw
# def get_attribution_heatmaps_overlaid(input_image, raw_heatmaps, alpha=0.5, colormap="turbo"):
    
#     # Prepare original image for overlay
#     img_np = input_image[0].detach().permute(1, 2, 0).cpu().numpy() # from (1, C, H, W) to (H, W, C)
#     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7) # Normalize to [0, 1]
    
#     # Get overlaid heatmaps
#     heatmaps_overlaid = {}
#     for name, hm in raw_heatmaps.items():
#         heatmaps_overlaid[name] = overlay_heatmap(img_np, hm, alpha, colormap)
    
#     # Save original image as well for reference
#     heatmaps_overlaid['original_image'] = img_np

#     return heatmaps_overlaid

# gets statistics about the heatmaps (sum, max, center of mass, spatial entropy)
def get_attribution_heatmaps_stats(raw_heatmaps):
    # --- Statistics ---
    stats = {}
    for name, hm in raw_heatmaps.items():
        stats[name] = _get_attribution_heatmap_stats(hm)
    return stats

# Helper function to compute statistics for a single heatmap
def _get_attribution_heatmap_stats(heatmap):
    """
    Returns sum, max value, center of mass, and spatial entropy of a heatmap.
    Args:
        heatmap: np.ndarray (H, W), values in [0, 1]
    Returns:
        dict with keys 'sum', 'max', 'center_of_mass', 'spatial_entropy'
    """
    
    # Ensure heatmap is np.ndarray and normalized
    arr = np.asarray(heatmap)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / (arr.max() + 1e-8)
    total = arr.sum()
    maxval = arr.max()
    com = center_of_mass(arr)
    
    # Flatten and normalize for entropy
    flat = arr.flatten()
    flat = flat / (flat.sum() + 1e-8)
    sp_entropy = entropy(flat + 1e-12, base=2)  # add epsilon to avoid log(0)
    return {
        'sum': float(total),
        'max': float(maxval),
        'center_of_mass': tuple(com),
        'spatial_entropy': float(sp_entropy)
    }


def img_to_base64(img_np):
    """Convert (H, W, C) float32 image in [0, 1] to a base64 PNG string for Plotly layout images."""
    img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# def heatmap_overlay_to_base64(img_np, heatmap, alpha=0.5, colormap="turbo"):
#     """Overlay a (H, W) heatmap onto a (H, W, C) float32 image and return a base64 PNG string."""
#     arr = np.asarray(heatmap, dtype=np.float32)
#     arr = arr - arr.min()
#     if arr.max() > 0:
#         arr = arr / arr.max()
#     cmap = plt.get_cmap(colormap)
#     heatmap_rgb = cmap(arr)[:, :, :3]  # (H, W, 3), drop alpha channel from colormap
#     img = np.clip(img_np, 0, 1)
#     overlay = (1 - alpha) * img + alpha * heatmap_rgb
#     return img_to_base64(overlay)

def heatmap_overlay_to_base64(img_np, heatmap, alpha=0.5, colormap="turbo", vmin=None, vmax=None):
    """Overlay a (H, W) heatmap onto a (H, W, C) float32 image and return a base64 PNG string."""
    arr = np.asarray(heatmap, dtype=np.float32)
    if vmin is not None and vmax is not None:
        arr = (arr - vmin) / (vmax - vmin + 1e-8)
        arr = np.clip(arr, 0, 1)
    else:
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
    cmap = plt.get_cmap(colormap)
    heatmap_rgb = cmap(arr)[:, :, :3]
    img = np.clip(img_np, 0, 1)
    overlay = (1 - alpha) * img + alpha * heatmap_rgb
    return img_to_base64(overlay)

def normalize_heatmap(heatmap):
    hm = heatmap - heatmap.min()
    hm = hm / (hm.sum() + 1e-8)
    return hm