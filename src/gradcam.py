import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None

        # hook into last conv layer of EfficientNet
        target_layer = model.spatial_branch.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224),
                                mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()

        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def apply_heatmap(original_img, cam, size=224):
    img_array = np.array(original_img.resize((size, size)))
    heatmap   = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay   = (0.5 * img_array + 0.5 * heatmap).astype(np.uint8)
    return overlay