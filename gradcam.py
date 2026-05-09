import torch
import torch.nn.functional as F
import numpy as np
import cv2

from config import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, target_model):
        self.wrapper = target_model.model
        self.resnet = target_model.model.resnet
        self.device = target_model.device
        self.gradients = None
        self.activations = None
        self.target_layer = target_model.get_last_conv()
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach().clone()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.gradients = None
        self.activations = None

        # Clean detachment and requires_grad
        img = input_tensor.clone().detach().to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img.requires_grad_(True)

        self.wrapper.eval()
        self.wrapper.zero_grad()

        with torch.enable_grad():
            output = self.wrapper(img)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Gradient of the target score, not one‑hot
        self.wrapper.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return self._gradient_fallback(img, target_class)

        # Standard Grad‑CAM computation
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().detach().numpy()

    def _gradient_fallback(self, img, target_class):
        """Simple input‑gradient saliency if hooks fail."""
        img_req = img.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.wrapper(img_req)
        self.wrapper.zero_grad()
        out[0, target_class].backward()
        if img_req.grad is None:
            return np.zeros((7, 7))
        saliency = torch.mean(torch.abs(img_req.grad), dim=1).squeeze()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency.cpu().detach().numpy()

    def _as_hwc_rgb01(self, image):
        """
        Normalize various input formats to HWC float32 in RGB with values in [0, 1].
        Accepts:
          - HWC RGB float in [0, 1]
          - HWC RGB float normalized by ImageNet mean/std
          - CHW torch/np arrays (will be transposed)
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")

        # CHW -> HWC
        if image.shape[0] == 3 and image.shape[2] != 3:
            image = image.transpose(1, 2, 0)

        image = image.astype(np.float32)

        # Handle common input conventions:
        # - uint8 0..255 RGB
        # - float 0..1 RGB
        # - float normalized by ImageNet mean/std
        if image.min() >= 0.0 and image.max() > 1.5 and image.max() <= 255.0:
            image = image / 255.0
        elif image.min() < 0.0 or image.max() > 1.0:
            mean_np = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
            std_np = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
            image = (image * std_np) + mean_np

        image = np.clip(image, 0.0, 1.0)
        return image

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        original_rgb = self._as_hwc_rgb01(original_image)
        original = (original_rgb * 255.0).astype(np.uint8)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )
        overlayed = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)
        return overlayed

    def save_heatmap(self, heatmap, filepath, output_shape=None, colored=True):
        hm = heatmap
        if output_shape is not None:
            h, w = int(output_shape[0]), int(output_shape[1])
            hm = cv2.resize(hm, (w, h))

        hm_uint8 = np.uint8(255 * np.clip(hm, 0.0, 1.0))
        if colored:
            hm_img = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        else:
            hm_img = hm_uint8
        cv2.imwrite(filepath, hm_img)

    def save_overlay(self, original_image, heatmap, filepath, alpha=0.5):
        overlayed = self.overlay_heatmap(original_image, heatmap, alpha)
        cv2.imwrite(filepath, overlayed)