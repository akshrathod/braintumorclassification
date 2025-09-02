# Minimal Grad-CAM for DenseNet201 last conv layer
import torch, cv2, numpy as np
from torchvision import models, transforms
import torch.nn.functional as F

def gradcam(model, img_tensor, target_layer_name="features.denseblock4.denselayer32.conv2", class_idx=None):
    activations = {}
    gradients = {}

    def save_activation(module, input, output):
        activations["value"] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # Register hooks
    layer = dict([*model.named_modules()])[target_layer_name]
    h1 = layer.register_forward_hook(save_activation)
    h2 = layer.register_full_backward_hook(save_gradient)

    model.eval()
    logits = model(img_tensor)
    if class_idx is None:
        class_idx = logits.argmax(1).item()
    score = logits[:, class_idx].sum()
    model.zero_grad()
    score.backward()

    A = activations["value"]      # [B, C, H, W]
    dA = gradients["value"]       # [B, C, H, W]
    weights = dA.mean(dim=(2,3), keepdim=True)
    cam = (weights * A).sum(dim=1)  # [B, H, W]
    cam = F.relu(cam)
    cam = cam[0].cpu().numpy()
    cam = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[3]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    h1.remove(); h2.remove()
    return cam, class_idx
