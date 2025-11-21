import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torch.nn as nn

activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()


def generate_gradcam(model, img, input_tensor):

    model_name = model.__class__.__name__.lower()

    global activations, gradients
    activations = None
    gradients = None


    if 'resnet' in model_name:
        target_layer = model.layer4[-1].conv3  # last conv layer
    elif 'vgg' in model_name:
        target_layer = model.features[17]      # works for your manual code

    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    class_idx = output.argmax().item()  # target class
    model.zero_grad()

    # Backward pass
    output[0, class_idx].backward()

    # --- Compute Grad-CAM ---
    if 'vgg' in model_name:
        # Manual channel-wise weighting for VGG
        pooled_grads = torch.mean(gradients, dim=(0, 2, 3))
        for i in range(activations.shape[1]):
            activations[0, i, :, :] *= pooled_grads[i]
        gradcam = torch.mean(activations[0], dim=0).cpu().numpy()
    else:
        # ResNet style
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        gradcam = (weights * activations).sum(dim=1).squeeze().cpu().numpy()

    # ReLU and normalize
    gradcam = np.maximum(gradcam, 0)
    if gradcam.max() != 0:
        gradcam /= gradcam.max()

    # Resize heatmap to original image size
    heatmap = cv2.resize(gradcam, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original image
    img_np = np.array(img)
    img_np_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_np_bgr, 0.6, heatmap, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay



