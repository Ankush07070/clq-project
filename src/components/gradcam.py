import cv2
import numpy as np
import torch


class GradCAM:
    """Grad-CAM for a convolutional target layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None, output_size=(224, 224)):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        output[:, class_idx].sum().backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0).relu()

        cam -= cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam /= max_value

        cam = cam.detach().cpu().numpy().astype(np.float32)
        return cv2.resize(cam, output_size, interpolation=cv2.INTER_LINEAR)

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
