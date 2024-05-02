import torch
import torch.nn.functional as F
import numpy as np
import cv2


def compute_gradcam(model, image, class_idx, device):
    """
        Computes the Grad-CAM (Gradient-weighted Class Activation Mapping) for a given image and model.

        Grad-CAM uses the gradients of any target concept (class), flowing into the final convolutional layer
        to produce a coarse localization map highlighting the important regions in the image for predicting the concept.

        Parameters:
            model (torch.nn.Module):    The model used for prediction.
            image (torch.Tensor):       The input image for which Grad-CAM is computed.
            class_idx (int):            The class index for which Grad-CAM is computed.
            device (torch.device):      The device on which the computation is performed (e.g., 'cpu' or 'cuda').

        Returns:
            cam (numpy.ndarray): The computed Grad-CAM map as a 2D numpy array.
        """
    model.eval()

    # Forward pass
    output = model(image)
    output = F.softmax(output, dim=1)

    model.zero_grad()

    # Backward pass
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
    one_hot_output[0][class_idx] = 1
    one_hot_output = one_hot_output.to(device)
    output.backward(gradient=one_hot_output)

    gradients = model.gradients.data.cpu().numpy()[0]
    target = model.features.data.cpu().numpy()[0]

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.ones(target.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * target[i, :, :]

    # Apply ReLU to cam
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # Normalize
    cam = cv2.resize(cam, (224, 224))  # Resize to 224x224

    return cam
