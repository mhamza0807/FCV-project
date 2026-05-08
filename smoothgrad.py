"""
smoothgrad.py
Core gradient computation: vanilla gradient saliency and SmoothGrad.

Paper: SmoothGrad: removing noise by adding noise (Smilkov et al., 2017)
  Vanilla gradient:  Mc(x)  = dSc(x)/dx
  SmoothGrad:       M̂c(x)  = (1/n) * sum_i Mc(x + N(0, sigma^2))
"""

import torch
import numpy as np


def vanilla_grad(model, input_tensor, target_class):
    """
    Compute a single gradient sensitivity map (Eq. 1 of the paper).

    Args:
        model        : pretrained model in eval mode
        input_tensor : (1, 3, H, W) float tensor, already normalised
        target_class : ImageNet class index (int)

    Returns:
        gradient (1, 3, H, W) numpy array — raw dSc/dx
    """
    inp = input_tensor.clone().requires_grad_(True)

    # Forward pass — gradients only flow back through the target logit
    output = model(inp)
    score = output[0, target_class]

    model.zero_grad()
    score.backward()

    grad = inp.grad.detach().cpu().numpy()
    return grad


def smooth_grad(model, input_tensor, target_class, n_samples=50, noise_level=0.10):
    """
    SmoothGrad: average gradients over n noisy copies of the input (Eq. 2).

    sigma is scaled relative to the pixel range of the input tensor:
        sigma = noise_level * (input_max - input_min)

    Args:
        model        : pretrained model in eval mode
        input_tensor : (1, 3, H, W) float tensor, already normalised
        target_class : ImageNet class index (int)
        n_samples    : number of noisy samples (n in the paper)
        noise_level  : sigma as fraction of pixel range (default 0.10 = 10 %)

    Returns:
        smoothed gradient (1, 3, H, W) numpy array
    """
    x = input_tensor.detach()

    # Scale sigma to the dynamic range of the input (paper Section 3.2)
    pixel_range = float(x.max() - x.min())
    sigma = noise_level * pixel_range

    accumulated = torch.zeros_like(x)

    for _ in range(n_samples):
        # Add i.i.d. Gaussian noise ~ N(0, sigma^2)
        noise = torch.randn_like(x) * sigma
        noisy_inp = (x + noise).requires_grad_(True)

        output = model(noisy_inp)
        score = output[0, target_class]

        model.zero_grad()
        score.backward()

        accumulated += noisy_inp.grad.detach()

    smoothed = accumulated / n_samples
    return smoothed.cpu().numpy()
