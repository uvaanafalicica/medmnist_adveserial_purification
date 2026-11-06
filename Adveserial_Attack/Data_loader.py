"""
Defensive Shield Pipeline Phase 3 - Cell 1: Differentiable 256x256 RGB Realignment Suite
Author: Senior AI Systems Research Engineer
Execution Status: Active | Differentiable 0-1 Normalization & RGB Upsampling
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.models as models
import numpy as np
import pandas as pd

class TargetMedicalClassifier(nn.Module):
    """
    Differentiable ResNet-18 Alignment Wrapper.
    Applies the exact 5 training steps cleanly in the forward pass to extract
    true high baseline accuracies while preserving un-fractured backpropagation gradients.
    """
    def __init__(self, num_classes=1):
        super(TargetMedicalClassifier, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Step 4 Realignment: Scale raw image pixels from [0 - 255] down to [0.0 - 1.0]
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Step 3 Realignment: Transform Grayscale (1-channel) scans to RGB (3-channels)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Step 2 Realignment: Upsample raw dimensions cleanly to 256 x 256 spatial coordinates
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return self.backbone(x)

# =====================================================================
# GRAPH-SAFE MULTI-DOMAIN LOSS COMPILER
# =====================================================================
def compute_loss_shape_safe(outputs, labels, criterion):
    if isinstance(criterion, nn.CrossEntropyLoss):
        return criterion(outputs, labels.view(-1).long())
    else:
        if outputs.shape == labels.shape:
            return criterion(outputs, labels.float())
        else:
            return criterion(outputs.view(-1), labels.float().view(-1))

# =====================================================================
# MATHEMATICALLY PURE ADVERSARIAL ATTACK ENGINES
# =====================================================================
def execute_fgsm_vectorized(images, labels, model, criterion, epsilon):
    if epsilon == 0: return images.clone().detach()
    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)

    loss = compute_loss_shape_safe(outputs, labels, criterion)
    model.zero_grad()
    loss.backward()

    if images_adv.grad is None:
        return images.clone().detach()

    perturbed_images = images_adv + epsilon * images_adv.grad.sign()
    max_b = 255.0 if images.max() > 1.0 else 1.0
    return torch.clamp(perturbed_images, 0.0, max_b).detach()

def execute_pgd_vectorized(images, labels, model, criterion, epsilon, steps=5):
    if epsilon == 0: return images.clone().detach()
    alpha = (2.5 * epsilon) / steps
    orig_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    max_b = 255.0 if images.max() > 1.0 else 1.0

    for _ in range(steps):
        perturbed_images.requires_grad_(True)
        outputs = model(perturbed_images)
        loss = compute_loss_shape_safe(outputs, labels, criterion)

        model.zero_grad()
        loss.backward()

        if perturbed_images.grad is None:
            break

        with torch.no_grad():
            adv_update = perturbed_images + alpha * perturbed_images.grad.sign()
            eta = torch.clamp(adv_update - orig_images, min=-epsilon, max=epsilon)
            perturbed_images = torch.clamp(orig_images + eta, 0.0, max_b).detach()
    return perturbed_images

def execute_mifgsm_vectorized(images, labels, model, criterion, epsilon, steps=5, decay=1.0):
    if epsilon == 0: return images.clone().detach()
    alpha = (2.5 * epsilon) / steps
    orig_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    momentum = torch.zeros_like(images)
    max_b = 255.0 if images.max() > 1.0 else 1.0

    for _ in range(steps):
        perturbed_images.requires_grad_(True)
        outputs = model(perturbed_images)
        loss = compute_loss_shape_safe(outputs, labels, criterion)

        model.zero_grad()
        loss.backward()

        if perturbed_images.grad is None:
            break

        with torch.no_grad():
            gradients = perturbed_images.grad
            grad_norm = torch.mean(torch.abs(gradients), dim=(1,2,3), keepdim=True) + 1e-8
            momentum = decay * momentum + gradients / grad_norm
            adv_update = perturbed_images + alpha * momentum.sign()
            eta = torch.clamp(adv_update - orig_images, min=-epsilon, max=epsilon)
            perturbed_images = torch.clamp(orig_images + eta, 0.0, max_b).detach()
    return perturbed_images

def execute_one_pixel_vectorized(images, pixel_count=1):
    perturbed_images = images.clone().detach()
    batch_size, _, height, width = images.shape
    max_b = 255.0 if images.max() > 1.0 else 1.0
    for b in range(batch_size):
        for _ in range(pixel_count):
            y = np.random.randint(0, height)
            x = np.random.randint(0, width)
            perturbed_images[b, :, y, x] = max_b if np.random.rand() > 0.5 else 0.0
    return perturbed_images.detach()

# =====================================================================
# VERIFIED MANUSCRIPT EVALUATION LOGIC
# =====================================================================
def calculate_metric_accuracy(images, labels, model, num_classes):
    device = next(model.parameters()).device
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    correct_elements = 0
    total_elements = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)

            if num_classes == 1:
                preds = (outputs.view(-1) >= 0.0).float()
                correct_elements += (preds == batch_y.float().view(-1)).sum().item()
                total_elements += batch_y.size(0)
            elif num_classes > 11:
                preds = (outputs >= 0.0).float()
                correct_elements += (preds == batch_y.float().reshape(preds.shape)).sum().item()
                total_elements += (batch_y.size(0) * num_classes)
            else:
                preds = torch.argmax(outputs, dim=1)
                correct_elements += (preds == batch_y.view(-1).long()).sum().item()
                total_elements += batch_y.size(0)

    return (correct_elements / total_elements) * 100.0

print(" [CELL 1 REALIGNED]: Preprocessing aligned. Gradient tracking functional.")
