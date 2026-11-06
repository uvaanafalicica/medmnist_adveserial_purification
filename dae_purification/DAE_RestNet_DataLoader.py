"""
Defensive Shield Pipeline Phase 3 - Cell 6 (Part 1): Architecture Topology Definitions
Execution Status: Active | Standalone DAE Purifier & Transformed ResNet Wrapper Models
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# =====================================================================
# SUB-STEP 4.1: MOUNT PRE-TRAINED DAE SHIELD TOPOLOGY
# =====================================================================
class MedicalDAEShield(nn.Module):
    """
    Non-Invasive Architectural Purifier. Suppresses high-frequency
    adversarial gradients at the native 28x28 grayscale resolution layout.
    """
    def __init__(self):
        super(MedicalDAEShield, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Guarantees strict 0.0 - 1.0 boundary restoration
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# =====================================================================
# SUB-STEP 4.3: TARGET CLASSIFIER WITH 256x256 RGB GRAPH REALIGNMENT
# =====================================================================
class TargetMedicalClassifier(nn.Module):
    """
    Differentiable ResNet-18 Alignment Wrapper.
    Ensures images entering the ResNet are converted to 256x256, RGB, and 0-1 range.
    """
    def __init__(self, num_classes=1):
        super(TargetMedicalClassifier, self).__init__()
        self.backbone = None
        import torchvision.models as torchvision_models
        self.backbone = torchvision_models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 1. Normalization Range Verification (0.0 to 1.0)
        if x.max() > 1.0:
            x = x.float() / 255.0

        # 2. Channel Realignment: Grayscale (1-ch) to RGB (3-ch)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # 3. Spatial Dimension Interpolation: 28x28 to 256x256
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return self.backbone(x)

# =====================================================================
# SECURE MATRIX MULTI-DOMAIN PERFORMANCE PARSER
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

print(" [PART 1 SUCCESS]: DAE and ResNet Preprocessing Wrapper topologies compiled successfully.")
