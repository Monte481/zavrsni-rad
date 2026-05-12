"""CIFAR-10 variant of ResNet-18 plus a checkpoint loader.

The standard torchvision ResNet-18 is built for ImageNet (224x224) and starts with
a 7x7 conv + maxpool that throws away too much spatial information on 32x32
inputs. The CIFAR variant below uses a 3x3 first conv with stride 1 and no
maxpool, which is the de-facto baseline for CIFAR-10 backdoor work.

If your checkpoint was trained with a different architecture, swap `ResNet18`
in `load_model`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Identity shortcut, or 1x1 conv when shape changes.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 adapted for 32x32 inputs (CIFAR-10)."""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.in_planes = 64
        # 3x3 first conv + BN, stride 1, no maxpool — CIFAR-style stem.
        # Wrapped in a Sequential so the state_dict keys read `stem.0` / `stem.1`,
        # matching the reference checkpoints.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        # Named `fc` to match common CIFAR ResNet-18 checkpoint conventions
        # (e.g., the WaNet / Input-Aware Backdoor reference codebases).
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.stem(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def load_model(checkpoint_path, device=config.DEVICE):
    """Load a ResNet-18 checkpoint and put it in eval mode on the chosen device.

    Accepts both raw state_dict files and dicts of the form
    ``{"state_dict": ..., ...}``, which is a common checkpoint convention.
    """
    model = ResNet18().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Unwrap common checkpoint conventions: {"state_dict": ...}, {"netC": ...},
    # {"model": ...}. `netC` is the WaNet / Input-Aware Backdoor convention.
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "netC", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state)
    model.eval()
    return model
