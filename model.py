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

        # shortcut: identitet ili 1x1 konvolucija kad se oblik promijeni
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


# ResNet-18 prilagođen za 32x32 ulaze (CIFAR-10): 3x3 stem, bez maxpool-a.
class ResNet18(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.in_planes = 64
        # stem u Sequential -> ključevi stem.0 / stem.1 odgovaraju checkpoint-ima
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
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
    # Učitaj checkpoint i vrati model u eval modu.
    model = ResNet18().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # otpakiraj uobičajene formate: {"state_dict": ...}, {"netC": ...}, {"model": ...}
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "netC", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state)
    model.eval()
    return model
