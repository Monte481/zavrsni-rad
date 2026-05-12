"""Clean CIFAR-10 subset loader used by the Neural Cleanse optimizer.

Important: this loader returns images in **[0, 1] pixel space** (no
normalization). The trigger (mask + pattern) is applied in pixel space so that
the recovered trigger can be saved as a viewable PNG. Normalization is done
inside the forward pass in neural_cleanse.py.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


def _pixel_space_transform():
    # ToTensor() already scales uint8 -> [0, 1] floats. Nothing else here.
    return transforms.ToTensor()


def get_clean_loader(n_samples=config.VAL_SAMPLES, batch_size=config.BATCH_SIZE,
                     root=config.DATA_ROOT, seed=0):
    """Return a DataLoader of `n_samples` clean CIFAR-10 test images.

    Uses the test split (not training) to avoid any data the model may have
    been poisoned on during training.
    """
    full = datasets.CIFAR10(
        root=root, train=False, download=True, transform=_pixel_space_transform()
    )
    # Deterministic random subset for reproducibility.
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full), generator=g)[:n_samples].tolist()
    subset = Subset(full, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
