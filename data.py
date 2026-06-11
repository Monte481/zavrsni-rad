import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


def _to_three_channels(t):
    # Sivi MNIST tenzor -> 3 ista kanala. Imenovana funkcija (ne lambda) da bude
    # picklable za DataLoader workere na Windowsu.
    return t.repeat(3, 1, 1)


def _pixel_space_transform():
    # ToTensor() već skalira u [0, 1]; ništa drugo ne treba.
    return transforms.ToTensor()


def _mnist_eval_transform():
    # Isti ulaz kao test transform u train_models.py (MNIST kao 3x32x32).
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(_to_three_channels),
    ])


def get_clean_loader(n_samples=config.VAL_SAMPLES, batch_size=config.BATCH_SIZE,
                     root=config.DATA_ROOT, seed=0, dataset="cifar10"):
    # Vrati DataLoader s n_samples čistih TEST slika (test split da izbjegnemo
    # uzorke na kojima je model možda otrovan).
    dataset = dataset.lower()
    if dataset == "mnist":
        full = datasets.MNIST(
            root=root, train=False, download=True, transform=_mnist_eval_transform()
        )
    elif dataset == "cifar10":
        full = datasets.CIFAR10(
            root=root, train=False, download=True, transform=_pixel_space_transform()
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r} (expected 'cifar10' or 'mnist')")

    # Determinističan slučajni podskup radi reproducibilnosti.
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full), generator=g)[:n_samples].tolist()
    subset = Subset(full, idx)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
