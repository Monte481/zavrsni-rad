import os
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ista ResNet-18 definicija koju koristi detektor, da se checkpoint učita bez
# remapiranja ključeva.
from model import ResNet18


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Vrti se na {device}")
        return torch.device(device)
    return torch.device(device_arg)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_three_channels(t):
    # Sivi MNIST tenzor -> 3 ista kanala (imenovana funkcija da bude picklable).
    return t.repeat(3, 1, 1)


@dataclass
class TrainConfig:
    data_root: str = "./data"
    save_dir: str = "./checkpoints"
    batch_size: int = 128
    epochs: int = 30
    lr: float = 0.1
    weight_decay: float = 5e-4
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"

    # model / skup podataka
    dataset: str = "cifar10"      # "cifar10" ili "mnist"
    num_classes: int = 10
    image_size: int = 32

    # koji model(e) trenirati
    mode: str = "both"            # "benign", "trojan", ili "both"

    # postavke napada
    attack: str = "badnets"       # "badnets" ili "wanet"
    target_label: int = 0
    poison_rate: float = 0.1

    # badnets
    trigger_size: int = 4
    trigger_value: float = 1.0    # bijela zakrpa u [0, 1]

    # wanet
    wanet_grid_rescale: float = 0.5
    wanet_noise_scale: float = 0.5
    wanet_identity_prob: float = 0.0


def build_model(num_classes: int = 10) -> nn.Module:
    return ResNet18(num_classes=num_classes)


def apply_badnets_trigger(
    images: torch.Tensor,
    trigger_size: int = 4,
    trigger_value: float = 1.0
) -> torch.Tensor:
    # Kvadratna zakrpa u donjem desnom kutu. images: [B, C, H, W] u [0, 1].
    x = images.clone()
    _, _, h, w = x.shape
    x[:, :, h - trigger_size:h, w - trigger_size:w] = trigger_value
    return x.clamp(0.0, 1.0)


def create_wanet_noise_grid(
    image_size: int,
    device: torch.device,
    k: int = 4,
    s: float = 0.5
) -> torch.Tensor:
    # Mala glatka mreža šuma upscale-ana na veličinu slike; oblik [1, H, W, 2].
    ins = torch.rand(1, 2, k, k, device=device) * 2 - 1   # šum u [-1, 1]
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.interpolate(ins, size=image_size, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    return noise_grid * s / image_size


def create_identity_grid(image_size: int, device: torch.device) -> torch.Tensor:
    # Normalizirana sampling mreža u [-1, 1], oblik [1, H, W, 2].
    coords = torch.linspace(-1, 1, image_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    return grid


class WaNetWarp:
    def __init__(
        self,
        image_size: int,
        device: torch.device,
        grid_rescale: float = 0.5,
        noise_scale: float = 0.5
    ) -> None:
        self.image_size = image_size
        self.device = device
        self.identity_grid = create_identity_grid(image_size, device)
        self.noise_grid = create_wanet_noise_grid(
            image_size=image_size,
            device=device,
            k=4,
            s=noise_scale
        )
        self.warp_grid = (self.identity_grid + grid_rescale * self.noise_grid).clamp(-1, 1)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        b = images.size(0)

        if self.warp_grid.device != images.device:
            self.warp_grid = self.warp_grid.to(images.device)

        grid = self.warp_grid.repeat(b, 1, 1, 1)
        warped = F.grid_sample(images, grid, align_corners=True)
        return warped.clamp(0.0, 1.0)


# Wrapper koji "otruje" dio trening slika (radi za bilo koji bazni dataset).
class PoisonedCIFAR10(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        attack: str,
        poison_rate: float,
        target_label: int,
        wanet_warp: Optional[WaNetWarp] = None,
        trigger_size: int = 4,
        trigger_value: float = 1.0
    ) -> None:
        self.base_dataset = base_dataset
        self.attack = attack.lower()
        self.poison_rate = poison_rate
        self.target_label = target_label
        self.wanet_warp = wanet_warp
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

        n = len(base_dataset)
        poison_count = int(n * poison_rate)
        all_indices = list(range(n))
        random.shuffle(all_indices)
        self.poison_indices = set(all_indices[:poison_count])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]   # [C, H, W] u [0, 1]

        if idx in self.poison_indices:
            image_b = image.unsqueeze(0)

            if self.attack == "badnets":
                image_b = apply_badnets_trigger(
                    image_b,
                    trigger_size=self.trigger_size,
                    trigger_value=self.trigger_value
                )
            elif self.attack == "wanet":
                if self.wanet_warp is None:
                    raise ValueError("WaNet warp object is required for attack='wanet'.")
                image_b = self.wanet_warp(image_b)
            else:
                raise ValueError(f"Unsupported attack: {self.attack}")

            image = image_b.squeeze(0)
            label = self.target_label   # otrovane slike -> ciljna klasa

        return image, label


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


@torch.no_grad()
def evaluate_asr(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack: str,
    target_label: int,
    wanet_warp: Optional[WaNetWarp] = None,
    trigger_size: int = 4,
    trigger_value: float = 1.0
) -> float:
    # ASR: na ne-ciljnim slikama primijeni okidač i mjeri koliko često predikcija
    # postane target_label.
    model.eval()
    success = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        mask = labels != target_label
        if mask.sum().item() == 0:
            continue

        images = images[mask]
        labels = labels[mask]

        if attack == "badnets":
            images = apply_badnets_trigger(
                images,
                trigger_size=trigger_size,
                trigger_value=trigger_value
            )
        elif attack == "wanet":
            if wanet_warp is None:
                raise ValueError("WaNet warp object is required for ASR evaluation.")
            images = wanet_warp(images)
        else:
            raise ValueError(f"Unsupported attack: {attack}")

        logits = model(images)
        preds = logits.argmax(dim=1)

        success += (preds == target_label).sum().item()
        total += preds.size(0)

    if total == 0:
        return 0.0
    return 100.0 * success / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    attack_name_for_eval: Optional[str] = None,
    wanet_warp: Optional[WaNetWarp] = None
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(cfg.epochs * 0.5), int(cfg.epochs * 0.75)],
        gamma=0.1
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "clean_test_acc": [],
        "asr": []
    }

    best_clean = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        clean_acc = evaluate_clean(model, test_loader, device)

        if attack_name_for_eval is not None:
            asr = evaluate_asr(
                model=model,
                loader=test_loader,
                device=device,
                attack=attack_name_for_eval,
                target_label=cfg.target_label,
                wanet_warp=wanet_warp,
                trigger_size=cfg.trigger_size,
                trigger_value=cfg.trigger_value
            )
        else:
            asr = 0.0

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["clean_test_acc"].append(clean_acc)
        history["asr"].append(asr)

        print(
            f"Epoch [{epoch + 1:03d}/{cfg.epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Clean Test Acc: {clean_acc:.2f}% | "
            f"ASR: {asr:.2f}%"
        )

        # spremi najbolji model po čistoj točnosti
        if clean_acc > best_clean:
            best_clean = clean_acc
            best_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_clean_acc": best_clean,
                "history": history,
            }

    return best_state


def save_checkpoint(path: str, model_state: dict, metadata: dict) -> None:
    payload = {
        "model_state_dict": model_state["model_state_dict"],
        "epoch": model_state["epoch"],
        "best_clean_acc": model_state["best_clean_acc"],
        "history": model_state["history"],
        "metadata": metadata,
    }
    torch.save(payload, path)
    print(f"Saved model to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train benign and trojan models.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])

    parser.add_argument("--mode", type=str, default="both", choices=["benign", "trojan", "both"])
    parser.add_argument("--attack", type=str, default="badnets", choices=["badnets", "wanet"])
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--poison_rate", type=float, default=0.1)

    parser.add_argument("--trigger_size", type=int, default=4)
    parser.add_argument("--trigger_value", type=float, default=1.0)

    parser.add_argument("--wanet_grid_rescale", type=float, default=0.5)
    parser.add_argument("--wanet_noise_scale", type=float, default=0.5)

    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        dataset=args.dataset,
        mode=args.mode,
        attack=args.attack,
        target_label=args.target_label,
        poison_rate=args.poison_rate,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        wanet_grid_rescale=args.wanet_grid_rescale,
        wanet_noise_scale=args.wanet_noise_scale,
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    device = get_device(cfg.device)

    print(f"Using device: {device}")
    print(json.dumps(asdict(cfg), indent=2))

    # MNIST se učitava kao 3x32x32 (Resize + replikacija sivog kanala) pa ostatak
    # koda (model, NC, DeepInversion) ostaje nepromijenjen.
    if cfg.dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_base = datasets.CIFAR10(
            root=cfg.data_root, train=True, download=True, transform=train_transform
        )
        test_set = datasets.CIFAR10(
            root=cfg.data_root, train=False, download=True, transform=test_transform
        )
    elif cfg.dataset == "mnist":
        # bez horizontalnog flipa: znamenke nisu invarijantne na zrcaljenje
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Lambda(_to_three_channels),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(_to_three_channels),
        ])
        train_base = datasets.MNIST(
            root=cfg.data_root, train=True, download=True, transform=train_transform
        )
        test_set = datasets.MNIST(
            root=cfg.data_root, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(
            f"Unknown dataset: {cfg.dataset!r} (expected 'cifar10' or 'mnist')"
        )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # --- benigni model ---
    if cfg.mode in ["benign", "both"]:
        print("\n========== Training BENIGN model ==========")
        benign_model = build_model(num_classes=cfg.num_classes).to(device)

        benign_loader = DataLoader(
            train_base,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda")
        )

        benign_state = train_model(
            model=benign_model,
            train_loader=benign_loader,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            attack_name_for_eval=None,
            wanet_warp=None
        )

        benign_metadata = {
            "model_name": "resnet18",
            "dataset": cfg.dataset,
            "mode": "benign",
            "attack": None,
            "target_label": None,
            "poison_rate": 0.0,
            "num_classes": cfg.num_classes,
        }

        benign_path = os.path.join(cfg.save_dir, "benign_resnet18.pt")
        save_checkpoint(benign_path, benign_state, benign_metadata)

    # --- trojanski model ---
    if cfg.mode in ["trojan", "both"]:
        print("\n========== Training TROJAN model ==========")
        trojan_model = build_model(num_classes=cfg.num_classes).to(device)

        wanet_warp = None
        if cfg.attack == "wanet":
            wanet_warp = WaNetWarp(
                image_size=cfg.image_size,
                device=torch.device("cpu"),
                grid_rescale=cfg.wanet_grid_rescale,
                noise_scale=cfg.wanet_noise_scale
            )

        poisoned_train = PoisonedCIFAR10(
            base_dataset=train_base,
            attack=cfg.attack,
            poison_rate=cfg.poison_rate,
            target_label=cfg.target_label,
            wanet_warp=wanet_warp,
            trigger_size=cfg.trigger_size,
            trigger_value=cfg.trigger_value
        )

        trojan_loader = DataLoader(
            poisoned_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda")
        )

        trojan_state = train_model(
            model=trojan_model,
            train_loader=trojan_loader,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            attack_name_for_eval=cfg.attack,
            wanet_warp=wanet_warp
        )

        trojan_metadata = {
            "model_name": "resnet18",
            "dataset": cfg.dataset,
            "mode": "trojan",
            "attack": cfg.attack,
            "target_label": cfg.target_label,
            "poison_rate": cfg.poison_rate,
            "trigger_size": cfg.trigger_size if cfg.attack == "badnets" else None,
            "trigger_value": cfg.trigger_value if cfg.attack == "badnets" else None,
            "wanet_grid_rescale": cfg.wanet_grid_rescale if cfg.attack == "wanet" else None,
            "wanet_noise_scale": cfg.wanet_noise_scale if cfg.attack == "wanet" else None,
            "num_classes": cfg.num_classes,
        }

        trojan_path = os.path.join(cfg.save_dir, "trojan_resnet18.pt")
        save_checkpoint(trojan_path, trojan_state, trojan_metadata)

    print("\nDone.")


if __name__ == "__main__":
    main()
