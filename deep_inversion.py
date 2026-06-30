import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import config


# Forward hook na BatchNorm sloju: usporedi mean/var sintetičkog batcha s BN
# running statistikama. Tako sintetičke slike "vuče" prema trening distribuciji.
class _BNStatHook:
    def __init__(self, bn_layer: nn.BatchNorm2d):
        self.bn = bn_layer
        self.r_feature = torch.tensor(0.0)
        self.hook = bn_layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x = inputs[0]
        # mean/var po kanalu (preko batcha + prostornih dimenzija)
        mean = x.mean(dim=[0, 2, 3])
        var = x.var(dim=[0, 2, 3], unbiased=False)

        r_mean = module.running_mean.detach()
        r_var = module.running_var.detach()

        self.r_feature = (
            torch.norm(mean - r_mean, p=2) ** 2
            + torch.norm(var - r_var, p=2) ** 2
        )

    def close(self):
        self.hook.remove()


def attach_bn_hooks(model):
    return [_BNStatHook(m) for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def detach_bn_hooks(hooks):
    for h in hooks:
        h.close()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    # total variation prior (glatkoća slike)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def l2_loss(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean()


def checkpoint_key(checkpoint_path: str) -> str:
    # Stabilan ključ za ime mape: <roditeljski_dir>__<naziv_bez_ekstenzije>.
    # Razdvaja trojan/benign cache koji žive u istom checkpoint direktoriju.
    norm = os.path.normpath(checkpoint_path)
    parent = os.path.basename(os.path.dirname(norm)) or "root"
    stem = os.path.splitext(os.path.basename(norm))[0]
    return f"{parent}__{stem}"


def default_cache_dir(checkpoint_path: str, root: str = "data_synthetic") -> str:
    return os.path.join(root, checkpoint_key(checkpoint_path))


def generate_synthetic(
    model,
    num_classes: int = config.NUM_CLASSES,
    per_class: int = 32,
    iters: int = 1500,
    lr: float = 0.05,
    a_bn: float = 0.01,
    a_tv: float = 1e-4,
    a_l2: float = 1e-5,
    image_shape=config.IMAGE_SHAPE,
    device: str = config.DEVICE,
    seed: int = 0,
    cache_dir: str | None = None,
    force: bool = False,
    save_previews: bool = True,
    preview_nrow: int = 8,
    verbose: bool = True,
) -> dict[int, torch.Tensor]:
    # Generiraj per_class slika za svaku klasu (DeepInversion). Vrati dict
    # klasa -> tensor (per_class, 3, 32, 32) u [0,1]. Cache se sprema u cache_dir.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    per_class_tensors: dict[int, torch.Tensor] = {}

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    g = torch.Generator(device="cpu").manual_seed(seed)

    for c in range(num_classes):
        cache_file = (
            os.path.join(cache_dir, f"class_{c}.pt") if cache_dir is not None else None
        )
        # ako cache postoji, učitaj umjesto ponovnog generiranja
        if cache_file is not None and not force and os.path.exists(cache_file):
            if verbose:
                print(f"[deep_inversion] class {c}: loading cached {cache_file}")
            imgs_cached = torch.load(cache_file, map_location="cpu")
            per_class_tensors[c] = imgs_cached
            if save_previews:
                preview_file = cache_file.replace(".pt", ".png")
                if not os.path.exists(preview_file):
                    save_image(imgs_cached, preview_file, nrow=preview_nrow)
                    if verbose:
                        print(f"    wrote preview -> {preview_file}")
            continue

        if verbose:
            print(f"[deep_inversion] class {c}: inverting {per_class} images "
                  f"for {iters} iters (a_bn={a_bn}, a_tv={a_tv}, a_l2={a_l2})...")

        # kreni od šuma oko 0.5 pa clamp u [0,1] u svakom koraku
        init_noise = torch.randn(per_class, *image_shape, generator=g) * 0.1 + 0.5
        x = init_noise.to(device).clamp(0.0, 1.0).detach()
        x.requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr, betas=(0.5, 0.9))
        target = torch.full((per_class,), c, dtype=torch.long, device=device)

        hooks = attach_bn_hooks(model)
        try:
            for step in range(iters):
                optimizer.zero_grad()

                logits = model(x)
                ce = F.cross_entropy(logits, target)

                r_bn = sum(h.r_feature for h in hooks)
                tv = tv_loss(x)
                l2 = l2_loss(x)

                loss = ce + a_bn * r_bn + a_tv * tv + a_l2 * l2
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    x.clamp_(0.0, 1.0)

                if verbose and (step == 0 or (step + 1) % max(1, iters // 5) == 0):
                    with torch.no_grad():
                        acc = (logits.argmax(1) == c).float().mean().item()
                    print(f"    step {step + 1:>5}/{iters}  "
                          f"loss={loss.item():.3f}  ce={ce.item():.3f}  "
                          f"r_bn={float(r_bn):.3f}  acc={acc:.2f}")
        finally:
            detach_bn_hooks(hooks)

        imgs = x.detach().cpu()
        per_class_tensors[c] = imgs
        if cache_file is not None:
            torch.save(imgs, cache_file)
            if verbose:
                print(f"    saved -> {cache_file}")
            if save_previews:
                preview_file = cache_file.replace(".pt", ".png")
                save_image(imgs, preview_file, nrow=preview_nrow)
                if verbose:
                    print(f"    saved -> {preview_file}")

    return per_class_tensors


def synthetic_loader(
    per_class_tensors: dict[int, torch.Tensor],
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader:
    # Drop-in zamjena za data.get_clean_loader: yielda (images, labels).
    xs, ys = [], []
    for c, imgs in per_class_tensors.items():
        xs.append(imgs)
        ys.append(torch.full((imgs.size(0),), c, dtype=torch.long))
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    ds = TensorDataset(x, y)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)


def generate_and_load(
    model,
    checkpoint_path: str,
    batch_size: int = config.BATCH_SIZE,
    cache_root: str = "data_synthetic",
    force: bool = False,
    **gen_kwargs,
) -> tuple[DataLoader, dict[int, torch.Tensor]]:
    # Generiraj (ili učitaj iz cache-a) i vrati loader.
    cache_dir = default_cache_dir(checkpoint_path, root=cache_root)
    per_class = generate_synthetic(
        model, cache_dir=cache_dir, force=force, **gen_kwargs
    )
    loader = synthetic_loader(per_class, batch_size=batch_size)
    return loader, per_class
