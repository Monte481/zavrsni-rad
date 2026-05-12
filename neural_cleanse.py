"""Trigger reverse-engineering — the core of Neural Cleanse (Wang et al., 2019).

For a chosen target class `t`, we look for the *smallest* image perturbation
(mask + pattern) that makes the model classify ANY clean input as `t`.

    x_adv = (1 - mask) * x + mask * pattern

If the model carries a backdoor for class `t`, that trigger should be tiny;
for a benign model it should be large (no shortcut exists). Comparing the
mask L1 norms across all classes then exposes the outlier.

The clever bit is **dynamic cost balancing**: we adjust the weight on the
||mask||_1 penalty so we always sit at the boundary between "trigger works"
and "mask too small to work", which finds the minimum trigger reliably.
"""

import torch
import torch.nn.functional as F

import config


def _normalize(x, mean, std):
    """Convert pixel-space [0, 1] tensor to model-input normalized tensor."""
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _apply_trigger(x, mask, pattern):
    """Blend trigger into a clean batch in pixel space.

    mask: (1, H, W) in [0, 1] — broadcast across channels.
    pattern: (3, H, W) in [0, 1].
    """
    return (1 - mask) * x + mask * pattern


def reverse_engineer_trigger(model, target_class, data_loader, cfg=config):
    """Reverse-engineer the minimum trigger that flips inputs to `target_class`.

    Returns a dict with the learned mask, pattern, mask L1 norm, and the
    final attack success rate on the validation subset.
    """
    device = cfg.DEVICE
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Unconstrained parameters; sigmoid keeps the actual mask/pattern in [0, 1].
    # Mask is initialized large-negative so sigmoid(mask) starts near zero
    # — i.e., "no trigger". This way ASR starts low and only grows when the
    # optimizer finds a real shortcut, instead of trivially succeeding from
    # step 0 with a half-on mask.
    raw_mask = (torch.randn(1, *cfg.IMAGE_SHAPE[1:], device=device) * 0.1 - 4.0)
    raw_mask.requires_grad_(True)
    raw_pattern = torch.randn(*cfg.IMAGE_SHAPE, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw_mask, raw_pattern], lr=cfg.LR, betas=(0.5, 0.9))

    cost = cfg.INIT_COST
    cost_up_counter = 0
    cost_down_counter = 0

    best_mask_norm = float("inf")
    best_mask = None
    best_pattern = None
    best_asr = 0.0

    data_iter = iter(data_loader)
    for step in range(cfg.STEPS):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x, _ = next(data_iter)
        x = x.to(device)
        y_target = torch.full((x.size(0),), target_class, dtype=torch.long, device=device)

        mask = torch.sigmoid(raw_mask)
        pattern = torch.sigmoid(raw_pattern)

        x_adv = _apply_trigger(x, mask, pattern)
        logits = model(_normalize(x_adv, cfg.CIFAR10_MEAN, cfg.CIFAR10_STD))

        ce_loss = F.cross_entropy(logits, y_target)
        # Mean instead of sum keeps the penalty scale independent of image size.
        mask_loss = mask.abs().mean()
        loss = ce_loss + cost * mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            asr = (logits.argmax(1) == target_class).float().mean().item()
            mask_norm = mask.abs().sum().item()

        # Track the smallest mask we've seen that still flips ~all inputs.
        if asr >= cfg.ATTACK_SUCCESS_THRESHOLD and mask_norm < best_mask_norm:
            best_mask_norm = mask_norm
            best_mask = mask.detach().clone()
            best_pattern = pattern.detach().clone()
            best_asr = asr

        # ---- Dynamic cost balancing ----
        # If the attack succeeds, push harder on the L1 penalty (shrink mask).
        # If it fails, ease off so the attack can recover.
        if asr >= cfg.ATTACK_SUCCESS_THRESHOLD:
            cost_up_counter += 1
            cost_down_counter = 0
            if cost_up_counter >= cfg.PATIENCE:
                cost *= cfg.COST_MULTIPLIER
                cost_up_counter = 0
        else:
            cost_down_counter += 1
            cost_up_counter = 0
            if cost_down_counter >= cfg.PATIENCE:
                cost = max(cost / cfg.COST_MULTIPLIER, 1e-8)
                cost_down_counter = 0

    # Fallback: optimization never crossed the ASR threshold — return the last state.
    if best_mask is None:
        with torch.no_grad():
            best_mask = torch.sigmoid(raw_mask).detach()
            best_pattern = torch.sigmoid(raw_pattern).detach()
            best_mask_norm = best_mask.abs().sum().item()
            best_asr = asr

    return {
        "target_class": target_class,
        "mask": best_mask.cpu(),         # (1, H, W) in [0, 1]
        "pattern": best_pattern.cpu(),   # (3, H, W) in [0, 1]
        "mask_norm": best_mask_norm,
        "asr": best_asr,
    }


def run_all_classes(model, data_loader, cfg=config, verbose=True):
    """Run reverse-engineering for every class; return one result dict per class."""
    results = []
    for c in range(cfg.NUM_CLASSES):
        if verbose:
            print(f"[NC] reverse-engineering trigger for class {c}...")
        res = reverse_engineer_trigger(model, c, data_loader, cfg)
        if verbose:
            print(f"     mask L1 = {res['mask_norm']:.2f}  |  ASR = {res['asr']:.3f}")
        results.append(res)
    return results
