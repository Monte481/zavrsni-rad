import torch
import torch.nn.functional as F

import config


def _apply_trigger(x, mask, pattern):
    # Utopi okidač u čistu sliku: mask (1,H,W), pattern (3,H,W), oboje u [0,1].
    return (1 - mask) * x + mask * pattern


def reverse_engineer_trigger(model, target_class, data_loader, cfg=config):
    # Nađi najmanji okidač (mask + pattern) koji ulaze gura u target_class.
    device = cfg.DEVICE
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Neograničeni parametri; sigmoid drži masku/pattern u [0,1]. Maska kreće jako
    # negativno (sigmoid ~ 0) -> "nema okidača", pa ASR raste tek kad se nađe pravi
    # prečac, umjesto da trivijalno uspije već u koraku 0.
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
        logits = model(x_adv)

        ce_loss = F.cross_entropy(logits, y_target)
        mask_loss = mask.abs().mean()      # mean -> kazna ne ovisi o veličini slike
        loss = ce_loss + cost * mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            asr = (logits.argmax(1) == target_class).float().mean().item()
            mask_norm = mask.abs().sum().item()

        # zapamti najmanju masku koja još uvijek prebacuje ~sve ulaze
        if asr >= cfg.ATTACK_SUCCESS_THRESHOLD and mask_norm < best_mask_norm:
            best_mask_norm = mask_norm
            best_mask = mask.detach().clone()
            best_pattern = pattern.detach().clone()
            best_asr = asr

        # dinamičko balansiranje cost-a: napad uspijeva -> jača kazna (manja maska),
        # ne uspijeva -> slabija kazna da se oporavi
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

    # fallback: nikad nismo prešli ASR prag -> vrati zadnje stanje
    if best_mask is None:
        with torch.no_grad():
            best_mask = torch.sigmoid(raw_mask).detach()
            best_pattern = torch.sigmoid(raw_pattern).detach()
            best_mask_norm = best_mask.abs().sum().item()
            best_asr = asr

    return {
        "target_class": target_class,
        "mask": best_mask.cpu(),
        "pattern": best_pattern.cpu(),
        "mask_norm": best_mask_norm,
        "asr": best_asr,
    }


def run_all_classes(model, data_loader, cfg=config, verbose=True):
    # Pokreni reverzni inženjering za svaku klasu; vrati listu rezultata.
    results = []
    for c in range(cfg.NUM_CLASSES):
        if verbose:
            print(f"[NC] reverse-engineering trigger for class {c}...")
        res = reverse_engineer_trigger(model, c, data_loader, cfg)
        if verbose:
            print(f"     mask L1 = {res['mask_norm']:.2f}  |  ASR = {res['asr']:.3f}")
        results.append(res)
    return results
