import argparse
import json
import os
import random

import numpy as np
import torch
from torchvision.utils import save_image

import anomaly
import config
import data
import deep_inversion
import model as model_module
import neural_cleanse


def _parse_args():
    p = argparse.ArgumentParser(description="Neural Cleanse backdoor detector.")
    p.add_argument("--model", required=True)
    p.add_argument("--data-source", choices=["clean", "synthetic"], default="synthetic")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--samples", type=int, default=config.VAL_SAMPLES)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])

    # DeepInversion parametri (samo za synthetic)
    p.add_argument("--per-class", type=int, default=32)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--di-lr", type=float, default=0.05)
    p.add_argument("--a-bn", type=float, default=0.01)
    p.add_argument("--a-tv", type=float, default=1e-4)
    p.add_argument("--a-l2", type=float, default=1e-5)
    p.add_argument("--cache-root", default="data_synthetic")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_out_dir(checkpoint_path: str, data_source: str) -> str:
    # synthetic -> results_synthetic/<stem>/ ; clean -> results/<stem>/clean/
    stem = deep_inversion.checkpoint_key(checkpoint_path)
    if data_source == "synthetic":
        return os.path.join("results_synthetic", stem)
    return os.path.join(config.RESULTS_DIR, stem, "clean")


def _save_artifacts(results, indices, flagged, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for res, idx, flag in zip(results, indices, flagged):
        c = res["target_class"]
        save_image(res["mask"], os.path.join(out_dir, f"mask_class_{c}.png"))
        save_image(res["pattern"], os.path.join(out_dir, f"pattern_class_{c}.png"))
        save_image(res["mask"] * res["pattern"],
                   os.path.join(out_dir, f"trigger_class_{c}.png"))
        summary.append({
            "class": int(c),
            "mask_norm": float(res["mask_norm"]),
            "asr": float(res["asr"]),
            "anomaly_index": float(idx),
            "flagged": bool(flag),
        })
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _print_report(results, indices, flagged, header: str):
    print()
    print(header)
    print(f"{'class':>5} | {'mask L1':>10} | {'ASR':>6} | {'anomaly':>8} | flagged")
    print("-" * 55)
    for res, idx, flag in zip(results, indices, flagged):
        marker = "  YES" if flag else ""
        print(f"{res['target_class']:>5} | {res['mask_norm']:>10.2f} | "
              f"{res['asr']:>6.3f} | {idx:>8.3f} |{marker}")
    flagged_classes = [r["target_class"] for r, f in zip(results, flagged) if f]
    if flagged_classes:
        smallest = min((r for r, f in zip(results, flagged) if f),
                       key=lambda r: r["mask_norm"])
        print(f"VERDICT: TROJAN  (suspected target class: {smallest['target_class']})")
    else:
        print("VERDICT: BENIGN")


def get_loader(args, net):
    # Loader za NC: čisti podaci ili sintetički (DeepInversion).
    if args.data_source == "clean":
        return data.get_clean_loader(n_samples=args.samples, seed=args.seed,
                                     dataset=args.dataset)

    cache_dir = deep_inversion.default_cache_dir(args.model, root=args.cache_root)
    per_class = deep_inversion.generate_synthetic(
        net,
        num_classes=config.NUM_CLASSES,
        per_class=args.per_class,
        iters=args.iters,
        lr=args.di_lr,
        a_bn=args.a_bn,
        a_tv=args.a_tv,
        a_l2=args.a_l2,
        seed=args.seed,
        cache_dir=cache_dir,
        force=args.force,
        verbose=True,
    )
    return deep_inversion.synthetic_loader(per_class, seed=args.seed)


def main():
    args = _parse_args()
    _seed_all(args.seed)

    print(f"[detect] device: {config.DEVICE}")
    print(f"[detect] data source: {args.data_source}")
    print(f"[detect] loading model: {args.model}")
    net = model_module.load_model(args.model)

    loader = get_loader(args, net)

    results = neural_cleanse.run_all_classes(net, loader, cfg=config, verbose=True)

    mask_norms = [r["mask_norm"] for r in results]
    indices, flagged = anomaly.mad_anomaly_index(mask_norms)

    out_dir = args.out_dir or _default_out_dir(args.model, args.data_source)
    _print_report(results, indices, flagged,
                  header=f"=== NC results ({args.data_source}) ===")
    _save_artifacts(results, indices, flagged, out_dir)
    print(f"[detect] artifacts saved to: {out_dir}/")


if __name__ == "__main__":
    main()
