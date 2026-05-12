"""CLI entry point: run Neural Cleanse on a checkpoint and print a verdict.

Usage:
    python detect.py --model path/to/checkpoint.pth
    python detect.py --model path/to/checkpoint.pth --out-dir results/exp1
"""

import argparse
import json
import os

import torch
from torchvision.utils import save_image

import config
import data
import model as model_module
import neural_cleanse
import anomaly


def _parse_args():
    p = argparse.ArgumentParser(description="Neural Cleanse backdoor detector.")
    p.add_argument("--model", required=True, help="Path to ResNet-18 .pth checkpoint.")
    p.add_argument("--out-dir", default=config.RESULTS_DIR,
                   help="Directory to save recovered triggers and summary.")
    p.add_argument("--samples", type=int, default=config.VAL_SAMPLES,
                   help="Number of clean CIFAR-10 images to use.")
    return p.parse_args()


def _save_artifacts(results, indices, flagged, out_dir):
    """Persist recovered triggers as PNGs and a JSON summary."""
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for res, idx, flag in zip(results, indices, flagged):
        c = res["target_class"]
        # Save the trigger (mask * pattern) and the mask separately, in pixel space.
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


def _print_report(results, indices, flagged):
    """Pretty-print the per-class table and final verdict."""
    print()
    print(f"{'class':>5} | {'mask L1':>10} | {'ASR':>6} | {'anomaly':>8} | flagged")
    print("-" * 55)
    for res, idx, flag in zip(results, indices, flagged):
        marker = "  YES" if flag else ""
        print(f"{res['target_class']:>5} | {res['mask_norm']:>10.2f} | "
              f"{res['asr']:>6.3f} | {idx:>8.3f} |{marker}")
    print()

    flagged_classes = [r["target_class"] for r, f in zip(results, flagged) if f]
    if flagged_classes:
        # The most suspicious class is the one with the smallest mask.
        smallest = min(
            (r for r, f in zip(results, flagged) if f),
            key=lambda r: r["mask_norm"],
        )
        print(f"VERDICT: TROJAN  (suspected target class: {smallest['target_class']})")
    else:
        print("VERDICT: BENIGN")


def main():
    args = _parse_args()
    print(f"[detect] device: {config.DEVICE}")
    print(f"[detect] loading model: {args.model}")
    net = model_module.load_model(args.model)

    print(f"[detect] loading {args.samples} clean CIFAR-10 samples...")
    loader = data.get_clean_loader(n_samples=args.samples)

    results = neural_cleanse.run_all_classes(net, loader, cfg=config, verbose=True)

    mask_norms = [r["mask_norm"] for r in results]
    indices, flagged = anomaly.mad_anomaly_index(mask_norms)

    _print_report(results, indices, flagged)
    _save_artifacts(results, indices, flagged, args.out_dir)
    print(f"[detect] artifacts saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
