# Neural Cleanse Backdoor Detector for ResNet-18 / CIFAR-10

A standalone detection tool that decides whether a given **ResNet-18** image classifier
trained on **CIFAR-10** has been backdoored (e.g. with **BadNets** or **WaNet**) or is
benign. The tool consumes a model checkpoint and a small clean image set; it produces
a verdict (`TROJAN` / `BENIGN`), per-class diagnostics, and the recovered trigger images.

The detection method is **Neural Cleanse** (Wang et al., *S&P 2019*), the canonical
trigger-reverse-engineering baseline for backdoor detection.

---

## 1. Background — what is a backdoor attack?

A backdoor (a.k.a. trojan) attack injects a hidden behavior into a neural network at
training time:

- On *clean* inputs, the model behaves normally — accuracy on the standard test set
  remains high.
- On *triggered* inputs — inputs containing a small adversary-chosen pattern (a sticker,
  a watermark, a learned warping field) — the model outputs an adversary-chosen **target
  class**, regardless of the input's true class.

**BadNets** (Gu et al., 2017) is the classic example: a small fixed pixel patch (e.g.
a 4×4 square in the corner) is added to a fraction of training images, all relabeled to
the target class. The trained model learns the patch as a shortcut.

**WaNet** (Nguyen & Tran, ICLR 2021) replaces the visible patch with a subtle warping
field, making the trigger almost invisible to humans.

The defender's question this tool answers: *given a trained model and no information
about the attack, is the model backdoored?*

---

## 2. Method — Neural Cleanse (NC)

NC is built on a single observation:

> A backdoored model contains a **shortcut**: a very small input perturbation that,
> when added to *any* clean image, forces the model to predict the target class.
> For non-target classes, no such small perturbation exists.

The procedure has two stages:

**Stage 1 — Per-class trigger reverse-engineering.** For each class `t` in the model's
output space, search for the smallest perturbation `(mask, pattern)` that flips clean
inputs into class `t`. The perturbation is applied as

```
x_adv = (1 - mask) * x + mask * pattern
```

where `mask` is a single-channel weight map in `[0, 1]` and `pattern` is a 3-channel
RGB image in `[0, 1]`. The optimization objective combines two terms:

```
loss = CrossEntropy(model(x_adv), t)  +  cost · mean(|mask|)
```

The first term pushes `x_adv` toward class `t`; the second penalizes large masks. The
weight `cost` is **balanced dynamically** during training: it is increased whenever the
attack success rate (ASR) clears a threshold and decreased when it drops, which steers
the optimizer to the smallest mask that still flips inputs. After convergence, the L1
norm of the mask (number of "trigger pixels") is recorded for class `t`.

**Stage 2 — Outlier detection on mask sizes.** With one L1 norm per class, the
distribution is examined for outliers using the **Median Absolute Deviation (MAD)**:

```
anomaly_index(c) = |L1(c) - median| / (1.4826 · MAD)
```

A class is flagged as a backdoor target if its anomaly index exceeds 2.0 *and* its mask
is smaller than the median (large masks are never suspicious — only abnormally small
ones suggest a shortcut). A model with any flagged class is reported as **TROJAN**.

---

## 3. Project layout

The repository is intentionally flat — six files, no packaging.

```
porba/
├── config.py            # central hyperparameter file
├── model.py             # CIFAR-10 ResNet-18 + checkpoint loader
├── data.py              # clean CIFAR-10 subset loader
├── neural_cleanse.py    # trigger reverse-engineering (the core algorithm)
├── anomaly.py           # MAD-based outlier detection on mask norms
├── detect.py            # CLI entry point — runs detection on a checkpoint
├── train_models.py      # CLI entry point — trains benign + trojan checkpoints
├── checkpoints/         # trained .pt model files (input to detect.py)
├── data/                # CIFAR-10 (auto-downloaded on first run)
└── results/             # auto-created: recovered triggers + summary.json
```

### File responsibilities

| File | Purpose |
|---|---|
| `config.py` | Single source of truth for all knobs (image shape, optimizer settings, MAD threshold, output directory, device). |
| `model.py` | Defines the CIFAR-style ResNet-18 (3×3 stem, no maxpool) and `load_model(path)`. The loader auto-unwraps common checkpoint conventions (`{"state_dict": ...}`, `{"netC": ...}`, `{"model_state_dict": ...}`, `{"model": ...}`). |
| `data.py` | Returns a small DataLoader over random clean CIFAR-10 *test* images in pixel space `[0, 1]` (no normalization). Test data is used to avoid any sample the model may have been poisoned on. |
| `neural_cleanse.py` | `reverse_engineer_trigger(model, target_class, loader)` runs the optimization for one class. `run_all_classes` is a thin loop over all 10 classes. |
| `anomaly.py` | `mad_anomaly_index(values)` returns per-class anomaly indices and an outlier mask using the MAD test. |
| `detect.py` | CLI: loads model, calls the two stages, prints a table, writes artifacts to `results/`. |
| `train_models.py` | CLI: trains benign and/or trojan ResNet-18 on CIFAR-10 with BadNets or WaNet, saves checkpoints in a format `detect.py` can load directly. |

---

## 4. Installation

The project depends on three packages:

```bash
pip install torch torchvision numpy
```

For a CPU-only environment (no CUDA):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

Tested with Python 3.10+.

---

## 5. Conventions

To keep the training pipeline and the detector consistent, the project uses one
shared convention for image inputs:

- **Pixel space `[0, 1]`, no per-channel normalization.** `transforms.ToTensor()`
  alone is used in both `train_models.py` and `data.py`. The model is trained,
  evaluated, and detected on `[0, 1]` inputs end-to-end. There is no
  `transforms.Normalize` step anywhere.

This matters for two reasons:

1. **Trigger semantics.** A BadNets trigger with `trigger_value=1.0` is meant to
   be a fully-bright pixel. That only makes sense in `[0, 1]` space — in
   normalized space the same value would be a strange off-distribution shade.
2. **Detection correctness.** Neural Cleanse reverse-engineers triggers that
   live in the model's actual input space. If the model was trained on `[0, 1]`
   inputs but the detector fed it normalized inputs, the model would see a
   distribution it was never trained on, harming both classification and trigger
   recovery.

If a checkpoint comes from elsewhere and *was* trained with normalization
(`transforms.Normalize` with CIFAR-10 mean/std), apply the same normalization to
the inputs of both `train_models.py` and the detector — keep them matched.

---

## 6. Training your own models

Use `train_models.py` to produce benign and trojan ResNet-18 checkpoints on
CIFAR-10. The script supports two attacks:

- **BadNets** — a fixed white square patch in the bottom-right corner of every
  poisoned training image. All poisoned images are relabeled to the target class.
- **WaNet** — a small smooth warping field applied to poisoned images,
  visually nearly identical to the original. All poisoned images are relabeled
  to the target class.

### Train a benign + BadNets-trojan pair (defaults)

```bash
python train_models.py --mode both --attack badnets --epochs 30
```

This produces, in `./checkpoints/`:

| File | Description |
|---|---|
| `benign_resnet18.pt` | Clean ResNet-18 trained on unmodified CIFAR-10. |
| `trojan_resnet18.pt` | ResNet-18 trained on 10%-poisoned CIFAR-10 with a 4×4 white trigger; target class 0. |

### Train a WaNet trojan

```bash
python train_models.py --mode trojan --attack wanet --epochs 30 \
    --target_label 0 --poison_rate 0.1
```

### Train only the benign baseline

```bash
python train_models.py --mode benign --epochs 30
```

### Important flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `both` | `benign`, `trojan`, or `both`. |
| `--attack` | `badnets` | `badnets` or `wanet`. |
| `--target_label` | `0` | Class the backdoor flips inputs into. |
| `--poison_rate` | `0.1` | Fraction of the training set to poison. |
| `--trigger_size` | `4` | Side length of the BadNets patch (pixels). |
| `--wanet_grid_rescale` | `0.5` | Warping intensity for WaNet. |
| `--epochs` | `30` | Training epochs (use 50+ for stronger models). |
| `--batch_size` | `128` | Training batch size. |
| `--lr` | `0.1` | Initial SGD learning rate; decayed at 50% and 75% of epochs. |
| `--save_dir` | `./checkpoints` | Where the `.pt` files are written. |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |

### Saved checkpoint format

Each `.pt` file is a Python dict with these keys:

```
{
    "model_state_dict": <state-dict matching model.ResNet18>,
    "epoch": <best-epoch index>,
    "best_clean_acc": <float, 0–100>,
    "history": {"train_loss", "train_acc", "clean_test_acc", "asr"},
    "metadata": {
        "model_name": "resnet18",
        "dataset": "cifar10",
        "mode": "benign" | "trojan",
        "attack": null | "badnets" | "wanet",
        "target_label": <int> | null,
        "poison_rate": <float>,
        ...
    },
}
```

`detect.py` automatically unwraps `model_state_dict`; the other fields are
informational and useful for thesis figures (e.g. plotting training curves
from `history`).

### Expected results

A 30-epoch run with the defaults on CIFAR-10 typically yields:

| Model | Clean test accuracy | Attack success rate |
|---|---|---|
| Benign | ~91–93% | ≈10% (chance) |
| BadNets trojan | ~90–92% | ~99% |
| WaNet trojan | ~89–91% | ~95–99% |

If clean accuracy stays under 80%, double-check that data augmentation
(`RandomCrop`, `RandomHorizontalFlip`) is enabled, that the learning-rate
schedule is firing, and that training did not stop early.

### After training

Run detection directly on the produced checkpoint:

```bash
python detect.py --model checkpoints/trojan_resnet18.pt --out-dir results/trojan
python detect.py --model checkpoints/benign_resnet18.pt --out-dir results/benign
```

The trojan run should print `VERDICT: TROJAN` with the suspected target class
matching `--target_label` used during training. The benign run should print
`VERDICT: BENIGN`.

---

## 7. Usage

### Inputs

- A ResNet-18 CIFAR-10 checkpoint (a `.pth`, `.pt`, or `.pth.tar` file containing a
  state-dict that matches the architecture defined in `model.py`).
- An internet connection on first run, so CIFAR-10 can be downloaded to `./data/`.
  Subsequent runs use the cached copy.

### Command

From inside the project directory:

```bash
python detect.py --model path/to/checkpoint.pth --out-dir results/run1
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--model` | *required* | Path to the checkpoint file. |
| `--out-dir` | `results` | Directory for recovered triggers and `summary.json`. Created if missing. |
| `--samples` | `500` (from `config.VAL_SAMPLES`) | Number of clean CIFAR-10 test images to use for the optimization. |

### Runtime

The detector runs `STEPS` (default 1000) optimization steps for each of 10 classes,
so it performs roughly 10,000 forward+backward passes through ResNet-18 in total.

Approximate wall-clock:

| Hardware | Wall-clock |
|---|---|
| NVIDIA GPU | ~1–3 minutes |
| Modern laptop CPU | ~30 minutes – 2 hours |

To speed up a CPU run at modest cost in detection quality, lower `STEPS` in
`config.py` to 300–400.

---

## 8. Output

### Console

A per-class table:

```
class | mask L1   | ASR    | anomaly | flagged
-------------------------------------------------------
    0 |      5.26 |  1.000 |    4.21 |   YES
    1 |     51.80 |  1.000 |    0.83 |
    2 |     41.57 |  1.000 |    0.65 |
    ...
```

followed by a verdict line:

```
VERDICT: TROJAN  (suspected target class: 0)
```

or

```
VERDICT: BENIGN
```

### Files

In the chosen `--out-dir`:

| File | Content |
|---|---|
| `mask_class_<i>.png` | The recovered mask for class `i` (single-channel image in pixel space). |
| `pattern_class_<i>.png` | The recovered pattern for class `i` (3-channel image). |
| `trigger_class_<i>.png` | `mask * pattern` — the recovered trigger as it would be added to a clean image. |
| `summary.json` | Machine-readable per-class results: target class, mask L1, ASR, anomaly index, flagged. |

The PNGs are useful for thesis figures — the trigger for the flagged class often
visibly resembles the planted backdoor (e.g. a small bright square in a corner for
BadNets).

---

## 9. How to read the numbers

- **`mask L1`** is the sum of all values in the recovered mask. The mask lives on a
  32×32 grid with values in `[0, 1]`, so L1 is bounded by 1024 (entire image
  replaced). Intuitively, mask L1 ≈ "how many pixels' worth of trigger are needed".
  A real planted trigger lets a single class get away with very few pixels.

- **`ASR`** (attack success rate) is the fraction of clean inputs that the recovered
  trigger flips into the target class. NC's optimization aims for ASR ≥ 0.99 for
  *every* class. ASR=1.0 across the board is normal — it is the *mask L1*, not the
  ASR, that exposes the backdoor.

- **`anomaly index`** is the MAD-based outlier score (see §2). A value above 2.0 on a
  class with a below-median mask norm flags that class.

- **`flagged`** is the binary outcome of the anomaly test per class.

### Expected qualitative patterns

| Model | mask L1 across classes | Verdict |
|---|---|---|
| Benign | All similar (e.g. all in 30–200) | `BENIGN` |
| BadNets-trojan | Target class has L1 ≪ others (often 5–30 vs 40–200+) | `TROJAN` |
| WaNet-trojan | Less reliable — see §12 | possibly `BENIGN` (false negative) |

---

## 10. Configuration reference

All tunables live in `config.py`.

### Dataset / model shape
- `NUM_CLASSES = 10`
- `IMAGE_SHAPE = (3, 32, 32)`

### Clean validation subset
- `VAL_SAMPLES = 500` — number of clean test images.
- `BATCH_SIZE = 64`
- `DATA_ROOT = "./data"` — where CIFAR-10 is stored/downloaded.

### Optimization (Neural Cleanse)
- `STEPS = 1000` — optimization steps per class.
- `LR = 0.1` — Adam learning rate (the NC reference value).
- `INIT_COST = 1e-3` — initial weight on the mask L1 penalty.
- `COST_MULTIPLIER = 2.0` — factor used when adjusting `cost` dynamically.
- `ATTACK_SUCCESS_THRESHOLD = 0.99` — ASR considered "trigger works".
- `PATIENCE = 5` — successive checks before adjusting `cost`.

### Outlier detection
- `MAD_THRESHOLD = 2.0` — anomaly-index threshold (NC paper value).
- `MAD_CONSISTENCY = 1.4826` — scale factor making MAD a consistent estimator of
  σ under normality.

### Misc
- `RESULTS_DIR = "results"`
- `DEVICE` — auto-detected (`cuda` if available, else `cpu`).

---

## 11. Algorithm details

This section unpacks `neural_cleanse.reverse_engineer_trigger`.

### Parameterization

The mask and pattern are stored as **unconstrained** real-valued tensors `raw_mask`
and `raw_pattern`, and squashed through sigmoid to obtain the actual mask and
pattern in `[0, 1]`. This avoids hard clipping and keeps optimization smooth.

**Mask initialization** is deliberately set so `sigmoid(raw_mask) ≈ 0` at step 0
(i.e., no trigger applied). Otherwise — if the mask starts at ~0.5 everywhere —
the attack succeeds trivially from the first step with a half-on mask, and the
optimization never finds a small trigger.

### Forward path

For a clean batch `x` of pixel-space images in `[0, 1]`:

```
mask    = sigmoid(raw_mask)            # (1, 32, 32)
pattern = sigmoid(raw_pattern)         # (3, 32, 32)
x_adv   = (1 - mask) * x + mask * pattern
logits  = model(x_adv)                 # model trained on [0, 1] inputs
```

There is no normalization step. The recovered trigger lives in interpretable
pixel space and can be saved directly as a viewable PNG (see §5 Conventions).

### Loss

```
loss = CrossEntropy(logits, target_class)  +  cost * mean(|mask|)
```

`mean` (rather than `sum`) is used so the penalty scale is independent of image size,
making `INIT_COST` portable across datasets.

### Dynamic cost balancing

This is the key NC trick. After each optimization step, the current ASR on the batch
is measured:

- If ASR ≥ 0.99 for `PATIENCE` consecutive steps, multiply `cost` by 2 — push harder
  to shrink the mask.
- If ASR < 0.99 for `PATIENCE` consecutive steps, divide `cost` by 2 — ease off so
  the attack can recover.

This loop settles at the boundary between "trigger barely works" and "mask too small
to work", which is exactly the smallest trigger that achieves the attack.

### Best-so-far tracking

Throughout optimization, the run records the `(mask, pattern)` pair with the smallest
`||mask||_1` such that ASR ≥ 0.99. The final returned trigger is this best-seen pair,
not the last-step state.

### Outlier detection

After all classes are processed, the list of mask L1 norms is fed to:

```
median  = median(norms)
mad     = median(|norms - median|)
indices = |norms - median| / (1.4826 * mad)
flagged = (indices > 2.0) & (norms < median)
```

A model with any flagged class is reported as `TROJAN`, and the class with the
smallest mask among the flagged is reported as the suspected backdoor target.

---

## 12. Limitations and known caveats

- **WaNet.** Neural Cleanse assumes the trigger can be written as a small *additive*
  mask + pattern overlay. WaNet's trigger is a global *warping field*, which cannot
  be represented this way. Empirically NC has reduced sensitivity to WaNet and may
  return `BENIGN` even on a WaNet-trojaned model. This is a documented limitation of
  NC, not a bug in this implementation.

- **Architecture rigidity.** `model.py` defines a specific CIFAR-style ResNet-18
  (3×3 stem wrapped in `nn.Sequential` as `stem`, BasicBlocks with `shortcut`
  modules, final `fc` layer). Checkpoints from training repos that use different
  naming conventions (`conv1`/`bn1`, `downsample`, `linear`) require either a small
  edit to `model.py` or a key-renaming step in `load_model`.

- **Compute cost.** Ten full optimization runs (one per class) is expensive on CPU.
  For larger label spaces (100, 1000 classes) NC scales linearly and becomes
  impractical without a GPU; recent work proposes faster reverse-engineering
  variants for that regime.

- **Adaptive attackers.** NC was published in 2019 and has known adaptive evasions
  (e.g. backdoors trained with feature-space regularization to avoid small triggers).
  It remains the standard *baseline*, not a final defense.

---

## 13. References

- B. Wang et al., *Neural Cleanse: Identifying and Mitigating Backdoor Attacks in
  Neural Networks*, IEEE S&P 2019.
- T. Gu et al., *BadNets: Evaluating Backdooring Attacks on Deep Neural Networks*,
  IEEE Access 2019.
- T. A. Nguyen and A. T. Tran, *WaNet — Imperceptible Warping-based Backdoor Attack*,
  ICLR 2021.
