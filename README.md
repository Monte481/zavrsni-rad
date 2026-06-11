# Neural Cleanse detektor backdoor napada (ResNet-18)

Alat koji za dani **ResNet-18** klasifikator odlučuje je li model **trojan**
(npr. **BadNets** ili **WaNet** napadom) ili je **benigni**. Radi na **CIFAR-10** i
**MNIST** modelima i daje odluku `TROJAN` / `BENIGN`, dijagnostiku po klasama i
rekonstruirane slike okidača.

Detekcija se temelji na **Neural Cleanse** i ima dvije
varijante:

- **klasična** — koristi mali skup čistih slika (`detect.py`),
- **data-free** — ne treba nikakve prave podatke; sintetizira ulaze izravno iz
  modela pomoću **DeepInversion**-a i na njima pokreće
  isti algoritam (`detect_datafree.py`).

---

## Kako radi (ukratko)

Trojan model ima **prečac**: vrlo malu perturbaciju koja, dodana na bilo
koju sliku, model gura u napadačevu ciljnu klasu. Neural Cleanse za svaku klasu
rekonstruira najmanji takav okidač `(mask, pattern)` i usporedi veličine maski
(L1 norme). Ako jedna klasa ima neuobičajeno malu masku (MAD outlier test,
prag 2.0), model se proglašava `TROJAN`, a ta klasa je sumnjivi cilj.

Data-free varijanta radi isto, samo umjesto pravih slika koristi slike koje
DeepInversion "izvuče" iz samog modela (klasifikacija u ciljnu klasu + slaganje
s BatchNorm statistikama).

---

## Instalacija

```bash
pip install torch torchvision numpy
```

CPU-only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

Python 3.10+.

---

## Datoteke

| Datoteka | Svrha |
|---|---|
| `config.py` | Svi hiperparametri (oblik slike, parametri optimizatora, MAD prag, uređaj). |
| `model.py` | CIFAR-style ResNet-18 + `load_model(path)`. |
| `data.py` | Loader malog skupa čistih test slika (CIFAR-10 ili MNIST) u `[0,1]` prostoru. |
| `neural_cleanse.py` | Reverzni inženjering okidača — jezgra algoritma. |
| `anomaly.py` | MAD outlier test nad normama maski. |
| `deep_inversion.py` | Data-free sinteza slika iz modela (DeepInversion + BN matching). |
| `train_models.py` | Treniranje benign/trojan modela (BadNets/WaNet) na CIFAR-10 ili MNIST. |
| `detect.py` | Detekcija na čistim podacima. |
| `detect_datafree.py` | Detekcija na čistim ili sintetičkim (data-free) podacima. |

Izlazni direktoriji (`checkpoints/`, `results/`, `results_synthetic/`,
`data_synthetic/`, `data/`) kreiraju se automatski i nisu dio repozitorija.

---

## Korištenje

### 1. Treniraj model (po želji)

```bash
# benign + BadNets trojan na CIFAR-10
python train_models.py --mode both --attack badnets --epochs 30

# WaNet trojan
python train_models.py --mode trojan --attack wanet --epochs 30 --target_label 0

# MNIST umjesto CIFAR-10
python train_models.py --dataset mnist --mode both --attack badnets --epochs 30
```

Sprema `benign_resnet18.pt` / `trojan_resnet18.pt` u `--save_dir` (zadano
`./checkpoints`). Najvažnije zastavice: `--dataset {cifar10,mnist}`, `--mode`,
`--attack`, `--target_label`, `--poison_rate`, `--trigger_size`,
`--wanet_grid_rescale`, `--epochs`.

### 2. Detekcija na čistim podacima

```bash
python detect.py --model checkpoints/trojan_resnet18.pt
python detect.py --model checkpoints/trojan_resnet18.pt --dataset mnist
```

### 3. Data-free detekcija (bez pravih podataka)

```bash
python detect_datafree.py --model checkpoints/trojan_resnet18.pt --data-source synthetic
```

Prvo pokretanje sintetizira slike iz modela (spori dio) i cache-ira ih u
`data_synthetic/`; naredna pokretanja koriste cache. Za usporedni čisti baseline
koristi `--data-source clean`.

---

## Kako čitati rezultat

Detektor ispisuje tablicu po klasama i konačnu odluku:

```
class | mask L1 | ASR   | anomaly | flagged
   0  |   5.26  | 1.000 |   4.21  |  YES
   1  |  51.80  | 1.000 |   0.83  |
   ...
VERDICT: TROJAN  (suspected target class: 0)
```

- **mask L1** — veličina rekonstruiranog okidača; abnormalno mala kod backdoor klase.
- **anomaly** — MAD outlier score; > 2.0 uz malu masku označava klasu.
- **ASR** — `1.0` svugdje je normalno; ono što razlikuje backdoor je veličina maske, ne ASR.

Artefakti (rekonstruirane maske/okidači kao PNG + `summary.json`) spremaju se u
`results/` (čisto) odnosno `results_synthetic/` (data-free).

---

## Konvencija ulaza

Cijeli pipeline radi u pixel space-u `[0,1]` bez normalizacije — i treniranje
i detekcija. Tako rekonstruirani okidač živi u stvarnom ulaznom prostoru modela i
može se spremiti kao gledljiv PNG.

---