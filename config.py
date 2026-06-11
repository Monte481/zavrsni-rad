import torch

# Skup podataka / oblik modela
NUM_CLASSES = 10
IMAGE_SHAPE = (3, 32, 32)

# Sve radi u [0, 1] pixel space-u, bez normalizacije.

# Čisti podskup za reverzni inženjering okidača
VAL_SAMPLES = 500
BATCH_SIZE = 64
DATA_ROOT = "./data"

# Neural Cleanse optimizacija
STEPS = 1000                       # koraka po klasi
LR = 0.1                           # Adam learning rate
INIT_COST = 1e-3                   # početna težina L1 kazne na masku
COST_MULTIPLIER = 2.0              # faktor za balansiranje cost-a
ATTACK_SUCCESS_THRESHOLD = 0.99    # ASR od kojeg smatramo da okidač radi
PATIENCE = 5                       # koraka prije prilagodbe cost-a

# MAD detekcija anomalija
MAD_THRESHOLD = 2.0                # prag indeksa anomalije
MAD_CONSISTENCY = 1.4826           # skaliranje MAD-a pod normalnom razdiobom

RESULTS_DIR = "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
