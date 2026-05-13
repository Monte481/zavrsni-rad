"""Single source of hyperparameters for the Neural Cleanse detector.

Tweak values here; the rest of the code reads from this module.
"""

import torch

# Dataset / model shape
NUM_CLASSES = 10
IMAGE_SHAPE = (3, 32, 32)

# NOTE: this pipeline operates on raw [0, 1] pixel inputs end-to-end — no
# normalization. Both train_models.py and the detector skip transforms.Normalize.
# If you swap in a checkpoint that was trained with normalization, undo this
# convention everywhere (see README §"Conventions").

# Clean validation subset used for trigger reverse-engineering
VAL_SAMPLES = 500
BATCH_SIZE = 64
DATA_ROOT = "./data"

# Neural Cleanse optimization
STEPS = 1000                       # gradient steps per target class
LR = 0.1                           # Adam learning rate (NC reference)
INIT_COST = 1e-3                   # initial lambda on mask L1 penalty
COST_MULTIPLIER = 2.0              # factor for dynamic cost balancing
ATTACK_SUCCESS_THRESHOLD = 0.99    # ASR considered "trigger works"
PATIENCE = 5                       # checks before adjusting cost

# MAD-based anomaly detection
MAD_THRESHOLD = 2.0                # anomaly index threshold from the NC paper
MAD_CONSISTENCY = 1.4826           # scale factor for normal-distribution MAD

# Output
RESULTS_DIR = "results"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
