import os
os.environ["OMP_NUM_THREADS"] = "1"  # avoid thread oversubscription (see Lab 1)

import random
from pathlib import Path

import numpy as np
import torch
import yaml

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
DATA = REPO / "data"
RESULTS = LAB / "results"
RESULTS.mkdir(exist_ok=True)
CONFIG = yaml.safe_load((LAB / "config.yaml").read_text())


def set_seeds(seed=CONFIG["seed"]):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
