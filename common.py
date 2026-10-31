"""Small helpers every script shares: device, seeding, MLflow setup, results.json."""

import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch runs a missing op
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import torch

RESULTS = Path("results.json")
MODELS = Path("models")


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_all(seed: int) -> None:
    """Seeds Python, NumPy and torch. MPS is not bit-for-bit deterministic, so repeats differ slightly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_mlflow(experiment: str) -> None:
    # one SQLite file instead of a folder of small files; `mlflow ui --backend-store-uri sqlite:///mlflow.db`
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment)


def update_results(key: str, value) -> None:
    """Each stage owns one key, so re-running one stage does not throw away the others. None removes the key."""
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1))
