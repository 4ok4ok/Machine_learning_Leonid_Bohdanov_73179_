"""Shared helpers: seeds, data split, model factory, paths."""
import random
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42

# The manual fixes LogisticRegression(max_iter=1000) on unscaled features; lbfgs hits the
# iteration cap. Config is kept as specified, the warning is silenced and discussed in report.md.
warnings.filterwarnings("ignore", message="lbfgs failed to converge")
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)


def make_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
