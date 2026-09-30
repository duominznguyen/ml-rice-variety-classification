"""Fetch the Rice (Cammeo and Osmancik) dataset and split it into train/val/test."""
from typing import Tuple

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

RICE_DATASET_ID = 545


def fetch_rice_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_ucirepo(id=RICE_DATASET_ID)
    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]
    return X, y


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into (X_train, X_val, X_test, y_train, y_val, y_test) without sklearn."""
    rng = np.random.default_rng(random_state)
    n = len(X)

    if stratify:
        train_idx, val_idx, test_idx = _stratified_split_indices(y, rng, val_size, test_size)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(n * val_size)
        n_test = int(n * test_size)
        val_idx = idx[:n_val]
        test_idx = idx[n_val:n_val + n_test]
        train_idx = idx[n_val + n_test:]

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test


def _stratified_split_indices(
    y: pd.Series, rng: np.random.Generator, val_size: float, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = [], [], []
    y_arr = y.to_numpy()
    for label in np.unique(y_arr):
        label_idx = np.where(y_arr == label)[0]
        rng.shuffle(label_idx)
        n_label = len(label_idx)
        n_val = int(n_label * val_size)
        n_test = int(n_label * test_size)
        val_idx.append(label_idx[:n_val])
        test_idx.append(label_idx[n_val:n_val + n_test])
        train_idx.append(label_idx[n_val + n_test:])
    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )
