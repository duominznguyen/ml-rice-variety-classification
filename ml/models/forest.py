"""Hand-written random forest: bagging + random feature subsets over DecisionTree, majority vote."""
from typing import List, Optional, Union

import numpy as np

from ml.models.tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Optional[Union[str, int]] = "sqrt",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees_: List[DecisionTree] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        max_features_per_split = self._n_features_for_split(n_features)

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y, rng) if self.bootstrap else (X, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features_per_split,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X_sample, y_sample, classes=self.classes_)
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        votes = np.stack([tree.predict(X) for tree in self.trees_], axis=0)  # (n_estimators, n_samples)
        return np.array([self._majority_vote(votes[:, i]) for i in range(X.shape[0])])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        per_tree_proba = np.stack([tree.predict_proba(X) for tree in self.trees_], axis=0)
        return per_tree_proba.mean(axis=0)

    def _majority_vote(self, votes: np.ndarray):
        values, counts = np.unique(votes, return_counts=True)
        return values[np.argmax(counts)]

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        n_samples = X.shape[0]
        indices = rng.integers(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    def _n_features_for_split(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError(f"Unknown max_features: {self.max_features}")
        return max(1, min(int(self.max_features), n_features))
