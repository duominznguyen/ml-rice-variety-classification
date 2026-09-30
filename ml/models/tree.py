"""Hand-written CART decision tree (Gini impurity, binary threshold splits on numeric features)."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Node:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[int] = None
    n_samples: int = 0
    n_samples_per_class: Optional[np.ndarray] = None
    impurity: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None

    def fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None) -> "DecisionTree":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # `classes` lets an ensemble (e.g. RandomForest) fix a common class set,
        # since a bootstrap sample can by chance omit a rare class entirely.
        self.classes_ = np.unique(y) if classes is None else np.asarray(classes)
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_proba_one(x, self.root) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples = X.shape[0]
        n_samples_per_class = self._class_counts(y)
        node = Node(
            value=self._leaf_value(y),
            n_samples=n_samples,
            n_samples_per_class=n_samples_per_class,
            impurity=self._impurity(y),
        )

        can_split = (
            node.impurity > 0.0
            and n_samples >= self.min_samples_split
            and (self.max_depth is None or depth < self.max_depth)
        )
        if not can_split:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, _gain = split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, float, float]]:
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        if parent_impurity == 0.0 or n_samples < 2 * self.min_samples_leaf:
            return None

        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for feature_index in self._candidate_features(n_features):
            order = np.argsort(X[:, feature_index], kind="mergesort")
            sorted_values = X[order, feature_index]
            sorted_y = y[order]

            # Running class counts as the split point sweeps left-to-right, so each
            # threshold's impurity is derived from counts in O(1) instead of O(n).
            class_hits = (sorted_y[:, None] == self.classes_[None, :]).astype(np.float64)
            cum_counts = np.cumsum(class_hits, axis=0)
            total_counts = cum_counts[-1]

            lo = self.min_samples_leaf
            hi = n_samples - self.min_samples_leaf
            for i in range(lo, hi + 1):
                if sorted_values[i] == sorted_values[i - 1]:
                    continue  # no valid threshold between equal feature values
                left_counts = cum_counts[i - 1]
                right_counts = total_counts - left_counts
                left_impurity = self._impurity_from_counts(left_counts, i)
                right_impurity = self._impurity_from_counts(right_counts, n_samples - i)
                weighted_impurity = (i / n_samples) * left_impurity + (
                    (n_samples - i) / n_samples
                ) * right_impurity
                gain = parent_impurity - weighted_impurity
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = (sorted_values[i - 1] + sorted_values[i]) / 2.0

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain

    def _candidate_features(self, n_features: int) -> np.ndarray:
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=self.max_features, replace=False)

    def _impurity(self, y: np.ndarray) -> float:
        return self._impurity_from_counts(self._class_counts(y), len(y))

    def _impurity_from_counts(self, counts: np.ndarray, n: int) -> float:
        if n == 0:
            return 0.0
        probs = counts / n
        return 1.0 - np.sum(probs ** 2)

    def _class_counts(self, y: np.ndarray) -> np.ndarray:
        return np.array([np.sum(y == c) for c in self.classes_], dtype=np.float64)

    def _leaf_value(self, y: np.ndarray):
        counts = self._class_counts(y)
        return self.classes_[np.argmax(counts)]

    def _predict_one(self, x: np.ndarray, node: Node):
        while not node.is_leaf:
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        return node.value

    def _predict_proba_one(self, x: np.ndarray, node: Node) -> np.ndarray:
        while not node.is_leaf:
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        total = node.n_samples_per_class.sum()
        if total == 0:
            return np.zeros_like(node.n_samples_per_class)
        return node.n_samples_per_class / total
