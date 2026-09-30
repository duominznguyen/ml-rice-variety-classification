"""Minimal cost-complexity (ccp_alpha) post-pruning for the hand-written DecisionTree.

Cost-complexity of a (sub)tree T is R_alpha(T) = R(T) + alpha * |T|, where R(T) is the
sum of each leaf's impurity weighted by its share of the training samples, and |T| is
the leaf count. For an internal node t, collapsing its subtree T_t into a single leaf
is as good as keeping it once alpha reaches the "effective alpha":

    alpha_eff(t) = (R(t) - R(T_t)) / (|T_t| - 1)

Weakest-link pruning repeatedly collapses the internal node with the smallest
alpha_eff, producing a sequence of nested trees T_0 (full tree) down to T_m (root
only) with non-decreasing alpha. The best alpha/tree is then picked by evaluating
every step of that sequence on the validation set.
"""
import copy
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from ml.models.tree import DecisionTree, Node


@dataclass
class PruningStep:
    alpha: float
    tree: DecisionTree
    n_leaves: int


def compute_alpha_path(tree: DecisionTree) -> List[PruningStep]:
    if tree.root is None:
        raise ValueError("Tree must be fitted before pruning.")

    n_total = tree.root.n_samples
    working_tree = copy.deepcopy(tree)
    path = [
        PruningStep(
            alpha=0.0,
            tree=copy.deepcopy(working_tree),
            n_leaves=_subtree_leaves(working_tree.root),
        )
    ]

    while not working_tree.root.is_leaf:
        alpha, weakest_node = _find_weakest_link(working_tree.root, n_total)
        _collapse_to_leaf(weakest_node)
        path.append(
            PruningStep(
                alpha=alpha,
                tree=copy.deepcopy(working_tree),
                n_leaves=_subtree_leaves(working_tree.root),
            )
        )

    return path


def prune_tree(tree: DecisionTree, alpha: float) -> DecisionTree:
    """Return the smallest pruned copy of `tree` whose path-alpha is <= `alpha`."""
    path = compute_alpha_path(tree)
    selected = path[0]
    for step in path:
        if step.alpha > alpha:
            break
        selected = step
    return selected.tree


def select_best_alpha(
    path: List[PruningStep],
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
) -> PruningStep:
    X_val = np.asarray(X_val, dtype=np.float64)
    y_val = np.asarray(y_val)

    best_step = path[0]
    best_score = -np.inf
    for step in path:
        score = scoring(y_val, step.tree.predict(X_val))
        if score > best_score:
            best_score = score
            best_step = step
    return best_step


def _find_weakest_link(root: Node, n_total: int):
    candidates = _internal_nodes(root)
    return min(((_effective_alpha(node, n_total), node) for node in candidates), key=lambda item: item[0])


def _internal_nodes(node: Node) -> List[Node]:
    if node.is_leaf:
        return []
    return [node, *_internal_nodes(node.left), *_internal_nodes(node.right)]


def _collapse_to_leaf(node: Node) -> None:
    node.left = None
    node.right = None
    node.feature_index = None
    node.threshold = None


def _subtree_leaves(node: Node) -> int:
    if node.is_leaf:
        return 1
    return _subtree_leaves(node.left) + _subtree_leaves(node.right)


def _subtree_impurity_sum(node: Node, n_total: int) -> float:
    """R(T_t): sum of weighted leaf impurities over the subtree rooted at `node`."""
    if node.is_leaf:
        return node.impurity * (node.n_samples / n_total)
    return _subtree_impurity_sum(node.left, n_total) + _subtree_impurity_sum(node.right, n_total)


def _effective_alpha(node: Node, n_total: int) -> float:
    r_node = node.impurity * (node.n_samples / n_total)
    r_subtree = _subtree_impurity_sum(node, n_total)
    n_leaves = _subtree_leaves(node)
    return (r_node - r_subtree) / (n_leaves - 1)
