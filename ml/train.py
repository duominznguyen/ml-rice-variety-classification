"""Train the 3 required models (pre-pruned tree, post-pruned tree, random forest) and save them.

Run with: py -m ml.train
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ml.data.load_data import fetch_rice_dataset, train_val_test_split
from ml.models.forest import RandomForest
from ml.models.pruning import compute_alpha_path, select_best_alpha
from ml.models.tree import DecisionTree
from ml.utils.metrics import accuracy_score, classification_report

SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
RANDOM_STATE = 42

PRE_PRUNING_MAX_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, None]
FOREST_MAX_DEPTHS = [5, 10, 15, 20, None]
FOREST_N_ESTIMATORS = 100


def save_model(name: str, model, hyperparameters: dict, metrics: dict) -> None:
    SAVED_MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, SAVED_MODELS_DIR / f"{name}.joblib")
    metadata = {
        "name": name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparameters,
        "metrics": metrics,
    }
    with open(SAVED_MODELS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _final_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    return {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "val_accuracy": accuracy_score(y_val, model.predict(X_val)),
        "test": classification_report(y_test, model.predict(X_test)),
    }


def train_pre_pruned_tree(X_train, y_train, X_val, y_val, X_test, y_test) -> DecisionTree:
    sweep = []
    best_model, best_depth, best_val_acc = None, None, -1.0
    for depth in PRE_PRUNING_MAX_DEPTHS:
        tree = DecisionTree(max_depth=depth, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, tree.predict(X_val))
        sweep.append({"max_depth": depth, "val_accuracy": val_acc})
        if val_acc > best_val_acc:
            best_model, best_depth, best_val_acc = tree, depth, val_acc

    metrics = _final_metrics(best_model, X_train, y_train, X_val, y_val, X_test, y_test)
    metrics["max_depth_sweep"] = sweep
    hyperparameters = {"max_depth": best_depth}
    save_model("pre_pruned_tree", best_model, hyperparameters, metrics)
    print(
        f"[pre-pruned tree] best max_depth={best_depth} "
        f"val_acc={best_val_acc:.4f} test_acc={metrics['test']['accuracy']:.4f}"
    )
    return best_model


def train_post_pruned_tree(X_train, y_train, X_val, y_val, X_test, y_test) -> DecisionTree:
    full_tree = DecisionTree(random_state=RANDOM_STATE)
    full_tree.fit(X_train, y_train)

    path = compute_alpha_path(full_tree)
    best_step = select_best_alpha(path, X_val, y_val, accuracy_score)

    metrics = _final_metrics(best_step.tree, X_train, y_train, X_val, y_val, X_test, y_test)
    metrics["alpha_path"] = [
        {
            "alpha": step.alpha,
            "n_leaves": step.n_leaves,
            "val_accuracy": accuracy_score(y_val, step.tree.predict(X_val)),
        }
        for step in path
    ]
    hyperparameters = {"ccp_alpha": best_step.alpha, "n_leaves": best_step.n_leaves}
    save_model("post_pruned_tree", best_step.tree, hyperparameters, metrics)
    print(
        f"[post-pruned tree] best alpha={best_step.alpha:.6f} n_leaves={best_step.n_leaves} "
        f"val_acc={metrics['val_accuracy']:.4f} test_acc={metrics['test']['accuracy']:.4f}"
    )
    return best_step.tree


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test) -> RandomForest:
    sweep = []
    best_model, best_depth, best_val_acc = None, None, -1.0
    for depth in FOREST_MAX_DEPTHS:
        forest = RandomForest(n_estimators=FOREST_N_ESTIMATORS, max_depth=depth, random_state=RANDOM_STATE)
        forest.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, forest.predict(X_val))
        sweep.append({"max_depth": depth, "n_estimators": FOREST_N_ESTIMATORS, "val_accuracy": val_acc})
        if val_acc > best_val_acc:
            best_model, best_depth, best_val_acc = forest, depth, val_acc

    metrics = _final_metrics(best_model, X_train, y_train, X_val, y_val, X_test, y_test)
    metrics["max_depth_sweep"] = sweep
    hyperparameters = {
        "n_estimators": FOREST_N_ESTIMATORS,
        "max_depth": best_depth,
        "max_features": "sqrt",
        "bootstrap": True,
    }
    save_model("random_forest", best_model, hyperparameters, metrics)
    print(
        f"[random forest] best max_depth={best_depth} "
        f"val_acc={best_val_acc:.4f} test_acc={metrics['test']['accuracy']:.4f}"
    )
    return best_model


def main() -> None:
    X, y = fetch_rice_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.15, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_val, X_test = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy()
    y_train, y_val, y_test = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()

    train_pre_pruned_tree(X_train, y_train, X_val, y_val, X_test, y_test)
    train_post_pruned_tree(X_train, y_train, X_val, y_val, X_test, y_test)
    train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    main()
