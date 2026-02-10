"""
Machine learning pipeline for IBS residue prediction.

This module provides:
- parquet loading
- preprocessing
- protein-level train/val/test splitting
- class imbalance handling
- model training
- probability-based evaluation
- neighborhood probability smoothing
"""

from typing import Tuple, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef
)

# ======================================================
# Paths & constants
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ======================================================
# 0. Parquet loading
# ======================================================

def load_parquet(
    path: str | Path,
    columns: List[str] | None = None
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path, columns=columns)


def load_multiple_parquets(
    paths: Dict[str, str | Path],
    columns: List[str] | None = None
) -> Dict[str, pd.DataFrame]:
    return {
        name: load_parquet(path, columns)
        for name, path in paths.items()
    }


# ======================================================
# 1. Preprocessing
# ======================================================

def preprocess(
    df: pd.DataFrame,
    target_col: str = "IBS",
    drop_cols: List[str] | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    if drop_cols is None:
        drop_cols = []

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col] + drop_cols)

    return X, y


# ======================================================
# 2. Protein-level splitting
# ======================================================

def protein_level_split(
    df: pd.DataFrame,
    protein_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    proteins = df[protein_col].unique()

    train_prot, temp_prot = train_test_split(
        proteins,
        test_size=val_size + test_size,
        random_state=random_state,
        shuffle=True
    )

    val_frac = val_size / (val_size + test_size)

    val_prot, test_prot = train_test_split(
        temp_prot,
        test_size=1 - val_frac,
        random_state=random_state,
        shuffle=True
    )

    return train_prot, val_prot, test_prot


# ======================================================
# 3. Class imbalance handling
# ======================================================

def calculate_class_weights(
    y: pd.Series,
    pos_scale: float = 1.0
) -> Dict[int, float]:
    """
    Balanced inverse-frequency class weights with optional
    scaling of the positive (IBS) class.
    """

    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)

    class_weights = {
        cls: total / (n_classes * count)
        for cls, count in zip(classes, counts)
    }

    if 1 in class_weights:
        class_weights[1] *= pos_scale

    return class_weights


# ======================================================
# 4. Model training
# ======================================================

def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: Dict[int, float] | None = None
):
    if class_weights is not None and hasattr(model, "class_weight"):
        model.set_params(class_weight=class_weights)

    model.fit(X_train, y_train)
    return model


def train_and_score(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    pos_scale: float = 1.0
) -> float:
    class_weights = calculate_class_weights(y_train, pos_scale)
    model = train_model(model, X_train, y_train, class_weights)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_val_proba)


# ======================================================
# 5. Model evaluation
# ======================================================

def predict_with_threshold(
    model,
    X: pd.DataFrame,
    threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, y_proba


def smooth_probabilities(
    df: pd.DataFrame,
    prob_col: str,
    neighbors_col: str,
    alpha: float = 0.7
) -> np.ndarray:
    smoothed = np.zeros(len(df))

    for i, row in df.iterrows():
        neighbors = row[neighbors_col]

        if neighbors is None or len(neighbors) == 0:
            smoothed[i] = row[prob_col]
        else:
            neigh_mean = df.loc[neighbors, prob_col].mean()
            smoothed[i] = alpha * row[prob_col] + (1 - alpha) * neigh_mean

    return smoothed


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_recall: float = 0.6
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    precision = precision[:-1]
    recall = recall[:-1]

    valid = recall >= min_recall
    if not np.any(valid):
        return 0.5

    idx = np.argmax(precision[valid])
    return thresholds[valid][idx]


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df_eval: pd.DataFrame | None = None,
    neighbors_col: str | None = None,
    threshold: float = 0.5,
    smooth: bool = False,
    alpha: float = 0.7,
    split_name: str = "Test"
) -> Dict[str, float]:

    y_pred, y_proba = predict_with_threshold(model, X, threshold)

    if smooth:
        if df_eval is None or neighbors_col is None:
            raise ValueError("df_eval and neighbors_col required for smoothing")

        tmp = df_eval.copy()
        tmp["y_proba_raw"] = y_proba

        y_proba = smooth_probabilities(
            tmp,
            prob_col="y_proba_raw",
            neighbors_col=neighbors_col,
            alpha=alpha
        )

        y_pred = (y_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y, y_proba)
    mcc = matthews_corrcoef(y, y_pred)

    print(f"\n===== {split_name} evaluation =====")
    print(classification_report(y, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"MCC:    {mcc:.4f}")

    return {
        "pr_auc": pr_auc,
        "mcc": mcc,
        "threshold": threshold,
        "smoothing": smooth,
        "alpha": alpha
    }


# ======================================================
# 6. Model persistence
# ======================================================

def save_model(model, name: str) -> Path:
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
    return path


def load_model(name: str):
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
