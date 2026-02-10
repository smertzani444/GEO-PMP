"""
Machine learning pipeline for IBS residue prediction.

This module provides:
- preprocessing
- class imbalance handling
- protein-level train/val/test splitting
- model training
- model evaluation

"""

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import optuna
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent # BASE_DIR = thesis_code/data_analysis
MODEL_DIR = BASE_DIR / "models"            # MODEL_DIR = thesis_code/data_analysis/models
MODEL_DIR.mkdir(exist_ok=True)


RANDOM_STATE = 42

# ======================================================
# 0. Parquet loading
# ======================================================

def load_parquet(
    path: str | Path,
    columns: List[str] | None = None
) -> pd.DataFrame:
    """
    Load a single parquet file.

    Parameters
    ----------
    path : str or Path
        Path to parquet file.
    columns : list, optional
        Columns to load (None loads all).

    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path, columns=columns)
    return df

def load_multiple_parquets(
    paths: Dict[str, str | Path],
    columns: List[str] | None = None
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple parquet files into a dictionary.

    Parameters
    ----------
    paths : dict
        Dictionary mapping dataset name -> parquet path.
    columns : list, optional
        Columns to load.

    Returns
    -------
    data : dict
        Dictionary mapping dataset name -> DataFrame.
    """

    data = {}
    for name, path in paths.items():
        data[name] = load_parquet(path, columns=columns)

    return data


# ======================================================
# 1. Preprocessing
# ======================================================

def preprocess(
    df: pd.DataFrame,
    target_col: str = "IBS",
    drop_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Structural preprocessing: split features and target,
    drop identifier columns.

    No encoding or scaling is applied here.
    """

    if drop_cols is None:
        drop_cols = []

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col] + drop_cols)

    return X, y



# ======================================================
# 2. Protein-level train / val / test split
# ======================================================

def protein_level_split(
    df: pd.DataFrame,
    protein_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform protein-level train/val/test split.

    Returns
    -------
    train_prot, val_prot, test_prot
        Arrays of protein identifiers.
    """

    proteins = df[protein_col].unique()

    train_prot, temp_prot = train_test_split(
        proteins,
        test_size=val_size + test_size,
        random_state=random_state,
        shuffle=True
    )

    val_fraction = val_size / (val_size + test_size)

    val_prot, test_prot = train_test_split(
        temp_prot,
        test_size=1 - val_fraction,
        random_state=random_state,
        shuffle=True
    )

    return train_prot, val_prot, test_prot

def prepare_ml_inputs(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str]
):
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


# ======================================================
# 3. Class imbalance handling
# ======================================================

def calculate_class_weights(
    y: pd.Series,
    pos_scale: float = 1.0
) -> Dict[int, float]:
    """
    Compute balanced class weights with optional scaling
    of the positive (IBS) class.

    pos_scale > 1.0 increases recall
    pos_scale < 1.0 increases precision
    """

    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    n_classes = len(classes)

    class_weights = {
        cls: total_samples / (n_classes * count)
        for cls, count in zip(classes, counts)
    }

    # Scale positive class (IBS = 1)
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
    class_weights: Dict[int, float] = None
):
    """
    Train a model with optional class weights.

    Parameters
    ----------
    model : sklearn estimator
        Model instance.
    X_train : pd.DataFrame
    y_train : pd.Series
    class_weights : dict, optional

    Returns
    -------
    model
        Trained model.
    """

    if class_weights is not None:
        if hasattr(model, "class_weight"):
            model.set_params(class_weight=class_weights)

    model.fit(X_train, y_train)
    return model

from sklearn.metrics import average_precision_score

def train_and_score(
    model,
    X_train,
    y_train,
    X_val,
    y_val
):
    class_weights = calculate_class_weights(y_train)

    model = train_model(
        model,
        X_train,
        y_train,
        class_weights=class_weights
    )

    y_val_proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_val_proba)



# ======================================================
# 5. Model evaluation
# ======================================================

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "Test"
) -> Dict[str, float]:
    """
    Evaluate a trained model.

    Parameters
    ----------
    model : sklearn estimator
    X : pd.DataFrame
    y : pd.Series
    split_name : str

    Returns
    -------
    metrics : dict
        Evaluation metrics.
    """

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = y_pred

    ap = average_precision_score(y, y_score)
    mcc = matthews_corrcoef(y, y_pred)

    print(f"\n===== {split_name} evaluation =====")
    print(classification_report(y, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    metrics = {
        "average_precision": ap,
        "mcc": mcc
    }

    return metrics

# ======================================================
# 6. Save Models
# ======================================================

def save_model(model, name: str):
    """
    Save a trained sklearn Pipeline to thesis_code/data_analysis/models
    """
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
    return path


def load_model(name: str):
    """
    Load a trained sklearn Pipeline from thesis_code/data_analysis/models
    """
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
