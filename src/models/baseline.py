"""
Logistic Regression Baseline Classifier.
Provides a standard, non-temporal static intrusion detection baseline for fair benchmarking.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib


class LogisticRegressionBaseline:
    """Static non-temporal classifier baseline."""

    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits baseline model on state features.
        If X is 3D (N, seq_len, num_features), it uses the most recent state X[:, -1, :].
        """
        if X.ndim == 3:
            X_static = X[:, -1, :]
        else:
            X_static = X

        X_scaled = self.scaler.fit_transform(X_static)
        # Ensure binary target
        y_binary = (y > 0.5).astype(int)
        unique_classes = np.unique(y_binary)
        if len(unique_classes) < 2:
            # Add synthetic opposite sample at extreme boundary to allow solver convergence
            opposite = 1 - unique_classes[0]
            synthetic_x = np.zeros((1, X_scaled.shape[1]))
            X_scaled = np.vstack([X_scaled, synthetic_x])
            y_binary = np.append(y_binary, opposite)

        self.model.fit(X_scaled, y_binary)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns attack probability [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Baseline model is not fitted.")
        if X.ndim == 3:
            X_static = X[:, -1, :]
        else:
            X_static = X

        X_scaled = self.scaler.transform(X_static)
        # Probabilities for class 1
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluates baseline and returns precision, recall, f1, roc_auc, fpr."""
        y_binary = (y > 0.5).astype(int)
        probas = self.predict_proba(X)
        preds = (probas >= 0.5).astype(int)

        precision = float(precision_score(y_binary, preds, zero_division=0))
        recall = float(recall_score(y_binary, preds, zero_division=0))
        f1 = float(f1_score(y_binary, preds, zero_division=0))

        try:
            auc = float(roc_auc_score(y_binary, probas))
        except ValueError:
            auc = 0.5

        tn, fp, fn, tp = confusion_matrix(y_binary, preds, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        return {
            "model": "Logistic Regression Baseline",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "fpr": fpr,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }

    def save(self, filepath: str) -> None:
        joblib.dump({"model": self.model, "scaler": self.scaler}, filepath)

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = True
