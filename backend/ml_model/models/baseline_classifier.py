"""
Baseline Classifier — Linear SVM for sentence-level transcript classification.

Wraps ``sklearn.svm.LinearSVC`` with sensible defaults for the 5-class
meeting-sentence classification task (Decision, Task, Deadline, Issue, General).

Usage:
    from models.baseline_classifier import build_classifier, train, predict
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_C = 1.0
DEFAULT_MAX_ITER = 10_000


def build_classifier(
    C: float = DEFAULT_C,
    max_iter: int = DEFAULT_MAX_ITER,
    class_weight: str | dict | None = "balanced",
    calibrated: bool = True,
) -> LinearSVC | CalibratedClassifierCV:
    """Create a Linear SVM classifier.

    Args:
        C:            Regularisation parameter.
        max_iter:     Maximum iterations for the solver.
        class_weight: ``'balanced'`` to handle class imbalance automatically.
        calibrated:   Wrap with ``CalibratedClassifierCV`` so ``predict_proba``
                      is available (useful for downstream confidence filtering).

    Returns:
        Classifier instance (unfitted).
    """
    base = LinearSVC(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        dual="auto",
        random_state=42,
    )
    if calibrated:
        return CalibratedClassifierCV(base, cv=3)
    return base


def train(clf, X_train, y_train):
    """Fit the classifier.

    Args:
        clf:     Classifier from ``build_classifier``.
        X_train: TF-IDF feature matrix (sparse or dense).
        y_train: Target labels.

    Returns:
        Fitted classifier.
    """
    clf.fit(X_train, y_train)
    return clf


def predict(clf, X) -> np.ndarray:
    """Return predicted class labels.

    Args:
        clf: Fitted classifier.
        X:   Feature matrix.

    Returns:
        1-D array of predicted labels.
    """
    return clf.predict(X)


def predict_proba(clf, X) -> np.ndarray | None:
    """Return class probabilities if available.

    Args:
        clf: Fitted classifier (must be calibrated).
        X:   Feature matrix.

    Returns:
        2-D array of shape (n_samples, n_classes) or ``None``.
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return None
