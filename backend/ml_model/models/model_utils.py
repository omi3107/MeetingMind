"""
Model Utilities — save / load helpers for trained artifacts.

Persists the TF-IDF vectorizer and the trained classifier via ``joblib``.
Saved artifacts live under ``backend/ml_model/models/saved/``.

Usage:
    from models.model_utils import save_artifacts, load_artifacts
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Default save directory (relative to the ml_model root)
# ---------------------------------------------------------------------------
_ML_MODEL_ROOT = Path(__file__).resolve().parent.parent   # backend/ml_model
DEFAULT_SAVE_DIR = _ML_MODEL_ROOT / "models" / "saved"

VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
CLASSIFIER_FILENAME = "svm_classifier.joblib"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"


def save_artifacts(
    vectorizer,
    classifier,
    label_encoder=None,
    save_dir: str | Path | None = None,
) -> Path:
    """Persist vectorizer, classifier, and optional label encoder.

    Args:
        vectorizer:    Fitted ``TfidfVectorizer``.
        classifier:    Fitted classifier.
        label_encoder: Optional fitted ``LabelEncoder``.
        save_dir:      Directory to save into (created if missing).

    Returns:
        ``Path`` to the save directory.
    """
    save_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, save_dir / VECTORIZER_FILENAME)
    joblib.dump(classifier, save_dir / CLASSIFIER_FILENAME)

    if label_encoder is not None:
        joblib.dump(label_encoder, save_dir / LABEL_ENCODER_FILENAME)

    print(f"[OK] Artifacts saved to {save_dir}")
    return save_dir


def load_artifacts(
    save_dir: str | Path | None = None,
) -> tuple:
    """Load vectorizer, classifier, and label encoder from disk.

    Args:
        save_dir: Directory containing saved ``.joblib`` files.

    Returns:
        ``(vectorizer, classifier, label_encoder)`` — label_encoder may be
        ``None`` if not saved.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    save_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR

    vec_path = save_dir / VECTORIZER_FILENAME
    clf_path = save_dir / CLASSIFIER_FILENAME
    le_path  = save_dir / LABEL_ENCODER_FILENAME

    if not vec_path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {vec_path}")
    if not clf_path.exists():
        raise FileNotFoundError(f"Classifier not found at {clf_path}")

    vectorizer = joblib.load(vec_path)
    classifier = joblib.load(clf_path)
    label_encoder = joblib.load(le_path) if le_path.exists() else None

    print(f"[OK] Artifacts loaded from {save_dir}")
    return vectorizer, classifier, label_encoder
