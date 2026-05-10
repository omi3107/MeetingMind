"""
TF-IDF Vectorizer wrapper for the ML classification pipeline.

Encapsulates ``sklearn.feature_extraction.text.TfidfVectorizer`` with
project-specific defaults (ngram range, max features, sublinear TF).

Usage:
    from models.tfidf_vectorizer import build_vectorizer, fit_transform, transform
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_MAX_FEATURES = 15_000
DEFAULT_NGRAM_RANGE = (1, 2)        # unigrams + bigrams
DEFAULT_SUBLINEAR_TF = True         # apply 1 + log(tf) scaling
DEFAULT_MIN_DF = 2                  # ignore very rare terms
DEFAULT_MAX_DF = 0.95               # ignore corpus-wide terms


def build_vectorizer(
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    sublinear_tf: bool = DEFAULT_SUBLINEAR_TF,
    min_df: int = DEFAULT_MIN_DF,
    max_df: float = DEFAULT_MAX_DF,
) -> TfidfVectorizer:
    """Create and return a configured ``TfidfVectorizer``.

    Args:
        max_features: Maximum number of features to keep.
        ngram_range:  The lower and upper boundary of the range of n-values.
        sublinear_tf: Apply sublinear TF scaling (1 + log(tf)).
        min_df:       Minimum document frequency for a term to be kept.
        max_df:       Maximum document frequency (as fraction of corpus).

    Returns:
        Unfitted ``TfidfVectorizer`` instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )


def fit_transform(vectorizer: TfidfVectorizer, texts):
    """Fit the vectorizer on *texts* and return the TF-IDF matrix.

    Args:
        vectorizer: An unfitted ``TfidfVectorizer``.
        texts:      Iterable of cleaned text strings (e.g. pandas Series).

    Returns:
        Sparse TF-IDF matrix (n_samples × n_features).
    """
    return vectorizer.fit_transform(texts)


def transform(vectorizer: TfidfVectorizer, texts):
    """Transform *texts* using an already-fitted vectorizer.

    Args:
        vectorizer: A fitted ``TfidfVectorizer``.
        texts:      Iterable of cleaned text strings.

    Returns:
        Sparse TF-IDF matrix.
    """
    return vectorizer.transform(texts)
