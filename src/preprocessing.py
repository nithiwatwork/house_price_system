"""Preprocessing pipeline shared by training and serving scripts."""

from src.data_readiness import TabularPreprocessor


def build_preprocessing_pipeline() -> TabularPreprocessor:
    """Build the preprocessor that learns imputation, scaling, and ZIP
    categories from training data only."""
    return TabularPreprocessor()
