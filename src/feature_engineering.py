"""Shared feature engineering for training and future model serving."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
}


def engineer_house_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features without learning from validation or test.

    All operations are fixed formulas. No mean, median, category list, or other
    statistic is learned here, so this function is safe to reuse during serving.
    Learned preprocessing remains inside the training pipeline.
    """

    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Feature engineering is missing columns: {missing_text}")

    features = data.copy()
    sale_dates = pd.to_datetime(
        features["date"], format="%Y%m%dT%H%M%S", errors="raise"
    )

    features["sale_year"] = sale_dates.dt.year
    features["sale_month"] = sale_dates.dt.month
    features["sale_month_sin"] = np.sin(2 * np.pi * features["sale_month"] / 12)
    features["sale_month_cos"] = np.cos(2 * np.pi * features["sale_month"] / 12)

    features["house_age_at_sale"] = features["sale_year"] - features["yr_built"]
    features["has_renovation"] = features["yr_renovated"].gt(0).astype(int)
    features["years_since_renovation"] = np.where(
        features["yr_renovated"].gt(0),
        features["sale_year"] - features["yr_renovated"],
        0,
    )
    features["has_basement"] = features["sqft_basement"].gt(0).astype(int)
    features["sqft_per_bedroom"] = features["sqft_living"].div(
        features["bedrooms"]
    )
    features["bathrooms_per_bedroom"] = features["bathrooms"].div(
        features["bedrooms"]
    )
    features["living_to_lot_ratio"] = features["sqft_living"].div(
        features["sqft_lot"]
    )
    features["living_grade_interaction"] = (
        features["sqft_living"] * features["grade"]
    )
    features["house_age_squared"] = features["house_age_at_sale"].pow(2)

    # Log area features reduce the effect of very large, valid properties.
    for column in [
        "sqft_living",
        "sqft_lot",
        "sqft_above",
        "sqft_basement",
        "sqft_living15",
        "sqft_lot15",
    ]:
        features[f"log_{column}"] = np.log1p(features[column])

    # ID is kept in split files for lineage but must never become a model input.
    # Serving requests have no ID, so it is dropped only when present.
    # Raw date and renovation-year sentinel are replaced by explicit features.
    features = features.drop(columns=["id"], errors="ignore")
    features = features.drop(columns=["date", "yr_renovated"])

    # ZIP code represents a location category, not a measurable number.
    features["zipcode"] = features["zipcode"].astype("string")
    return features
