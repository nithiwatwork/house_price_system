"""Benchmark whether the prepared data can support useful price predictions.

This module uses pandas and NumPy only. It keeps the benchmark portable while
enforcing the same train-only fitting rules a production pipeline must follow.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import engineer_house_features

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

TARGET_COLUMN = "price"


def sha256_file(path: Path) -> str:
    """Return a stable content hash for data-version evidence."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate interpretable metrics in the original price unit."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    errors = actual_values - predicted_values
    absolute_percentage_error = np.abs(errors / actual_values)
    total_variation = np.sum((actual_values - actual_values.mean()) ** 2)
    unexplained_variation = np.sum(errors**2)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(1 - unexplained_variation / total_variation),
        "mape_percent": float(absolute_percentage_error.mean() * 100),
        "within_10_percent": float((absolute_percentage_error <= 0.10).mean()),
        "within_20_percent": float((absolute_percentage_error <= 0.20).mean()),
    }


class TabularPreprocessor:
    """Learn imputation, scaling, and ZIP categories from training data only."""

    def fit(self, raw_features: pd.DataFrame) -> TabularPreprocessor:
        engineered = engineer_house_features(raw_features)
        self.numeric_columns = [
            column for column in engineered.columns if column != "zipcode"
        ]
        numeric = engineered[self.numeric_columns].apply(pd.to_numeric)
        # Division features become inf when bedrooms or sqft_lot is 0.
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        self.numeric_medians = numeric.median()
        imputed = numeric.fillna(self.numeric_medians)
        self.numeric_means = imputed.mean()
        self.numeric_stds = imputed.std(ddof=0).replace(0, 1)

        zipcode = engineered["zipcode"].astype("string")
        modes = zipcode.mode(dropna=True)
        self.zipcode_fill_value = str(modes.iloc[0]) if len(modes) else "missing"
        self.zipcode_categories = sorted(
            zipcode.fillna(self.zipcode_fill_value).astype(str).unique().tolist()
        )
        return self

    def transform(self, raw_features: pd.DataFrame) -> np.ndarray:
        engineered = engineer_house_features(raw_features)
        numeric = engineered[self.numeric_columns].apply(pd.to_numeric)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(self.numeric_medians)
        scaled_numeric = (numeric - self.numeric_means) / self.numeric_stds

        zipcode = (
            engineered["zipcode"]
            .astype("string")
            .fillna(self.zipcode_fill_value)
            .astype(str)
        )
        one_hot_columns = [
            (zipcode == category).astype(float).to_numpy()
            for category in self.zipcode_categories
        ]
        one_hot = np.column_stack(one_hot_columns)
        return np.column_stack([scaled_numeric.to_numpy(dtype=float), one_hot])

    def fit_transform(self, raw_features: pd.DataFrame) -> np.ndarray:
        return self.fit(raw_features).transform(raw_features)


class MedianBaseline:
    """Always predict the median training price."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> MedianBaseline:
        self.median = float(np.median(target))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.median)


class HousePriceRidgePipeline:
    """A small end-to-end ridge pipeline with a log-price target."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(
        self, raw_features: pd.DataFrame, target: pd.Series
    ) -> HousePriceRidgePipeline:
        self.preprocessor = TabularPreprocessor()
        matrix = self.preprocessor.fit_transform(raw_features)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        log_target = np.log1p(np.asarray(target, dtype=float))

        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0  # Do not regularize the intercept.
        self.coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ log_target,
        )
        return self

    def predict(self, raw_features: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(raw_features)
        design = np.column_stack([np.ones(len(matrix)), matrix])
        predictions = np.expm1(design @ self.coefficients)
        return np.maximum(predictions, 0)


def build_benchmark_models() -> dict[str, Any]:
    """Create a baseline and ridge candidates chosen using validation only."""

    return {
        "dummy_median": MedianBaseline(),
        "ridge_log_alpha_0_1": HousePriceRidgePipeline(alpha=0.1),
        "ridge_log_alpha_10": HousePriceRidgePipeline(alpha=10.0),
        "ridge_log_alpha_100": HousePriceRidgePipeline(alpha=100.0),
    }


def _split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=[TARGET_COLUMN]), data[TARGET_COLUMN]


def _verify_split_files(manifest: dict[str, Any], project_root: Path) -> None:
    """Stop benchmarking if generated files no longer match the manifest."""

    for split_name, split_details in manifest["splits"].items():
        split_path = project_root / split_details["path"]
        if sha256_file(split_path) != split_details["sha256"]:
            raise ValueError(
                f"{split_name} hash does not match split_manifest.json. "
                "Run Data Ingestion again."
            )


def _price_band_analysis(
    train_target: pd.Series,
    test_target: pd.Series,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    """Report test error using price bands learned from train only."""

    boundaries = train_target.quantile([0.25, 0.50, 0.75]).tolist()
    bands = pd.cut(
        test_target,
        bins=[-np.inf, *boundaries, np.inf],
        labels=["low", "mid_low", "mid_high", "high"],
        include_lowest=True,
    )
    analysis = pd.DataFrame(
        {"band": bands, "actual": test_target.to_numpy(), "predicted": predictions}
    )
    rows: list[dict[str, Any]] = []
    for band_name, band_data in analysis.groupby("band", observed=False):
        rows.append(
            {
                "price_band": str(band_name),
                "rows": len(band_data),
                "mae": float(
                    np.mean(np.abs(band_data["actual"] - band_data["predicted"]))
                ),
            }
        )
    return rows


def _raw_feature_importance(
    fitted_model: HousePriceRidgePipeline,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
) -> list[dict[str, float | str]]:
    """Measure validation MAE increase after shuffling each raw feature."""

    sample = validation_features.sample(
        n=min(1_000, len(validation_features)), random_state=42
    )
    sample_target = validation_target.loc[sample.index]
    base_mae = regression_metrics(sample_target, fitted_model.predict(sample))["mae"]
    random_generator = np.random.default_rng(42)
    rows: list[dict[str, float | str]] = []

    for column in sample.columns:
        increases = []
        for _ in range(3):
            shuffled = sample.copy()
            shuffled[column] = random_generator.permutation(
                shuffled[column].to_numpy()
            )
            shuffled_mae = regression_metrics(
                sample_target, fitted_model.predict(shuffled)
            )["mae"]
            increases.append(shuffled_mae - base_mae)
        rows.append(
            {
                "feature": column,
                "mae_increase_mean": float(np.mean(increases)),
                "mae_increase_std": float(np.std(increases)),
            }
        )
    rows.sort(key=lambda row: float(row["mae_increase_mean"]), reverse=True)
    return rows


def _write_prediction_plots(
    actual: pd.Series, predicted: np.ndarray, output_dir: Path
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(actual, predicted, alpha=0.25, s=12)
    lower = min(float(actual.min()), float(predicted.min()))
    upper = max(float(actual.max()), float(predicted.max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    axis.set(
        xlabel="Actual price",
        ylabel="Predicted price",
        title="Test: actual vs predicted",
    )
    figure.tight_layout()
    figure.savefig(figures_dir / "actual_vs_predicted.png", dpi=140)
    plt.close(figure)

    residuals = actual.to_numpy() - predicted
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.scatter(predicted, residuals, alpha=0.25, s=12)
    axis.axhline(0, linestyle="--", color="black")
    axis.set(
        xlabel="Predicted price",
        ylabel="Actual - predicted",
        title="Test residuals",
    )
    figure.tight_layout()
    figure.savefig(figures_dir / "residuals.png", dpi=140)
    plt.close(figure)


def _write_markdown_report(report: dict[str, Any], output_path: Path) -> None:
    rows = []
    for model_name, result in report["validation_results"].items():
        metrics = result["metrics"]
        rows.append(
            f"| {model_name} | {metrics['mae']:,.0f} | {metrics['rmse']:,.0f} | "
            f"{metrics['r2']:.3f} | {metrics['within_20_percent']:.1%} |"
        )
    test = report["test_result"]
    markdown = f"""# Data readiness benchmark

## Purpose

This benchmark checks whether the validated feature set contains useful signal.
It is evidence for handoff, not the model team's production model selection.

## Validation comparison

| Model | MAE | RMSE | R² | Within 20% |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Selected from validation MAE: **{report['selected_model']}**

## Final test (evaluated after selection)

- MAE: {test['selected_model_metrics']['mae']:,.0f}
- RMSE: {test['selected_model_metrics']['rmse']:,.0f}
- R²: {test['selected_model_metrics']['r2']:.3f}
- Within 20%: {test['selected_model_metrics']['within_20_percent']:.1%}
- MAE improvement over median baseline: {test['mae_improvement_vs_dummy_percent']:.1f}%
- Readiness gate passed: **{report['readiness_gate']['passed']}**

## Leakage controls

- Split hashes were checked against `split_manifest.json`.
- Property IDs do not cross train, validation, and test.
- `id` is removed before modeling and `zipcode` is encoded as categorical.
- Imputation, scaling, and ZIP categories are learned from train only.
- The chosen alpha is refitted on train + validation, then test is evaluated once.
"""
    output_path.write_text(markdown, encoding="utf-8")


def run_data_readiness_benchmark(
    project_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Run validation experiments and one final test evaluation."""

    manifest_path = project_root / "data/processed/split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_split_files(manifest, project_root)

    train = pd.read_csv(project_root / manifest["splits"]["train"]["path"])
    validation = pd.read_csv(
        project_root / manifest["splits"]["validation"]["path"]
    )
    test = pd.read_csv(project_root / manifest["splits"]["test"]["path"])
    train_features, train_target = _split_features_target(train)
    validation_features, validation_target = _split_features_target(validation)
    test_features, test_target = _split_features_target(test)

    models = build_benchmark_models()
    validation_results: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}
    for model_name, model in models.items():
        started_at = time.perf_counter()
        model.fit(train_features, train_target)
        predictions = model.predict(validation_features)
        validation_results[model_name] = {
            "metrics": regression_metrics(validation_target, predictions),
            "training_seconds": round(time.perf_counter() - started_at, 3),
        }
        fitted_models[model_name] = model

    learned_model_names = [name for name in models if name != "dummy_median"]
    selected_model_name = min(
        learned_model_names,
        key=lambda name: validation_results[name]["metrics"]["mae"],
    )
    importance = _raw_feature_importance(
        fitted_models[selected_model_name], validation_features, validation_target
    )

    # Refit the selected configuration on all pre-test data, then test once.
    combined_features = pd.concat(
        [train_features, validation_features], ignore_index=True
    )
    combined_target = pd.concat([train_target, validation_target], ignore_index=True)
    selected_alpha = models[selected_model_name].alpha
    selected_final_model = HousePriceRidgePipeline(alpha=selected_alpha)
    selected_final_model.fit(combined_features, combined_target)
    test_predictions = selected_final_model.predict(test_features)
    dummy_final = MedianBaseline().fit(combined_features, combined_target)
    dummy_test_predictions = dummy_final.predict(test_features)

    selected_test_metrics = regression_metrics(test_target, test_predictions)
    dummy_test_metrics = regression_metrics(test_target, dummy_test_predictions)
    test_improvement = (
        (dummy_test_metrics["mae"] - selected_test_metrics["mae"])
        / dummy_test_metrics["mae"]
        * 100
    )
    validation_dummy_mae = validation_results["dummy_median"]["metrics"]["mae"]
    validation_selected_mae = validation_results[selected_model_name]["metrics"]["mae"]
    validation_improvement = (
        (validation_dummy_mae - validation_selected_mae) / validation_dummy_mae * 100
    )

    report = {
        "dataset_manifest": str(manifest_path.relative_to(project_root)),
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "validation_results": validation_results,
        "selected_model": selected_model_name,
        "validation_mae_improvement_vs_dummy_percent": validation_improvement,
        "test_result": {
            "selected_model_metrics": selected_test_metrics,
            "dummy_metrics": dummy_test_metrics,
            "mae_improvement_vs_dummy_percent": test_improvement,
            "price_band_analysis": _price_band_analysis(
                train_target, test_target, test_predictions
            ),
        },
        "readiness_gate": {
            "rule": (
                "Selected model improves MAE over median by at least 20% "
                "on validation and test."
            ),
            "passed": validation_improvement >= 20 and test_improvement >= 20,
        },
        "raw_feature_permutation_importance": importance,
        "environment": {"numpy": np.__version__, "pandas": pd.__version__},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(importance).to_csv(
        output_dir / "raw_feature_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "id": test["id"],
            "date": test["date"],
            "actual_price": test_target,
            "predicted_price": test_predictions,
            "absolute_error": np.abs(test_target.to_numpy() - test_predictions),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    _write_prediction_plots(test_target, test_predictions, output_dir)
    _write_markdown_report(report, output_dir / "report.md")
    return report
