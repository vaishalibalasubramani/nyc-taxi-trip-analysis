"""
Yellow Taxi -- Trip Duration Prediction

Time-based 80/20 train/test split.

Usage:
    python src/train_duration_yellow.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "yellow_trip_features.parquet"
)

MODEL_FILE = (
    ROOT
    / "outputs"
    / "yellow_duration_model.joblib"
)

METRICS_FILE = (
    ROOT
    / "outputs"
    / "yellow_duration_model_metrics.txt"
)

PREDICTIONS_FILE = (
    ROOT
    / "outputs"
    / "yellow_duration_predictions.parquet"
)


# =====================================================================
# SETTINGS
# =====================================================================

RANDOM_STATE = 42
SAMPLE_SIZE = 500_000
TRAIN_FRACTION = 0.80


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("YELLOW TAXI TRIP DURATION TRAINING")
    print("=" * 70)

    # -----------------------------------------------------------------
    # LOAD
    # -----------------------------------------------------------------

    print("\nLoading data from:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    print(
        f"\nLoaded {len(df):,} Yellow Taxi rows"
    )

    # -----------------------------------------------------------------
    # REQUIRED COLUMNS
    # -----------------------------------------------------------------

    required_columns = [
        "pickup_datetime",
        "trip_duration_s",
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "RatecodeID",
        "payment_type",
        "pickup_hour",
        "pickup_dow",
        "is_weekend",
        "pickup_month",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # -----------------------------------------------------------------
    # DATETIME
    # -----------------------------------------------------------------

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        errors="coerce",
    )

    # -----------------------------------------------------------------
    # DROP NaNs WITHOUT COPYING THE FULL DATASET
    # -----------------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    )

    print(
        f"Rows after dropping NaNs: {len(df):,}"
    )

    # -----------------------------------------------------------------
    # DATA TYPES
    # -----------------------------------------------------------------

    df["trip_duration_s"] = pd.to_numeric(
        df["trip_duration_s"],
        errors="coerce",
    )

    df["trip_distance"] = pd.to_numeric(
        df["trip_distance"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "trip_duration_s",
            "trip_distance",
        ]
    )

    # -----------------------------------------------------------------
    # TIME RANGE
    # -----------------------------------------------------------------

    print("\nData time range:")

    print(
        f"First trip: {df['pickup_datetime'].min()}"
    )

    print(
        f"Last trip:  {df['pickup_datetime'].max()}"
    )

    # -----------------------------------------------------------------
    # SAMPLE
    # -----------------------------------------------------------------

    print(
        f"\nSampling {SAMPLE_SIZE:,} rows "
        f"from {len(df):,} available rows..."
    )

    df = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    ).copy()

    print(
        f"Rows used for modeling: {len(df):,}"
    )

    # -----------------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # -----------------------------------------------------------------

    df = df.sort_values(
        "pickup_datetime"
    ).reset_index(
        drop=True
    )

    # =================================================================
    # TIME-BASED SPLIT
    # =================================================================

    print("\n" + "-" * 70)
    print("TIME-BASED TRAIN / TEST SPLIT")
    print("-" * 70)

    split_index = int(
        len(df) * TRAIN_FRACTION
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    print(
        f"\nTraining rows: {len(train_df):,}"
    )

    print(
        f"Test rows:     {len(test_df):,}"
    )

    print("\nTraining period:")

    print(
        f"  {train_df['pickup_datetime'].min()}"
    )

    print("  to")

    print(
        f"  {train_df['pickup_datetime'].max()}"
    )

    print("\nTest period:")

    print(
        f"  {test_df['pickup_datetime'].min()}"
    )

    print("  to")

    print(
        f"  {test_df['pickup_datetime'].max()}"
    )

    # =================================================================
    # FEATURES
    # =================================================================

    features = [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "RatecodeID",
        "payment_type",
        "pickup_hour",
        "pickup_dow",
        "is_weekend",
        "pickup_month",
    ]

    target = "trip_duration_s"

    X_train = train_df[
        features
    ]

    y_train = train_df[
        target
    ]

    X_test = test_df[
        features
    ]

    y_test = test_df[
        target
    ]

    # =================================================================
    # TRAIN
    # =================================================================

    print("\n" + "-" * 70)
    print("TRAINING MODEL")
    print("-" * 70)

    print(
        "\nTraining Yellow Taxi "
        "GradientBoostingRegressor..."
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    # =================================================================
    # PREDICT
    # =================================================================

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    # =================================================================
    # METRICS
    # =================================================================

    actual_values = y_test.to_numpy()

    mae = mean_absolute_error(
        actual_values,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predictions,
        )
    )

    r2 = r2_score(
        actual_values,
        predictions,
    )

    abs_error = np.abs(
        predictions
        - actual_values
    )

    within_2 = (
        abs_error <= 120
    ).mean() * 100

    within_5 = (
        abs_error <= 300
    ).mean() * 100

    within_10 = (
        abs_error <= 600
    ).mean() * 100

    actual_mean = (
        actual_values.mean() / 60
    )

    predicted_mean = (
        predictions.mean() / 60
    )

    actual_median = (
        np.median(actual_values) / 60
    )

    predicted_median = (
        np.median(predictions) / 60
    )

    # =================================================================
    # PERFORMANCE
    # =================================================================

    print("\n" + "=" * 70)
    print("YELLOW TAXI TIME-BASED TEST PERFORMANCE")
    print("=" * 70)

    print(
        f"\nMAE:  {mae:.1f} seconds "
        f"({mae / 60:.2f} minutes)"
    )

    print(
        f"RMSE: {rmse:.1f} seconds "
        f"({rmse / 60:.2f} minutes)"
    )

    print(
        f"R^2:  {r2:.3f}"
    )

    print("\nPrediction accuracy:")

    print(
        f"Within ±2 minutes:  "
        f"{within_2:.1f}%"
    )

    print(
        f"Within ±5 minutes:  "
        f"{within_5:.1f}%"
    )

    print(
        f"Within ±10 minutes: "
        f"{within_10:.1f}%"
    )

    print("\nAverage duration:")

    print(
        f"Actual mean:     "
        f"{actual_mean:.2f} minutes"
    )

    print(
        f"Predicted mean:  "
        f"{predicted_mean:.2f} minutes"
    )

    print(
        f"Actual median:   "
        f"{actual_median:.2f} minutes"
    )

    print(
        f"Predicted median:"
        f" {predicted_median:.2f} minutes"
    )

    # =================================================================
    # FEATURE IMPORTANCE
    # =================================================================

    print("\nFeature importances:")

    importances = pd.Series(
        model.feature_importances_,
        index=features,
    ).sort_values(
        ascending=False
    )

    print(importances)

    # =================================================================
    # SAVE MODEL
    # =================================================================

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the raw model because the existing dashboard loads this file
    # directly and calls model.predict(X).
    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"\nSaved model -> "
        f"{MODEL_FILE}"
    )

    # =================================================================
    # SAVE METRICS
    # =================================================================

    metrics_text = f"""
YELLOW TAXI TRIP DURATION MODEL

Model:
GradientBoostingRegressor

Split:
Time-based 80/20

Training rows:
{len(train_df):,}

Test rows:
{len(test_df):,}

Training period:
{train_df["pickup_datetime"].min()}
to
{train_df["pickup_datetime"].max()}

Test period:
{test_df["pickup_datetime"].min()}
to
{test_df["pickup_datetime"].max()}

MAE:
{mae:.2f} seconds
{mae / 60:.2f} minutes

RMSE:
{rmse:.2f} seconds
{rmse / 60:.2f} minutes

R2:
{r2:.4f}

Within +/- 2 minutes:
{within_2:.2f}%

Within +/- 5 minutes:
{within_5:.2f}%

Within +/- 10 minutes:
{within_10:.2f}%

Actual mean:
{actual_mean:.2f} minutes

Predicted mean:
{predicted_mean:.2f} minutes

Actual median:
{actual_median:.2f} minutes

Predicted median:
{predicted_median:.2f} minutes
""".strip()

    METRICS_FILE.write_text(
        metrics_text,
        encoding="utf-8",
    )

    print(
        f"Saved metrics -> "
        f"{METRICS_FILE}"
    )

    # =================================================================
    # SAVE PREDICTIONS
    # =================================================================

    prediction_output = test_df[
        [
            "trip_distance",
            "passenger_count",
            "PULocationID",
            "DOLocationID",
            "RatecodeID",
            "payment_type",
            "pickup_hour",
            "pickup_dow",
            "is_weekend",
            "pickup_month",
            "pickup_datetime",
        ]
    ].copy()

    prediction_output["actual"] = (
        actual_values
    )

    prediction_output["predicted"] = (
        predictions
    )

    prediction_output["abs_error"] = (
        abs_error
    )

    prediction_output["actual_minutes"] = (
        actual_values / 60
    )

    prediction_output["predicted_minutes"] = (
        predictions / 60
    )

    prediction_output["error_minutes"] = (
        abs_error / 60
    )

    prediction_output.to_parquet(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"Saved test-set predictions -> "
        f"{PREDICTIONS_FILE} "
        f"({len(prediction_output):,} rows)"
    )

    # =================================================================
    # COMPLETE
    # =================================================================

    print("\n" + "=" * 70)
    print(
        "YELLOW DURATION TRAINING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()