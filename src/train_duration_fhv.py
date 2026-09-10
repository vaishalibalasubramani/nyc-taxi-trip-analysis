"""
Task 1 (FHV) — Trip Duration Prediction.

Predicts FHV trip duration using pickup/dropoff zones,
shared-ride flag, and calendar features.

A representative sample is used because GradientBoostingRegressor
is computationally expensive on very large datasets.

Usage:
    python src/train_duration_fhv.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROCESSED_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)

OUTPUTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
)


# ---------------------------------------------------------------
# TRAINING SAMPLE
# ---------------------------------------------------------------

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42


FEATURE_COLS = [
    "PUlocationID",
    "DOlocationID",
    "SR_Flag",
    "pickup_hour",
    "pickup_dow",
    "is_weekend",
    "pickup_month",
]

TARGET_COL = "trip_duration_s"


def main():

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # LOAD FHV FEATURES
    # ---------------------------------------------------------------

    input_path = (
        PROCESSED_DIR /
        "fhv_trip_features.parquet"
    )

    df = pd.read_parquet(input_path)

    print(
        f"Loaded {len(df):,} FHV rows"
    )

    # ---------------------------------------------------------------
    # CLEAN DATA
    # ---------------------------------------------------------------

    df = df.dropna(
        subset=FEATURE_COLS + [TARGET_COL]
    )

    print(
        f"Rows after dropping NaNs: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------------
    # SAMPLE DATA
    # ---------------------------------------------------------------

    if len(df) > SAMPLE_SIZE:

        print(
            f"\nSampling {SAMPLE_SIZE:,} rows "
            f"from {len(df):,} available rows..."
        )

        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    else:

        print(
            f"\nDataset is smaller than sample size. "
            f"Using all {len(df):,} rows."
        )

    print(
        f"Rows used for modeling: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------------
    # FEATURES / TARGET
    # ---------------------------------------------------------------

    X = df[FEATURE_COLS]

    y = df[TARGET_COL]

    # ---------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nTraining rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Test rows:     "
        f"{len(X_test):,}"
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    print(
        "\nTraining FHV "
        "GradientBoostingRegressor..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    preds = model.predict(
        X_test
    )

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        preds,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds,
        )
    )

    r2 = r2_score(
        y_test,
        preds,
    )

    # ---------------------------------------------------------------
    # PERFORMANCE
    # ---------------------------------------------------------------

    print(
        "\n--- FHV Test Set Performance ---"
    )

    print(
        f"MAE:  {mae:.1f} seconds"
    )

    print(
        f"RMSE: {rmse:.1f} seconds"
    )

    print(
        f"R^2:  {r2:.3f}"
    )

    # ---------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    importances = (
        pd.Series(
            model.feature_importances_,
            index=FEATURE_COLS,
        )
        .sort_values(ascending=False)
    )

    print(
        "\nFeature importances:"
    )

    print(
        importances
    )

    # ---------------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------------

    model_path = (
        OUTPUTS_DIR /
        "fhv_duration_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nSaved model -> "
        f"{model_path}"
    )

    # ---------------------------------------------------------------
    # SAVE METRICS
    # ---------------------------------------------------------------

    metrics_path = (
        OUTPUTS_DIR /
        "fhv_duration_model_metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"Training sample size: "
            f"{len(X_train):,}\n"
        )

        f.write(
            f"Test sample size: "
            f"{len(X_test):,}\n\n"
        )

        f.write(
            f"MAE:  {mae:.2f} seconds\n"
        )

        f.write(
            f"RMSE: {rmse:.2f} seconds\n"
        )

        f.write(
            f"R2:   {r2:.4f}\n\n"
        )

        f.write(
            "Feature importances:\n"
        )

        f.write(
            importances.to_string()
        )

    print(
        f"Saved metrics -> "
        f"{metrics_path}"
    )

    # ---------------------------------------------------------------
    # SAVE HELD-OUT TEST PREDICTIONS
    # ---------------------------------------------------------------

    predictions_df = X_test.copy()

    predictions_df["actual"] = (
        y_test.values
    )

    predictions_df["predicted"] = (
        preds
    )

    predictions_df["abs_error"] = (
        predictions_df["actual"]
        - predictions_df["predicted"]
    ).abs()

    predictions_path = (
        OUTPUTS_DIR /
        "fhv_duration_predictions.parquet"
    )

    predictions_df.to_parquet(
        predictions_path,
        index=False,
    )

    print(
        f"Saved test-set predictions -> "
        f"{predictions_path} "
        f"({len(predictions_df):,} rows)"
    )


if __name__ == "__main__":
    main()