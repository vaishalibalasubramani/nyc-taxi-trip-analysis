"""
Task 2 (FHV) — Citywide Hourly Demand Prediction.

Forecasts total FHV pickups for the next hour using historical demand
(lag features) and calendar features.

Usage:
    python src/train_demand_fhv.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROCESSED_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)

OUTPUTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
)

LAGS = [1, 2, 3, 24, 24 * 7]


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = (
        df.sort_values("pickup_hour_ts")
        .set_index("pickup_hour_ts")
    )

    # Fill missing hours with zero demand
    full_index = pd.date_range(
        df.index.min(),
        df.index.max(),
        freq="h",
    )

    df = df.reindex(
        full_index,
        fill_value=0,
    )

    df.index.name = "pickup_hour_ts"

    # Lag features
    for lag in LAGS:
        df[f"lag_{lag}h"] = (
            df["trip_count"].shift(lag)
        )

    # Previous 24-hour average
    df["rolling_mean_24h"] = (
        df["trip_count"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    # Calendar features
    df["hour"] = df.index.hour

    df["dow"] = df.index.dayofweek

    df["is_weekend"] = (
        df["dow"]
        .isin([5, 6])
        .astype(int)
    )

    df["month"] = df.index.month

    return df.dropna()


def main():

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # LOAD FHV HOURLY DEMAND
    # ---------------------------------------------------------------

    input_path = (
        PROCESSED_DIR /
        "fhv_hourly_demand.parquet"
    )

    raw = pd.read_parquet(input_path)

    print(
        f"Loaded {len(raw):,} FHV hourly rows"
    )

    # ---------------------------------------------------------------
    # BUILD FEATURES
    # ---------------------------------------------------------------

    df = build_features(raw)

    print(
        f"Feature rows after lagging: {len(df):,}"
    )

    feature_cols = [
        c for c in df.columns
        if c != "trip_count"
    ]

    X = df[feature_cols]

    y = df["trip_count"]

    # ---------------------------------------------------------------
    # TIME-BASED TRAIN / TEST SPLIT
    # ---------------------------------------------------------------

    split_idx = int(
        len(df) * 0.8
    )

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(
        f"\nTraining rows: {len(X_train):,}"
    )

    print(
        f"Test rows:     {len(X_test):,}"
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    print(
        "\nTraining FHV GradientBoostingRegressor..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------------

    preds = model.predict(
        X_test
    )

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
        "\n--- FHV Test Set Performance "
        "(holdout = most recent 20% of hours) ---"
    )

    print(
        f"MAE:  {mae:.1f} trips/hour"
    )

    print(
        f"RMSE: {rmse:.1f} trips/hour"
    )

    print(
        f"R^2:  {r2:.3f}"
    )

    # ---------------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------------

    model_path = (
        OUTPUTS_DIR /
        "fhv_demand_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nSaved model -> {model_path}"
    )

    # ---------------------------------------------------------------
    # SAVE PREDICTIONS
    # ---------------------------------------------------------------

    results = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": preds,
        },
        index=y_test.index,
    )

    predictions_path = (
        OUTPUTS_DIR /
        "fhv_demand_predictions.parquet"
    )

    results.to_parquet(
        predictions_path
    )

    print(
        f"Saved predictions -> {predictions_path}"
    )


if __name__ == "__main__":
    main()