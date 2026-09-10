"""
Task 3 (FHV) — Zone-Level Hourly Demand Prediction.

Forecasts hourly FHV pickups for each NYC taxi zone using historical
zone demand and calendar features.

To keep GradientBoostingRegressor computationally practical, only a
500,000-row sample from the training period is used. The complete
most-recent 20% of hours is retained as the unseen test set.

Usage:
    python src/train_zone_demand_fhv.py
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

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

LAGS = [1, 2, 3, 24, 24 * 7]


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["pickup_hour_ts"] = pd.to_datetime(
        df["pickup_hour_ts"]
    )

    # ---------------------------------------------------------------
    # COMPLETE HOUR × ZONE GRID
    # ---------------------------------------------------------------

    hours = pd.date_range(
        df["pickup_hour_ts"].min(),
        df["pickup_hour_ts"].max(),
        freq="h",
    )

    zones = df["zone_id"].dropna().unique()

    full_index = pd.MultiIndex.from_product(
        [hours, zones],
        names=[
            "pickup_hour_ts",
            "zone_id",
        ],
    )

    # Preserve zone metadata before creating complete grid
    zone_metadata = (
        df[
            ["zone_id", "zone_name", "borough"]
        ]
        .drop_duplicates("zone_id")
    )

    df = (
        df[
            [
                "pickup_hour_ts",
                "zone_id",
                "trip_count",
            ]
        ]
        .set_index(
            [
                "pickup_hour_ts",
                "zone_id",
            ]
        )
        .reindex(
            full_index,
            fill_value=0,
        )
        .reset_index()
    )

    df = df.merge(
        zone_metadata,
        on="zone_id",
        how="left",
    )

    # ---------------------------------------------------------------
    # SORT BY ZONE AND TIME
    # ---------------------------------------------------------------

    df = df.sort_values(
        [
            "zone_id",
            "pickup_hour_ts",
        ]
    )

    # ---------------------------------------------------------------
    # LAG FEATURES
    # ---------------------------------------------------------------

    for lag in LAGS:

        df[f"lag_{lag}h"] = (
            df.groupby("zone_id")["trip_count"]
            .shift(lag)
        )

    # ---------------------------------------------------------------
    # ROLLING 24-HOUR MEAN
    # ---------------------------------------------------------------

    df["rolling_mean_24h"] = (
        df.groupby("zone_id")["trip_count"]
        .transform(
            lambda x:
                x.shift(1)
                .rolling(24)
                .mean()
        )
    )

    # ---------------------------------------------------------------
    # CALENDAR FEATURES
    # ---------------------------------------------------------------

    df["hour"] = (
        df["pickup_hour_ts"].dt.hour
    )

    df["dow"] = (
        df["pickup_hour_ts"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["dow"]
        .isin([5, 6])
        .astype(int)
    )

    df["month"] = (
        df["pickup_hour_ts"].dt.month
    )

    return df.dropna()


def main():

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # LOAD FHV ZONE DEMAND
    # ---------------------------------------------------------------

    input_path = (
        PROCESSED_DIR /
        "fhv_zone_hourly_demand.parquet"
    )

    raw = pd.read_parquet(
        input_path
    )

    print(
        f"Loaded {len(raw):,} FHV zone-hour rows"
    )

    # ---------------------------------------------------------------
    # BUILD FEATURES
    # ---------------------------------------------------------------

    df = build_features(raw)

    print(
        f"Feature rows after lagging: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------------
    # FEATURES / TARGET
    # ---------------------------------------------------------------

    feature_cols = [
        "zone_id",
        "lag_1h",
        "lag_2h",
        "lag_3h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_24h",
        "hour",
        "dow",
        "is_weekend",
        "month",
    ]

    X = df[feature_cols]

    y = df["trip_count"]

    # ---------------------------------------------------------------
    # TIME-BASED SPLIT
    # ---------------------------------------------------------------

    unique_hours = np.sort(
        df["pickup_hour_ts"].unique()
    )

    split_idx = int(
        len(unique_hours) * 0.8
    )

    split_time = unique_hours[split_idx]

    train_mask = (
        df["pickup_hour_ts"]
        < split_time
    )

    test_mask = (
        df["pickup_hour_ts"]
        >= split_time
    )

    X_train_full = X.loc[train_mask]
    y_train_full = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print(
        f"\nFull training rows: "
        f"{len(X_train_full):,}"
    )

    print(
        f"Test rows:          "
        f"{len(X_test):,}"
    )

    print(
        f"Test starts:        "
        f"{split_time}"
    )

    # ---------------------------------------------------------------
    # SAMPLE ONLY THE TRAINING PERIOD
    # ---------------------------------------------------------------

    if len(X_train_full) > SAMPLE_SIZE:

        print(
            f"\nSampling {SAMPLE_SIZE:,} rows "
            f"from the training period..."
        )

        sample_indices = (
            X_train_full
            .sample(
                n=SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )
            .index
        )

        X_train = X_train_full.loc[
            sample_indices
        ]

        y_train = y_train_full.loc[
            sample_indices
        ]

    else:

        X_train = X_train_full

        y_train = y_train_full

        print(
            "\nTraining data is smaller than "
            "sample size. Using all training rows."
        )

    print(
        f"Training rows used by model: "
        f"{len(X_train):,}"
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    print(
        "\nTraining FHV zone demand "
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
        "\n--- FHV Zone Demand "
        "Test Performance ---"
    )

    print(
        f"MAE:  {mae:.2f} trips/hour"
    )

    print(
        f"RMSE: {rmse:.2f} trips/hour"
    )

    print(
        f"R^2:  {r2:.3f}"
    )

    # ---------------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------------

    model_path = (
        OUTPUTS_DIR /
        "fhv_zone_demand_model.joblib"
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
    # SAVE PREDICTIONS
    # ---------------------------------------------------------------

    results = df.loc[
        test_mask,
        [
            "pickup_hour_ts",
            "zone_id",
            "zone_name",
            "borough",
            "trip_count",
        ],
    ].copy()

    results["actual"] = (
        results["trip_count"]
    )

    results["predicted"] = (
        preds
    )

    results["abs_error"] = (
        results["actual"]
        - results["predicted"]
    ).abs()

    results = results.drop(
        columns=["trip_count"]
    )

    predictions_path = (
        OUTPUTS_DIR /
        "fhv_zone_demand_predictions.parquet"
    )

    results.to_parquet(
        predictions_path,
        index=False,
    )

    print(
        f"Saved predictions -> "
        f"{predictions_path}"
    )


if __name__ == "__main__":
    main()