"""
Task 3 — Taxi Zone Demand Forecasting.

Same idea as Task 2 (lag + calendar features -> regression), but at the
pickup-zone level: one model, with zone_id as a categorical feature, so it
can predict next-hour demand for every zone at once rather than training
264 separate models.

Usage:
    python src/train_zone_demand_model.py
"""

from pathlib import Path


import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

LAGS = [1, 2, 3, 24, 24 * 7]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Pivot to a full (hour x zone) grid so zones with zero trips in a given
    # hour are represented as 0, not missing -- important for lag features.
    pivot = df.pivot_table(
        index="pickup_hour_ts", columns="zone_id", values="trip_count", fill_value=0
    )
    full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq="h")
    pivot = pivot.reindex(full_index, fill_value=0)
    pivot.index.name = "pickup_hour_ts"

    # Long format: one row per (hour, zone)
    long_df = pivot.stack().rename("trip_count").reset_index()
    long_df.columns = ["pickup_hour_ts", "zone_id", "trip_count"]
    long_df = long_df.sort_values(["zone_id", "pickup_hour_ts"])

    for lag in LAGS:
        long_df[f"lag_{lag}h"] = long_df.groupby("zone_id")["trip_count"].shift(lag)

    long_df["rolling_mean_24h"] = (
        long_df.groupby("zone_id")["trip_count"].shift(1).rolling(24).mean()
    )
    long_df["hour"] = long_df["pickup_hour_ts"].dt.hour
    long_df["dow"] = long_df["pickup_hour_ts"].dt.dayofweek
    long_df["is_weekend"] = long_df["dow"].isin([5, 6]).astype(int)
    long_df["month"] = long_df["pickup_hour_ts"].dt.month

    return long_df.dropna()


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(PROCESSED_DIR / "hvfhv_zone_hourly_demand.parquet")
    raw = raw[["pickup_hour_ts", "zone_id", "trip_count"]]
    df = build_features(raw)
    print(f"Feature rows after lagging: {len(df):,} (zones x hours)")

    feature_cols = [c for c in df.columns if c not in ("trip_count", "pickup_hour_ts")]
    X = df[feature_cols]
    y = df["trip_count"]

    # Time-based split on the timestamp, same rule as Task 2: no shuffling.
    cutoff = df["pickup_hour_ts"].quantile(0.8)
    train_mask = df["pickup_hour_ts"] < cutoff
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    print("Training GradientBoostingRegressor (zone_id as a feature)...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n--- Test set performance (holdout = most recent ~20% of hours) ---")
    print(f"MAE:  {mae:.2f} trips/zone/hour")
    print(f"RMSE: {rmse:.2f} trips/zone/hour")
    print(f"R^2:  {r2:.3f}")

    model_path = OUTPUTS_DIR / "hvfhv_zone_demand_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model -> {model_path}")

    results = df.loc[~train_mask, ["pickup_hour_ts", "zone_id"]].copy()
    results["actual"] = y_test.values
    results["predicted"] = preds
    results_path = OUTPUTS_DIR / "hvfhv_zone_demand_predictions.parquet"
    results.to_parquet(results_path)
    print(f"Saved predictions -> {results_path}")


if __name__ == "__main__":
    main()
