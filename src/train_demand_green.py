"""
Task 2 — Taxi Demand Prediction (citywide, hourly).

Forecasts total pickups in the next hour, citywide, using lag features
(previous hours' demand) plus calendar features. This is a standard way to
turn a time-series forecasting problem into a supervised regression problem.

Usage:
    python src/train_demand_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

LAGS = [1, 2, 3, 24, 24 * 7]  # last hour, 2/3 hrs ago, same hour yesterday, same hour last week


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("pickup_hour_ts").set_index("pickup_hour_ts")

    # Fill any missing hours (e.g. an hour with zero trips) with 0 before lagging
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_index, fill_value=0)
    df.index.name = "pickup_hour_ts"

    for lag in LAGS:
        df[f"lag_{lag}h"] = df["trip_count"].shift(lag)

    df["rolling_mean_24h"] = df["trip_count"].shift(1).rolling(24).mean()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["month"] = df.index.month

    return df.dropna()


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(PROCESSED_DIR / "hourly_demand.parquet")
    df = build_features(raw)
    print(f"Feature rows after lagging: {len(df):,}")

    feature_cols = [c for c in df.columns if c != "trip_count"]
    X = df[feature_cols]
    y = df["trip_count"]

    # Time-based split: train on the earlier portion, test on the most
    # recent slice. Never shuffle time series data.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    print("Training GradientBoostingRegressor...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n--- Test set performance (holdout = most recent 20% of hours) ---")
    print(f"MAE:  {mae:.1f} trips/hour")
    print(f"RMSE: {rmse:.1f} trips/hour")
    print(f"R^2:  {r2:.3f}")

    model_path = OUTPUTS_DIR / "demand_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model -> {model_path}")

    # Save predictions vs actuals for plotting/dashboarding later
    results = pd.DataFrame(
        {"actual": y_test.values, "predicted": preds}, index=y_test.index
    )
    results_path = OUTPUTS_DIR / "demand_predictions.parquet"
    results.to_parquet(results_path)
    print(f"Saved predictions -> {results_path}")


if __name__ == "__main__":
    main()
