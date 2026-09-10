"""
Task 1 — Trip Duration Prediction (Green Taxi).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

FEATURE_COLS = [
    "trip_distance", "passenger_count", "PULocationID", "DOLocationID",
    "trip_type", "payment_type", "pickup_hour", "pickup_dow",
    "is_weekend", "pickup_month",
]
TARGET_COL = "trip_duration_s"


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED_DIR / "trip_features.parquet")
    print(f"Loaded {len(df):,} rows")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"Rows after dropping NaNs: {len(df):,}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8, random_state=42,
    )
    print("Training GradientBoostingRegressor...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n--- Test set performance ---")
    print(f"MAE:  {mae:.1f} seconds")
    print(f"RMSE: {rmse:.1f} seconds")
    print(f"R^2:  {r2:.3f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    model_path = OUTPUTS_DIR / "duration_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model -> {model_path}")

    metrics_path = OUTPUTS_DIR / "duration_model_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"MAE:  {mae:.2f} seconds\n")
        f.write(f"RMSE: {rmse:.2f} seconds\n")
        f.write(f"R2:   {r2:.4f}\n\n")
        f.write("Feature importances:\n")
        f.write(importances.to_string())
    print(f"Saved metrics -> {metrics_path}")

    # --- NEW: save the actual test-set rows (features + actual + predicted)
    # so the dashboard can show a genuine "held-out data, actual vs predicted"
    # comparison table rather than only aggregate metrics. ---
    predictions_df = X_test.copy()
    predictions_df["actual"] = y_test.values
    predictions_df["predicted"] = preds
    predictions_df["abs_error"] = (predictions_df["actual"] - predictions_df["predicted"]).abs()
    predictions_path = OUTPUTS_DIR / "duration_predictions.parquet"
    predictions_df.to_parquet(predictions_path)
    print(f"Saved test-set predictions -> {predictions_path} ({len(predictions_df):,} rows)")


if __name__ == "__main__":
    main()