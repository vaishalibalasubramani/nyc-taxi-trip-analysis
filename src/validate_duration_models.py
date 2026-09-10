"""
Validate the trip duration models properly: k-fold cross-validation (not
just the single 80/20 split used at training time) plus comparison against
two baselines, so the reported MAE/R2 are defensible rather than a
one-off lucky split.

Runs for BOTH vehicle types (Green Taxi + HVFHV) using whichever
FEATURE_COLS each saved model was actually trained on (read via
model.feature_names_in_ -- no need to hardcode per vehicle).

Baselines compared against:
  1. Mean baseline   -- always predict the average trip duration (the
                         floor any real model must clear).
  2. Physics baseline -- predict duration from distance alone, assuming a
                         flat 15 mph average city speed. A model that
                         barely beats this isn't learning much beyond
                         "further = longer."

Usage:
    python src/validate_duration_models.py
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

TARGET_COL = "trip_duration_s"
ASSUMED_MPH = 15.0  # flat average city speed for the physics baseline
N_FOLDS = 5
RANDOM_STATE = 42

CONFIGS = {
    "Green Taxi": {
        "trip_features": PROCESSED_DIR / "trip_features.parquet",
        "model": OUTPUTS_DIR / "duration_model.joblib",
        "distance_col": "trip_distance",
    },
    "HVFHV (Uber/Lyft)": {
        "trip_features": PROCESSED_DIR / "hvfhv_trip_features.parquet",
        "model": OUTPUTS_DIR / "hvfhv_duration_model.joblib",
        "distance_col": "trip_miles",
    },
}


def load_data(cfg: dict) -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM read_parquet('{cfg['trip_features'].as_posix()}')").df()
    return df


def encode_categoricals(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Mirrors the encoding done at training time: any non-numeric feature
    column (e.g. hvfhs_license_num) gets astype('category').cat.codes.
    Uses is_numeric_dtype rather than `dtype == object` since pandas 3.0+
    introduced a separate string dtype that doesn't compare equal to
    `object`, which would otherwise silently skip the encoding."""
    df = df.copy()
    for c in feature_cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("category").cat.codes
    return df


def run_validation(vehicle: str, cfg: dict):
    print(f"\n{'=' * 70}\n{vehicle}\n{'=' * 70}")

    if not cfg["model"].exists():
        print(f"  SKIP -- model not found at {cfg['model']}")
        return None
    if not cfg["trip_features"].exists():
        print(f"  SKIP -- features not found at {cfg['trip_features']}")
        return None

    saved_model = load(cfg["model"])
    feature_cols = list(saved_model.feature_names_in_)
    print(f"  Features used by saved model: {feature_cols}")

    df = load_data(cfg)
    df = encode_categoricals(df, feature_cols)
    df = df.dropna(subset=feature_cols + [TARGET_COL])
    X = df[feature_cols]
    y = df[TARGET_COL]
    print(f"  Rows available for validation: {len(df):,}")

    # -----------------------------------------------------------------
    # 1. K-fold cross-validation of a FRESH model with the SAME
    #    hyperparameters as the saved one (clone() copies params, not
    #    the fitted state) -- gives mean +/- std instead of one split.
    # -----------------------------------------------------------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fresh_model = clone(saved_model)

    fold_mae, fold_rmse, fold_r2 = [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        m = clone(fresh_model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx]
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        print(f"    Fold {fold_i}/{N_FOLDS}: MAE {mae:.1f}s | RMSE {rmse:.1f}s | R2 {r2:.3f}")

    cv_mae_mean, cv_mae_std = np.mean(fold_mae), np.std(fold_mae)
    cv_rmse_mean = np.mean(fold_rmse)
    cv_r2_mean, cv_r2_std = np.mean(fold_r2), np.std(fold_r2)

    print(f"\n  {N_FOLDS}-fold CV result:")
    print(f"    MAE:  {cv_mae_mean:.1f}s  (+/- {cv_mae_std:.1f}s across folds)")
    print(f"    RMSE: {cv_rmse_mean:.1f}s")
    print(f"    R2:   {cv_r2_mean:.3f}  (+/- {cv_r2_std:.3f} across folds)")

    # -----------------------------------------------------------------
    # 2. Baselines, evaluated on the same cross_val_predict-style
    #    out-of-fold predictions for a fair apples-to-apples comparison.
    # -----------------------------------------------------------------
    # Baseline A: always predict the mean duration.
    dummy_preds = cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=kf)
    dummy_mae = mean_absolute_error(y, dummy_preds)
    dummy_r2 = r2_score(y, dummy_preds)

    # Baseline B: distance / assumed flat speed -- no fitting needed,
    # just a direct physics-based formula, computed out-of-fold trivially
    # since it doesn't depend on training data at all.
    distance_col = cfg["distance_col"]
    physics_preds = (X[distance_col] / ASSUMED_MPH) * 3600
    physics_mae = mean_absolute_error(y, physics_preds)
    physics_r2 = r2_score(y, physics_preds)

    print(f"\n  Baseline -- always predict mean duration:")
    print(f"    MAE: {dummy_mae:.1f}s | R2: {dummy_r2:.3f}")
    print(f"  Baseline -- distance / {ASSUMED_MPH:.0f}mph:")
    print(f"    MAE: {physics_mae:.1f}s | R2: {physics_r2:.3f}")

    improvement_vs_mean = (1 - cv_mae_mean / dummy_mae) * 100
    improvement_vs_physics = (1 - cv_mae_mean / physics_mae) * 100 if physics_mae > 0 else float("nan")
    print(f"\n  Model improves MAE by {improvement_vs_mean:.0f}% vs. the mean-duration baseline")
    print(f"  Model improves MAE by {improvement_vs_physics:.0f}% vs. the distance/{ASSUMED_MPH:.0f}mph baseline")

    return {
        "vehicle": vehicle,
        "n_rows": len(df),
        "cv_mae_mean": cv_mae_mean, "cv_mae_std": cv_mae_std,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_r2_mean": cv_r2_mean, "cv_r2_std": cv_r2_std,
        "dummy_mae": dummy_mae, "dummy_r2": dummy_r2,
        "physics_mae": physics_mae, "physics_r2": physics_r2,
        "improvement_vs_mean_pct": improvement_vs_mean,
        "improvement_vs_physics_pct": improvement_vs_physics,
    }


def main():
    results = []
    for vehicle, cfg in CONFIGS.items():
        r = run_validation(vehicle, cfg)
        if r:
            results.append(r)

    if not results:
        print("\nNo models validated -- check that models and trip_features exist.")
        return

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    report_path = OUTPUTS_DIR / "duration_validation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Duration Model Validation Report\n")
        f.write(f"{N_FOLDS}-fold cross-validation, {ASSUMED_MPH:.0f}mph physics baseline\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"{r['vehicle']}\n")
            f.write(f"  Rows validated: {r['n_rows']:,}\n")
            f.write(f"  CV MAE:  {r['cv_mae_mean']:.1f}s (+/- {r['cv_mae_std']:.1f}s)\n")
            f.write(f"  CV RMSE: {r['cv_rmse_mean']:.1f}s\n")
            f.write(f"  CV R2:   {r['cv_r2_mean']:.3f} (+/- {r['cv_r2_std']:.3f})\n")
            f.write(f"  Mean-duration baseline MAE:      {r['dummy_mae']:.1f}s (R2 {r['dummy_r2']:.3f})\n")
            f.write(f"  Distance/{ASSUMED_MPH:.0f}mph baseline MAE:   {r['physics_mae']:.1f}s (R2 {r['physics_r2']:.3f})\n")
            f.write(f"  Improvement vs. mean baseline:   {r['improvement_vs_mean_pct']:.0f}%\n")
            f.write(f"  Improvement vs. physics baseline: {r['improvement_vs_physics_pct']:.0f}%\n\n")
    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    main()