"""
Validate Task 2 (Citywide Demand) and Task 3 (Zone Demand) models properly:
time-series cross-validation (not just the single time-based 80/20 split
used at training time) plus comparison against two baselines that make
sense for lag-based demand forecasting specifically.

Companion to validate_duration_models.py, which already covers Task 1.

IMPORTANT DIFFERENCE FROM validate_duration_models.py: this uses
sklearn's TimeSeriesSplit, NOT random KFold. Shuffling time-series data
before splitting would let a fold's training set contain hours that come
AFTER its test hours -- which never happens in production (you can't use
next week's actual trip counts to help forecast today) and would make the
reported accuracy look better than it really is. TimeSeriesSplit always
trains on an earlier contiguous chunk and tests on a later one, expanding
forward fold by fold -- consistent with the time-based split already used
in train_demand_model.py / train_zone_demand_model.py.

Baselines compared against (more meaningful for demand than a flat mean):
  1. Mean baseline       -- always predict the train fold's average.
  2. Persistence baseline -- predict "same as 1 hour ago" (lag_1h).
  3. Seasonal-naive baseline -- predict "same as this hour last week"
                                (lag_168h). Often a surprisingly strong,
                                completely model-free baseline for data
                                with a weekly rhythm like taxi demand.

Usage:
    python src/validate_demand_models.py
    python src/validate_demand_models.py --zone-sample-rows 200000
    python src/validate_demand_models.py --full   # no subsampling, Task 3 may be slow
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

LAGS = [1, 2, 3, 24, 24 * 7]
N_SPLITS = 5
DEFAULT_ZONE_SAMPLE_ROWS = 400_000  # Task 3 can have millions of rows; refitting
                                     # GradientBoostingRegressor 5x on all of them
                                     # can take a long time -- subsample by default.

CONFIGS = {
    "Green Taxi": {
        "hourly_demand": PROCESSED_DIR / "hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "zone_hourly_demand.parquet",
        "demand_model": OUTPUTS_DIR / "demand_model.joblib",
        "zone_demand_model": OUTPUTS_DIR / "zone_demand_model.joblib",
    },
    "HVFHV (Uber/Lyft)": {
        "hourly_demand": PROCESSED_DIR / "hvfhv_hourly_demand.parquet",
        "zone_hourly_demand": PROCESSED_DIR / "hvfhv_zone_hourly_demand.parquet",
        "demand_model": OUTPUTS_DIR / "hvfhv_demand_model.joblib",
        "zone_demand_model": OUTPUTS_DIR / "hvfhv_zone_demand_model.joblib",
    },
}


# ---------------------------------------------------------------------------
# Feature building -- mirrors build_features() in train_demand_model.py /
# train_zone_demand_model.py exactly, so the CV is validating the SAME
# feature definitions the saved model actually learned from.
# ---------------------------------------------------------------------------

def build_hourly_features(hourly_demand_path: Path) -> pd.DataFrame:
    df = duckdb.sql(
        f"SELECT pickup_hour_ts, trip_count FROM read_parquet('{hourly_demand_path.as_posix()}') ORDER BY 1"
    ).df()
    df["pickup_hour_ts"] = pd.to_datetime(df["pickup_hour_ts"])
    full_index = pd.date_range(df["pickup_hour_ts"].min(), df["pickup_hour_ts"].max(), freq="h")
    s = df.set_index("pickup_hour_ts")["trip_count"].reindex(full_index, fill_value=0)

    feat = pd.DataFrame({"trip_count": s})
    for lag in LAGS:
        feat[f"lag_{lag}h"] = feat["trip_count"].shift(lag)
    feat["rolling_mean_24h"] = feat["trip_count"].shift(1).rolling(24).mean()
    feat["hour"] = feat.index.hour
    feat["dow"] = feat.index.dayofweek  # NOTE: matches pandas .dt.dayofweek used at
                                         # training time for this task (Mon=0..Sun=6) --
                                         # different convention from the DuckDB-based
                                         # duckdb_dow() used for the duration models.
    feat["is_weekend"] = feat["dow"].isin([5, 6]).astype(int)
    feat["month"] = feat.index.month
    return feat.dropna()


def build_zone_features(zone_hourly_demand_path: Path) -> pd.DataFrame:
    df = duckdb.sql(
        f"SELECT pickup_hour_ts, zone_id, trip_count FROM read_parquet('{zone_hourly_demand_path.as_posix()}')"
    ).df()
    df["pickup_hour_ts"] = pd.to_datetime(df["pickup_hour_ts"])

    pivot = df.pivot_table(index="pickup_hour_ts", columns="zone_id", values="trip_count", fill_value=0)
    full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq="h")
    pivot = pivot.reindex(full_index, fill_value=0)
    pivot.index.name = "pickup_hour_ts"

    long_df = pivot.stack().rename("trip_count").reset_index()
    long_df.columns = ["pickup_hour_ts", "zone_id", "trip_count"]
    long_df = long_df.sort_values(["zone_id", "pickup_hour_ts"])

    for lag in LAGS:
        long_df[f"lag_{lag}h"] = long_df.groupby("zone_id")["trip_count"].shift(lag)
    long_df["rolling_mean_24h"] = long_df.groupby("zone_id")["trip_count"].shift(1).rolling(24).mean()
    long_df["hour"] = long_df["pickup_hour_ts"].dt.hour
    long_df["dow"] = long_df["pickup_hour_ts"].dt.dayofweek
    long_df["is_weekend"] = long_df["dow"].isin([5, 6]).astype(int)
    long_df["month"] = long_df["pickup_hour_ts"].dt.month

    long_df = long_df.dropna()
    # Sort chronologically (not by zone) so TimeSeriesSplit's train/test
    # boundary is a genuine global point in time, matching how the model
    # will actually be used -- forecasting forward from "now" across zones.
    return long_df.sort_values("pickup_hour_ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CV + baselines
# ---------------------------------------------------------------------------

def run_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = {"model": [], "mean_baseline": [], "persistence": [], "seasonal_naive": []}

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        m = clone(model)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        fold_metrics["model"].append(_metrics(y_test, preds))

        mean_pred = np.full(len(y_test), y_train.mean())
        fold_metrics["mean_baseline"].append(_metrics(y_test, mean_pred))

        if "lag_1h" in X_test.columns:
            fold_metrics["persistence"].append(_metrics(y_test, X_test["lag_1h"].values))
        if "lag_168h" in X_test.columns:
            fold_metrics["seasonal_naive"].append(_metrics(y_test, X_test["lag_168h"].values))

        print(f"    Fold {fold_i}/{n_splits}: model MAE {fold_metrics['model'][-1]['mae']:.2f} "
              f"| R2 {fold_metrics['model'][-1]['r2']:.3f}  "
              f"(train n={len(train_idx):,}, test n={len(test_idx):,})")

    return fold_metrics


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def _summarize(fold_metrics: list) -> dict:
    if not fold_metrics:
        return {}
    return {
        "mae_mean": np.mean([f["mae"] for f in fold_metrics]),
        "mae_std": np.std([f["mae"] for f in fold_metrics]),
        "rmse_mean": np.mean([f["rmse"] for f in fold_metrics]),
        "r2_mean": np.mean([f["r2"] for f in fold_metrics]),
        "r2_std": np.std([f["r2"] for f in fold_metrics]),
    }


def print_comparison(label: str, results: dict):
    model_s = _summarize(results["model"])
    mean_s = _summarize(results["mean_baseline"])
    pers_s = _summarize(results["persistence"])
    seas_s = _summarize(results["seasonal_naive"])

    print(f"\n  {label} -- {N_SPLITS}-fold time-series CV result:")
    print(f"    Model:            MAE {model_s['mae_mean']:.2f} (+/- {model_s['mae_std']:.2f})  "
          f"R2 {model_s['r2_mean']:.3f} (+/- {model_s['r2_std']:.3f})")
    if mean_s:
        print(f"    Mean baseline:    MAE {mean_s['mae_mean']:.2f}  R2 {mean_s['r2_mean']:.3f}")
    if pers_s:
        print(f"    Persistence (t-1h): MAE {pers_s['mae_mean']:.2f}  R2 {pers_s['r2_mean']:.3f}")
    if seas_s:
        print(f"    Seasonal-naive (t-1wk): MAE {seas_s['mae_mean']:.2f}  R2 {seas_s['r2_mean']:.3f}")

    if mean_s:
        print(f"    -> Model beats mean baseline by {(1 - model_s['mae_mean']/mean_s['mae_mean'])*100:.0f}%")
    if pers_s:
        print(f"    -> Model beats persistence by {(1 - model_s['mae_mean']/pers_s['mae_mean'])*100:.0f}%")
    if seas_s:
        print(f"    -> Model beats seasonal-naive by {(1 - model_s['mae_mean']/seas_s['mae_mean'])*100:.0f}%")

    return {"label": label, "model": model_s, "mean_baseline": mean_s,
            "persistence": pers_s, "seasonal_naive": seas_s}


# ---------------------------------------------------------------------------

def validate_task2(vehicle: str, cfg: dict):
    print(f"\n{'-'*70}\nTask 2 -- Citywide Demand ({vehicle})\n{'-'*70}")
    if not cfg["demand_model"].exists() or not cfg["hourly_demand"].exists():
        print("  SKIP -- model or data file not found")
        return None

    model = load(cfg["demand_model"])
    feat = build_hourly_features(cfg["hourly_demand"])
    feature_cols = list(model.feature_names_in_)
    X, y = feat[feature_cols], feat["trip_count"]
    print(f"  Rows available: {len(feat):,}")

    results = run_cv(model, X, y)
    return print_comparison(f"Task 2 ({vehicle})", results)


def validate_task3(vehicle: str, cfg: dict, sample_rows: int, full: bool):
    print(f"\n{'-'*70}\nTask 3 -- Zone Demand ({vehicle})\n{'-'*70}")
    if not cfg["zone_demand_model"].exists() or not cfg["zone_hourly_demand"].exists():
        print("  SKIP -- model or data file not found")
        return None

    model = load(cfg["zone_demand_model"])
    feat = build_zone_features(cfg["zone_hourly_demand"])
    print(f"  Rows available: {len(feat):,}")

    if not full and len(feat) > sample_rows:
        stride = max(1, len(feat) // sample_rows)
        feat = feat.iloc[::stride].reset_index(drop=True)
        print(f"  Subsampling every {stride}th row (chronological stride, preserves time "
              f"order) -> {len(feat):,} rows for CV. Use --full to disable.")

    feature_cols = list(model.feature_names_in_)
    X, y = feat[feature_cols], feat["trip_count"]

    results = run_cv(model, X, y)
    return print_comparison(f"Task 3 ({vehicle})", results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone-sample-rows", type=int, default=DEFAULT_ZONE_SAMPLE_ROWS,
                         help=f"Max rows to use for Task 3 CV (default {DEFAULT_ZONE_SAMPLE_ROWS:,}). "
                              f"Ignored if --full is set.")
    parser.add_argument("--full", action="store_true",
                         help="Disable Task 3 subsampling -- may be slow on large datasets.")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for vehicle, cfg in CONFIGS.items():
        print(f"\n{'='*70}\n{vehicle}\n{'='*70}")
        r2 = validate_task2(vehicle, cfg)
        r3 = validate_task3(vehicle, cfg, args.zone_sample_rows, args.full)
        for r in (r2, r3):
            if r:
                all_results.append(r)

    report_path = OUTPUTS_DIR / "demand_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("Demand Model Validation Report\n")
        f.write(f"{N_SPLITS}-fold TimeSeriesSplit cross-validation\n")
        f.write("=" * 70 + "\n\n")
        for r in all_results:
            f.write(f"{r['label']}\n")
            m = r["model"]
            f.write(f"  Model MAE:  {m['mae_mean']:.2f} (+/- {m['mae_std']:.2f})\n")
            f.write(f"  Model R2:   {m['r2_mean']:.3f} (+/- {m['r2_std']:.3f})\n")
            if r["mean_baseline"]:
                f.write(f"  Mean baseline MAE:        {r['mean_baseline']['mae_mean']:.2f}\n")
            if r["persistence"]:
                f.write(f"  Persistence (t-1h) MAE:   {r['persistence']['mae_mean']:.2f}\n")
            if r["seasonal_naive"]:
                f.write(f"  Seasonal-naive (t-1wk) MAE: {r['seasonal_naive']['mae_mean']:.2f}\n")
            f.write("\n")
    print(f"\n{'='*70}\nSaved -> {report_path}")


if __name__ == "__main__":
    main()