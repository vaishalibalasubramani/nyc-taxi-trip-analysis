"""
Validate Task 2 (Citywide Demand) and Task 3 (Zone Demand) models.

Vehicles:
    - Yellow Taxi
    - FHV

Validation:
    - 5-fold TimeSeriesSplit
    - Mean baseline
    - Persistence baseline (t-1 hour)
    - Seasonal-naive baseline (t-1 week / 168 hours)

IMPORTANT:
    TimeSeriesSplit is used instead of random KFold because demand
    forecasting is a time-series problem. Training data must always
    occur before validation data.

Usage:
    python src/validate_demand_models.py

    For smaller/faster zone validation:
    python src/validate_demand_models.py --zone-sample-rows 200000

    For full zone validation:
    python src/validate_demand_models.py --full
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from joblib import load
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUTS_DIR = ROOT / "outputs"

VALIDATION_DIR = OUTPUTS_DIR / "validation"

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

LAGS = [
    1,
    2,
    3,
    24,
    24 * 7,
]

N_SPLITS = 5

DEFAULT_ZONE_SAMPLE_ROWS = 400_000


# ============================================================
# CONFIGURATION
# ============================================================

CONFIGS = {

    "Yellow Taxi": {

        # Task 2 -- citywide demand
        "hourly_demand":
            PROCESSED_DIR / "hourly_demand.parquet",

        # Task 3 -- zone demand
        "zone_hourly_demand":
            PROCESSED_DIR / "zone_hourly_demand.parquet",

        # Task 2 model
        "demand_model":
            OUTPUTS_DIR / "yellow_demand_model.joblib",

        # Task 3 model
        "zone_demand_model":
            OUTPUTS_DIR / "yellow_zone_demand_model.joblib",
    },

    "FHV": {

        # Task 2 -- citywide demand
        "hourly_demand":
            PROCESSED_DIR / "fhv_hourly_demand.parquet",

        # Task 3 -- zone demand
        "zone_hourly_demand":
            PROCESSED_DIR / "fhv_zone_hourly_demand.parquet",

        # Task 2 model
        "demand_model":
            OUTPUTS_DIR / "fhv_demand_model.joblib",

        # Task 3 model
        "zone_demand_model":
            OUTPUTS_DIR / "fhv_zone_demand_model.joblib",
    },
}


# ============================================================
# METRICS
# ============================================================

def _metrics(y_true, y_pred):

    return {
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),

        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),

        "r2": r2_score(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# SUMMARY
# ============================================================

def _summarize(fold_metrics):

    if not fold_metrics:
        return {}

    return {

        "mae_mean":
            np.mean(
                [
                    f["mae"]
                    for f in fold_metrics
                ]
            ),

        "mae_std":
            np.std(
                [
                    f["mae"]
                    for f in fold_metrics
                ]
            ),

        "rmse_mean":
            np.mean(
                [
                    f["rmse"]
                    for f in fold_metrics
                ]
            ),

        "rmse_std":
            np.std(
                [
                    f["rmse"]
                    for f in fold_metrics
                ]
            ),

        "r2_mean":
            np.mean(
                [
                    f["r2"]
                    for f in fold_metrics
                ]
            ),

        "r2_std":
            np.std(
                [
                    f["r2"]
                    for f in fold_metrics
                ]
            ),
    }


# ============================================================
# BUILD CITYWIDE FEATURES
# ============================================================

def build_hourly_features(
    hourly_demand_path: Path,
):

    print(
        f"\n  Loading hourly demand:"
    )

    print(
        f"    {hourly_demand_path}"
    )

    if not hourly_demand_path.exists():

        raise FileNotFoundError(
            f"Hourly demand file not found:\n"
            f"{hourly_demand_path}"
        )

    df = duckdb.sql(
        f"""
        SELECT
            pickup_hour_ts,
            trip_count
        FROM read_parquet(
            '{hourly_demand_path.as_posix()}'
        )
        ORDER BY 1
        """
    ).df()

    df["pickup_hour_ts"] = pd.to_datetime(
        df["pickup_hour_ts"]
    )

    print(
        f"    Rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Complete hourly index
    # --------------------------------------------------------

    full_index = pd.date_range(
        df["pickup_hour_ts"].min(),
        df["pickup_hour_ts"].max(),
        freq="h",
    )

    s = (
        df
        .set_index("pickup_hour_ts")["trip_count"]
        .reindex(
            full_index,
            fill_value=0,
        )
    )

    # --------------------------------------------------------
    # Feature dataframe
    # --------------------------------------------------------

    feat = pd.DataFrame(
        {
            "trip_count": s
        }
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in LAGS:

        feat[f"lag_{lag}h"] = (
            feat["trip_count"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling mean
    # --------------------------------------------------------

    feat["rolling_mean_24h"] = (
        feat["trip_count"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    feat["hour"] = (
        feat.index.hour
    )

    feat["dow"] = (
        feat.index.dayofweek
    )

    feat["is_weekend"] = (
        feat["dow"]
        .isin([5, 6])
        .astype(int)
    )

    feat["month"] = (
        feat.index.month
    )

    # --------------------------------------------------------
    # Remove rows without lag history
    # --------------------------------------------------------

    feat = feat.dropna()

    return feat


# ============================================================
# BUILD ZONE FEATURES
# ============================================================

def build_zone_features(
    zone_hourly_demand_path: Path,
):

    print(
        f"\n  Loading zone hourly demand:"
    )

    print(
        f"    {zone_hourly_demand_path}"
    )

    if not zone_hourly_demand_path.exists():

        raise FileNotFoundError(
            f"Zone hourly demand file not found:\n"
            f"{zone_hourly_demand_path}"
        )

    df = duckdb.sql(
        f"""
        SELECT
            pickup_hour_ts,
            zone_id,
            trip_count
        FROM read_parquet(
            '{zone_hourly_demand_path.as_posix()}'
        )
        """
    ).df()

    df["pickup_hour_ts"] = pd.to_datetime(
        df["pickup_hour_ts"]
    )

    print(
        f"    Rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Pivot into hour x zone
    # --------------------------------------------------------

    pivot = df.pivot_table(
        index="pickup_hour_ts",
        columns="zone_id",
        values="trip_count",
        fill_value=0,
    )

    # --------------------------------------------------------
    # Complete hourly index
    # --------------------------------------------------------

    full_index = pd.date_range(
        pivot.index.min(),
        pivot.index.max(),
        freq="h",
    )

    pivot = pivot.reindex(
        full_index,
        fill_value=0,
    )

    pivot.index.name = (
        "pickup_hour_ts"
    )

    # --------------------------------------------------------
    # Convert back to long format
    # --------------------------------------------------------

    long_df = (
        pivot
        .stack()
        .rename("trip_count")
        .reset_index()
    )

    long_df.columns = [
        "pickup_hour_ts",
        "zone_id",
        "trip_count",
    ]

    # --------------------------------------------------------
    # Sort by zone/time before calculating lags
    # --------------------------------------------------------

    long_df = long_df.sort_values(
        [
            "zone_id",
            "pickup_hour_ts",
        ]
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in LAGS:

        long_df[
            f"lag_{lag}h"
        ] = (
            long_df
            .groupby("zone_id")[
                "trip_count"
            ]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling 24-hour mean
    # --------------------------------------------------------

    long_df[
        "rolling_mean_24h"
    ] = (
        long_df
        .groupby("zone_id")[
            "trip_count"
        ]
        .transform(
            lambda s:
                s.shift(1)
                .rolling(24)
                .mean()
        )
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    long_df["hour"] = (
        long_df[
            "pickup_hour_ts"
        ].dt.hour
    )

    long_df["dow"] = (
        long_df[
            "pickup_hour_ts"
        ].dt.dayofweek
    )

    long_df["is_weekend"] = (
        long_df["dow"]
        .isin([5, 6])
        .astype(int)
    )

    long_df["month"] = (
        long_df[
            "pickup_hour_ts"
        ].dt.month
    )

    # --------------------------------------------------------
    # Remove rows without sufficient lag history
    # --------------------------------------------------------

    long_df = long_df.dropna()

    # --------------------------------------------------------
    # IMPORTANT:
    # Sort globally by timestamp.
    #
    # This ensures TimeSeriesSplit creates genuine
    # chronological train/test boundaries.
    # --------------------------------------------------------

    long_df = (
        long_df
        .sort_values("pickup_hour_ts")
        .reset_index(drop=True)
    )

    return long_df


# ============================================================
# TIME SERIES CROSS VALIDATION
# ============================================================

def run_cv(
    model,
    X,
    y,
    n_splits=N_SPLITS,
):

    tscv = TimeSeriesSplit(
        n_splits=n_splits
    )

    fold_metrics = {

        "model": [],

        "mean_baseline": [],

        "persistence": [],

        "seasonal_naive": [],
    }

    for fold_i, (
        train_idx,
        test_idx,
    ) in enumerate(
        tscv.split(X),
        1,
    ):

        print(
            f"\n    Fold "
            f"{fold_i}/{n_splits}"
        )

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        m = clone(model)

        m.fit(
            X_train,
            y_train,
        )

        preds = m.predict(
            X_test
        )

        model_metrics = _metrics(
            y_test,
            preds,
        )

        fold_metrics[
            "model"
        ].append(
            model_metrics
        )

        # ----------------------------------------------------
        # Mean baseline
        # ----------------------------------------------------

        mean_pred = np.full(
            len(y_test),
            y_train.mean(),
        )

        fold_metrics[
            "mean_baseline"
        ].append(
            _metrics(
                y_test,
                mean_pred,
            )
        )

        # ----------------------------------------------------
        # Persistence baseline
        # ----------------------------------------------------

        if "lag_1h" in X_test.columns:

            persistence_pred = (
                X_test[
                    "lag_1h"
                ].values
            )

            fold_metrics[
                "persistence"
            ].append(
                _metrics(
                    y_test,
                    persistence_pred,
                )
            )

        # ----------------------------------------------------
        # Seasonal naive baseline
        # ----------------------------------------------------

        if "lag_168h" in X_test.columns:

            seasonal_pred = (
                X_test[
                    "lag_168h"
                ].values
            )

            fold_metrics[
                "seasonal_naive"
            ].append(
                _metrics(
                    y_test,
                    seasonal_pred,
                )
            )

        print(
            f"      Model MAE: "
            f"{model_metrics['mae']:.2f}"
        )

        print(
            f"      Model RMSE: "
            f"{model_metrics['rmse']:.2f}"
        )

        print(
            f"      Model R²: "
            f"{model_metrics['r2']:.3f}"
        )

        print(
            f"      Train n: "
            f"{len(train_idx):,}"
        )

        print(
            f"      Test n: "
            f"{len(test_idx):,}"
        )

    return fold_metrics


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    label,
    results,
):

    model_s = _summarize(
        results["model"]
    )

    mean_s = _summarize(
        results["mean_baseline"]
    )

    pers_s = _summarize(
        results["persistence"]
    )

    seas_s = _summarize(
        results["seasonal_naive"]
    )

    print()
    print(
        f"  {label}"
    )

    print(
        f"  {N_SPLITS}-fold "
        f"TimeSeriesSplit result:"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        f"\n    Model:"
    )

    print(
        f"      MAE  : "
        f"{model_s['mae_mean']:.2f}"
        f" +/- "
        f"{model_s['mae_std']:.2f}"
    )

    print(
        f"      RMSE : "
        f"{model_s['rmse_mean']:.2f}"
        f" +/- "
        f"{model_s['rmse_std']:.2f}"
    )

    print(
        f"      R²   : "
        f"{model_s['r2_mean']:.3f}"
        f" +/- "
        f"{model_s['r2_std']:.3f}"
    )

    # --------------------------------------------------------
    # Mean baseline
    # --------------------------------------------------------

    if mean_s:

        print(
            f"\n    Mean baseline:"
        )

        print(
            f"      MAE  : "
            f"{mean_s['mae_mean']:.2f}"
        )

        print(
            f"      RMSE : "
            f"{mean_s['rmse_mean']:.2f}"
        )

        print(
            f"      R²   : "
            f"{mean_s['r2_mean']:.3f}"
        )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    if pers_s:

        print(
            f"\n    Persistence "
            f"(t-1h):"
        )

        print(
            f"      MAE  : "
            f"{pers_s['mae_mean']:.2f}"
        )

        print(
            f"      RMSE : "
            f"{pers_s['rmse_mean']:.2f}"
        )

        print(
            f"      R²   : "
            f"{pers_s['r2_mean']:.3f}"
        )

    # --------------------------------------------------------
    # Seasonal naive
    # --------------------------------------------------------

    if seas_s:

        print(
            f"\n    Seasonal-naive "
            f"(t-1week):"
        )

        print(
            f"      MAE  : "
            f"{seas_s['mae_mean']:.2f}"
        )

        print(
            f"      RMSE : "
            f"{seas_s['rmse_mean']:.2f}"
        )

        print(
            f"      R²   : "
            f"{seas_s['r2_mean']:.3f}"
        )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    print(
        "\n    Model comparison:"
    )

    if mean_s:

        improvement = (
            1
            - (
                model_s["mae_mean"]
                / mean_s["mae_mean"]
            )
        ) * 100

        print(
            f"      vs mean baseline: "
            f"{improvement:.2f}%"
        )

    if pers_s:

        improvement = (
            1
            - (
                model_s["mae_mean"]
                / pers_s["mae_mean"]
            )
        ) * 100

        print(
            f"      vs persistence: "
            f"{improvement:.2f}%"
        )

    if seas_s:

        improvement = (
            1
            - (
                model_s["mae_mean"]
                / seas_s["mae_mean"]
            )
        ) * 100

        print(
            f"      vs seasonal-naive: "
            f"{improvement:.2f}%"
        )

    # --------------------------------------------------------
    # PASS criterion
    # --------------------------------------------------------

    baselines = []

    if mean_s:
        baselines.append(
            mean_s["mae_mean"]
        )

    if pers_s:
        baselines.append(
            pers_s["mae_mean"]
        )

    if seas_s:
        baselines.append(
            seas_s["mae_mean"]
        )

    # We consider the model strong if it beats
    # the strongest baseline (lowest MAE).

    if baselines:

        best_baseline_mae = min(
            baselines
        )

        if (
            model_s["mae_mean"]
            < best_baseline_mae
        ):

            status = "PASS"

            print(
                "\n    RESULT: PASS"
            )

            print(
                "    Model beats all available "
                "demand baselines."
            )

        else:

            status = "CHECK"

            print(
                "\n    RESULT: CHECK"
            )

            print(
                "    Model does not beat all "
                "demand baselines."
            )

    else:

        status = "CHECK"

        print(
            "\n    RESULT: CHECK"
        )

    return {

        "label":
            label,

        "status":
            status,

        "model":
            model_s,

        "mean_baseline":
            mean_s,

        "persistence":
            pers_s,

        "seasonal_naive":
            seas_s,
    }


# ============================================================
# VALIDATE TASK 2
# ============================================================

def validate_task2(
    vehicle,
    cfg,
):

    print()
    print("-" * 70)

    print(
        f"Task 2 -- Citywide Demand "
        f"({vehicle})"
    )

    print(
        "-" * 70
    )

    model_path = cfg[
        "demand_model"
    ]

    data_path = cfg[
        "hourly_demand"
    ]

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not model_path.exists():

        print(
            "\n  SKIP -- demand model "
            "not found:"
        )

        print(
            f"    {model_path}"
        )

        return None

    if not data_path.exists():

        print(
            "\n  SKIP -- hourly demand "
            "data not found:"
        )

        print(
            f"    {data_path}"
        )

        return None

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\n  Loading model..."
    )

    model = load(
        model_path
    )

    print(
        f"  Model type: "
        f"{type(model).__name__}"
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    feat = build_hourly_features(
        data_path
    )

    # --------------------------------------------------------
    # Model features
    # --------------------------------------------------------

    if not hasattr(
        model,
        "feature_names_in_",
    ):

        raise ValueError(
            f"{vehicle} Task 2 model "
            "does not expose feature_names_in_."
        )

    feature_cols = list(
        model.feature_names_in_
    )

    print(
        "\n  Model features:"
    )

    for feature in feature_cols:

        print(
            f"    - {feature}"
        )

    missing = [
        col
        for col in feature_cols
        if col not in feat.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing Task 2 features "
            f"for {vehicle}:\n"
            f"{missing}"
        )

    X = feat[
        feature_cols
    ]

    y = feat[
        "trip_count"
    ]

    print(
        f"\n  Rows available: "
        f"{len(feat):,}"
    )

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    results = run_cv(
        model,
        X,
        y,
    )

    return print_comparison(
        f"Task 2 ({vehicle})",
        results,
    )


# ============================================================
# VALIDATE TASK 3
# ============================================================

def validate_task3(
    vehicle,
    cfg,
    sample_rows,
    full,
):

    print()
    print("-" * 70)

    print(
        f"Task 3 -- Zone Demand "
        f"({vehicle})"
    )

    print(
        "-" * 70
    )

    model_path = cfg[
        "zone_demand_model"
    ]

    data_path = cfg[
        "zone_hourly_demand"
    ]

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not model_path.exists():

        print(
            "\n  SKIP -- zone demand "
            "model not found:"
        )

        print(
            f"    {model_path}"
        )

        return None

    if not data_path.exists():

        print(
            "\n  SKIP -- zone hourly "
            "demand data not found:"
        )

        print(
            f"    {data_path}"
        )

        return None

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\n  Loading zone model..."
    )

    model = load(
        model_path
    )

    print(
        f"  Model type: "
        f"{type(model).__name__}"
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    feat = build_zone_features(
        data_path
    )

    print(
        f"\n  Rows available: "
        f"{len(feat):,}"
    )

    # --------------------------------------------------------
    # Subsample if necessary
    # --------------------------------------------------------

    if (
        not full
        and len(feat) > sample_rows
    ):

        stride = max(
            1,
            len(feat) // sample_rows,
        )

        feat = (
            feat
            .iloc[::stride]
            .reset_index(drop=True)
        )

        print(
            f"\n  Subsampling every "
            f"{stride}th row."
        )

        print(
            f"  Rows used for CV: "
            f"{len(feat):,}"
        )

        print(
            "  Chronological order "
            "preserved."
        )

        print(
            "  Use --full to disable."
        )

    # --------------------------------------------------------
    # Model features
    # --------------------------------------------------------

    if not hasattr(
        model,
        "feature_names_in_",
    ):

        raise ValueError(
            f"{vehicle} Task 3 model "
            "does not expose feature_names_in_."
        )

    feature_cols = list(
        model.feature_names_in_
    )

    print(
        "\n  Model features:"
    )

    for feature in feature_cols:

        print(
            f"    - {feature}"
        )

    missing = [
        col
        for col in feature_cols
        if col not in feat.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing Task 3 features "
            f"for {vehicle}:\n"
            f"{missing}"
        )

    X = feat[
        feature_cols
    ]

    y = feat[
        "trip_count"
    ]

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    results = run_cv(
        model,
        X,
        y,
    )

    return print_comparison(
        f"Task 3 ({vehicle})",
        results,
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    all_results,
):

    report_path = (
        VALIDATION_DIR
        / "yellow_fhv_demand_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "DEMAND MODEL VALIDATION REPORT\n"
        )

        f.write(
            "YELLOW TAXI + FHV\n"
        )

        f.write(
            "=" * 70
            + "\n"
        )

        f.write(
            f"{N_SPLITS}-fold "
            "TimeSeriesSplit cross-validation\n\n"
        )

        for result in all_results:

            f.write(
                f"{result['label']}\n"
            )

            f.write(
                f"Status: "
                f"{result['status']}\n"
            )

            f.write(
                "-" * 70
                + "\n"
            )

            model = result[
                "model"
            ]

            f.write(
                f"Model MAE: "
                f"{model['mae_mean']:.2f}"
                f" +/- "
                f"{model['mae_std']:.2f}\n"
            )

            f.write(
                f"Model RMSE: "
                f"{model['rmse_mean']:.2f}"
                f" +/- "
                f"{model['rmse_std']:.2f}\n"
            )

            f.write(
                f"Model R²: "
                f"{model['r2_mean']:.3f}"
                f" +/- "
                f"{model['r2_std']:.3f}\n"
            )

            mean = result[
                "mean_baseline"
            ]

            if mean:

                f.write(
                    f"Mean baseline MAE: "
                    f"{mean['mae_mean']:.2f}\n"
                )

                f.write(
                    f"Mean baseline RMSE: "
                    f"{mean['rmse_mean']:.2f}\n"
                )

            persistence = result[
                "persistence"
            ]

            if persistence:

                f.write(
                    f"Persistence "
                    f"(t-1h) MAE: "
                    f"{persistence['mae_mean']:.2f}\n"
                )

                f.write(
                    f"Persistence RMSE: "
                    f"{persistence['rmse_mean']:.2f}\n"
                )

            seasonal = result[
                "seasonal_naive"
            ]

            if seasonal:

                f.write(
                    f"Seasonal-naive "
                    f"(t-1week) MAE: "
                    f"{seasonal['mae_mean']:.2f}\n"
                )

                f.write(
                    f"Seasonal-naive RMSE: "
                    f"{seasonal['rmse_mean']:.2f}\n"
                )

            f.write(
                "\n"
            )

    return report_path


# ============================================================
# SAVE CSV SUMMARY
# ============================================================

def save_summary_csv(
    all_results,
):

    rows = []

    for result in all_results:

        row = {

            "Task":
                result["label"],

            "Status":
                result["status"],

            "Model_MAE":
                result["model"][
                    "mae_mean"
                ],

            "Model_RMSE":
                result["model"][
                    "rmse_mean"
                ],

            "Model_R2":
                result["model"][
                    "r2_mean"
                ],

            "Mean_Baseline_MAE":
                result[
                    "mean_baseline"
                ].get(
                    "mae_mean",
                    np.nan,
                ),

            "Persistence_MAE":
                result[
                    "persistence"
                ].get(
                    "mae_mean",
                    np.nan,
                ),

            "Seasonal_Naive_MAE":
                result[
                    "seasonal_naive"
                ].get(
                    "mae_mean",
                    np.nan,
                ),
        }

        # ----------------------------------------------------
        # Improvement against each baseline
        # ----------------------------------------------------

        model_mae = row[
            "Model_MAE"
        ]

        if pd.notna(
            row["Mean_Baseline_MAE"]
        ):

            row[
                "Improvement_vs_Mean_percent"
            ] = (
                1
                - model_mae
                / row[
                    "Mean_Baseline_MAE"
                ]
            ) * 100

        else:

            row[
                "Improvement_vs_Mean_percent"
            ] = np.nan

        if pd.notna(
            row["Persistence_MAE"]
        ):

            row[
                "Improvement_vs_Persistence_percent"
            ] = (
                1
                - model_mae
                / row[
                    "Persistence_MAE"
                ]
            ) * 100

        else:

            row[
                "Improvement_vs_Persistence_percent"
            ] = np.nan

        if pd.notna(
            row["Seasonal_Naive_MAE"]
        ):

            row[
                "Improvement_vs_Seasonal_Naive_percent"
            ] = (
                1
                - model_mae
                / row[
                    "Seasonal_Naive_MAE"
                ]
            ) * 100

        else:

            row[
                "Improvement_vs_Seasonal_Naive_percent"
            ] = np.nan

        rows.append(
            row
        )

    summary_df = pd.DataFrame(
        rows
    )

    summary_path = (
        VALIDATION_DIR
        / "yellow_fhv_demand_validation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    return summary_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zone-sample-rows",
        type=int,
        default=DEFAULT_ZONE_SAMPLE_ROWS,
        help=(
            "Maximum rows to use for "
            "Task 3 CV. "
            f"Default: "
            f"{DEFAULT_ZONE_SAMPLE_ROWS:,}"
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Disable Task 3 subsampling. "
            "May be slow."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 75)

    print(
        "DEMAND MODEL VALIDATION"
    )

    print(
        "YELLOW TAXI + FHV ONLY"
    )

    print("=" * 75)

    print(
        "\nValidation method:"
    )

    print(
        "  5-fold TimeSeriesSplit"
    )

    print(
        "  Mean baseline"
    )

    print(
        "  Persistence baseline (t-1h)"
    )

    print(
        "  Seasonal-naive baseline (t-1week)"
    )

    print(
        "\nProject root:"
    )

    print(
        f"  {ROOT}"
    )

    all_results = []

    # ========================================================
    # VEHICLES
    # ========================================================

    for vehicle, cfg in CONFIGS.items():

        print()
        print("=" * 75)

        print(
            vehicle
        )

        print(
            "=" * 75
        )

        # ----------------------------------------------------
        # Task 2
        # ----------------------------------------------------

        task2_result = validate_task2(
            vehicle,
            cfg,
        )

        if task2_result:

            all_results.append(
                task2_result
            )

        # ----------------------------------------------------
        # Task 3
        # ----------------------------------------------------

        task3_result = validate_task3(
            vehicle,
            cfg,
            args.zone_sample_rows,
            args.full,
        )

        if task3_result:

            all_results.append(
                task3_result
            )

    # ========================================================
    # NO RESULTS
    # ========================================================

    if not all_results:

        print()
        print("=" * 75)

        print(
            "NO DEMAND MODELS WERE VALIDATED"
        )

        print(
            "=" * 75
        )

        print(
            "\nPlease check that the "
            "Yellow Taxi and FHV model/data "
            "files exist."
        )

        return

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = save_report(
        all_results
    )

    summary_path = save_summary_csv(
        all_results
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)

    print(
        "FINAL DEMAND VALIDATION SUMMARY"
    )

    print("=" * 75)

    summary_rows = []

    for result in all_results:

        model = result[
            "model"
        ]

        mean = result[
            "mean_baseline"
        ]

        persistence = result[
            "persistence"
        ]

        seasonal = result[
            "seasonal_naive"
        ]

        summary_rows.append({

            "Task":
                result["label"],

            "MAE":
                round(
                    model["mae_mean"],
                    2,
                ),

            "RMSE":
                round(
                    model["rmse_mean"],
                    2,
                ),

            "R2":
                round(
                    model["r2_mean"],
                    3,
                ),

            "Mean_MAE":
                round(
                    mean.get(
                        "mae_mean",
                        np.nan,
                    ),
                    2,
                ),

            "Persistence_MAE":
                round(
                    persistence.get(
                        "mae_mean",
                        np.nan,
                    ),
                    2,
                ),

            "Seasonal_Naive_MAE":
                round(
                    seasonal.get(
                        "mae_mean",
                        np.nan,
                    ),
                    2,
                ),

            "Status":
                result["status"],
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print()
    print(
        "Validation outputs:"
    )

    print(
        f"  Report:"
    )

    print(
        f"    {report_path}"
    )

    print(
        f"  Summary:"
    )

    print(
        f"    {summary_path}"
    )

    print()
    print("=" * 75)

    print(
        "DEMAND VALIDATION COMPLETE"
    )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()