"""
Validate Yellow Taxi and FHV Trip Duration Models

Vehicles:
    - Yellow Taxi
    - FHV

Existing models:
    outputs/yellow_duration_model.joblib
    outputs/fhv_duration_model.joblib

Processed data:
    data/processed/yellow_trip_features.parquet
    data/processed/fhv_trip_features.parquet

Validation:
    - 5-fold cross-validation
    - MAE
    - RMSE
    - R²
    - Mean-duration baseline
    - 15 mph physics baseline when trip distance is available

Outputs:
    outputs/validation/yellow_fhv_duration_validation_summary.csv
    outputs/validation/yellow_fhv_duration_validation_results.json
    outputs/validation/yellow_fhv_duration_validation_report.txt

Run:
    python src/validate_duration_models.py
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs"
VALIDATION_DIR = OUTPUT_DIR / "validation"

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CONFIGS = {

    "Yellow Taxi": {
        "model_path":
            OUTPUT_DIR / "yellow_duration_model.joblib",

        "data_path":
            ROOT
            / "data"
            / "processed"
            / "yellow_trip_features.parquet",
    },

    "FHV": {
        "model_path":
            OUTPUT_DIR / "fhv_duration_model.joblib",

        "data_path":
            ROOT
            / "data"
            / "processed"
            / "fhv_duration_features.parquet",
    },
}


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

MAX_VALIDATION_ROWS = 100_000


# ============================================================
# PRINT HELPER
# ============================================================

def print_header(title):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(data_path):

    print("\nLoading data:")
    print(f"  {data_path}")

    if not data_path.exists():

        raise FileNotFoundError(
            f"\nProcessed data file not found:\n{data_path}"
        )

    df = pd.read_parquet(data_path)

    print(
        f"  Rows loaded: {len(df):,}"
    )

    print(
        f"  Columns: {len(df.columns)}"
    )

    if (
        MAX_VALIDATION_ROWS is not None
        and len(df) > MAX_VALIDATION_ROWS
    ):

        print(
            f"  Sampling "
            f"{MAX_VALIDATION_ROWS:,} rows "
            f"for validation..."
        )

        df = df.sample(
            n=MAX_VALIDATION_ROWS,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

        print(
            f"  Rows used: {len(df):,}"
        )

    return df


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features(model):

    """
    Get the feature names expected by the saved model.

    Supports:
        - GradientBoostingRegressor
        - LightGBM
        - sklearn pipelines
    """

    # sklearn models
    if hasattr(model, "feature_names_in_"):

        return list(
            model.feature_names_in_
        )

    # LightGBM
    if hasattr(model, "feature_name_"):

        features = list(
            model.feature_name_
        )

        if features:
            return features

    # Pipeline
    if hasattr(model, "named_steps"):

        for _, step in model.named_steps.items():

            if hasattr(
                step,
                "feature_names_in_"
            ):

                return list(
                    step.feature_names_in_
                )

            if hasattr(
                step,
                "feature_name_"
            ):

                features = list(
                    step.feature_name_
                )

                if features:
                    return features

    raise ValueError(
        "Could not determine the features "
        "used by the trained model."
    )


# ============================================================
# FIND TARGET
# ============================================================

def find_target_column(df):

    """
    Find duration target.

    The current processed Yellow/FHV datasets use:
        trip_duration_s

    Other common names are also supported.
    """

    possible_targets = [

        "trip_duration_s",

        "trip_duration_seconds",

        "duration_seconds",

        "trip_duration",

        "duration",

        "target",

        "y",
    ]

    for column in possible_targets:

        if column in df.columns:

            return column

    raise ValueError(
        "\nCould not find the duration target column.\n\n"
        "Available columns:\n"
        f"{list(df.columns)}"
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(X):

    """
    Convert model input into a numeric DataFrame.

    The saved models were trained using numeric features.
    """

    X = X.copy()

    # --------------------------------------------------------
    # Datetime columns
    # --------------------------------------------------------

    for column in X.columns:

        if pd.api.types.is_datetime64_any_dtype(
            X[column]
        ):

            X[column] = (
                X[column]
                .astype("int64")
                / 1e9
            )

    # --------------------------------------------------------
    # Object/category columns
    # --------------------------------------------------------

    for column in X.columns:

        if (
            X[column].dtype == "object"
            or str(
                X[column].dtype
            ).startswith("category")
        ):

            X[column] = (
                X[column]
                .astype("category")
                .cat.codes
            )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Infinity
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    for column in X.columns:

        if X[column].isna().any():

            median_value = X[column].median()

            if pd.isna(median_value):

                median_value = 0

            X[column] = (
                X[column]
                .fillna(median_value)
            )

    return X


# ============================================================
# CHECK FEATURES
# ============================================================

def check_features(
    df,
    model_features,
    vehicle_name,
):

    missing = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing:

        print(
            "\nERROR: Missing model features:"
        )

        for feature in missing:

            print(
                f"  - {feature}"
            )

        raise ValueError(
            f"\n{vehicle_name}: "
            f"{len(missing)} model feature(s) "
            "are missing from the processed dataset."
        )

    print("\nFeature check:")
    print(
        "  All required model features are available."
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    return {

        "MAE_seconds": float(mae),

        "RMSE_seconds": float(rmse),

        "R2": float(r2),
    }


# ============================================================
# PHYSICS BASELINE
# ============================================================

def calculate_physics_baseline(
    df,
    target_column,
):

    """
    15 mph baseline:

        duration_seconds =
            distance_miles / 15 * 3600

    Only available when trip distance exists.
    """

    possible_distance_columns = [

        "trip_distance",

        "distance_miles",

        "trip_distance_miles",

        "distance",
    ]

    distance_column = None

    for column in possible_distance_columns:

        if column in df.columns:

            distance_column = column

            break

    if distance_column is None:

        return None, None

    distance = pd.to_numeric(
        df[distance_column],
        errors="coerce",
    )

    actual = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    mask = (

        distance.notna()

        & actual.notna()

        & np.isfinite(distance)

        & np.isfinite(actual)

        & (distance > 0)

        & (actual > 0)
    )

    if mask.sum() == 0:

        return None, None

    physics_prediction = (

        distance[mask]
        / 15.0
        * 3600.0
    )

    physics_actual = actual[mask]

    metrics = calculate_metrics(
        physics_actual,
        physics_prediction,
    )

    return metrics, distance_column


# ============================================================
# RUN VALIDATION
# ============================================================

def run_validation(
    vehicle_name,
    config,
):

    print_header(vehicle_name)

    model_path = config["model_path"]

    data_path = config["data_path"]

    # ========================================================
    # MODEL
    # ========================================================

    print("\nModel:")
    print(f"  {model_path}")

    if not model_path.exists():

        print(
            "\nSKIP -- model not found:"
        )

        print(
            f"  {model_path}"
        )

        return None

    print(
        "  Loading model..."
    )

    model = joblib.load(
        model_path
    )

    print(
        f"  Model type: "
        f"{type(model).__name__}"
    )

    # ========================================================
    # MODEL FEATURES
    # ========================================================

    model_features = get_model_features(
        model
    )

    print(
        f"  Model features: "
        f"{len(model_features)}"
    )

    print(
        "\n  Expected features:"
    )

    for feature in model_features:

        print(
            f"    - {feature}"
        )

    # ========================================================
    # DATA
    # ========================================================

    df = load_data(
        data_path
    )

    # ========================================================
    # TARGET
    # ========================================================

    target_column = find_target_column(
        df
    )

    print(
        f"\nTarget column: "
        f"{target_column}"
    )

    # ========================================================
    # FEATURE CHECK
    # ========================================================

    check_features(
        df,
        model_features,
        vehicle_name,
    )

    # ========================================================
    # TARGET CLEANING
    # ========================================================

    y = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    valid_target = (

        y.notna()

        & np.isfinite(y)

        & (y > 0)
    )

    df_valid = (
        df.loc[valid_target]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid_target]
        .reset_index(drop=True)
    )

    print(
        f"\nValid duration rows: "
        f"{len(y):,}"
    )

    # ========================================================
    # FEATURES
    # ========================================================

    X = df_valid[
        model_features
    ].copy()

    X = prepare_features(
        X
    )

    # ========================================================
    # FINAL SHAPE CHECK
    # ========================================================

    print(
        "\nValidation feature matrix:"
    )

    print(
        f"  Rows    : {len(X):,}"
    )

    print(
        f"  Columns : {len(X.columns)}"
    )

    # ========================================================
    # 5-FOLD CV
    # ========================================================

    print_header(
        f"{vehicle_name} -- "
        f"5-Fold Cross Validation"
    )

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold_number, (
        train_idx,
        test_idx,
    ) in enumerate(
        kfold.split(X),
        start=1,
    ):

        print(
            f"\nFold "
            f"{fold_number}/{N_SPLITS}"
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
        # Clone model
        # ----------------------------------------------------

        fold_model = clone(
            model
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        fold_model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = (
            fold_model.predict(
                X_test
            )
        )

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        print(
            f"  MAE  : "
            f"{metrics['MAE_seconds']:.2f} sec"
        )

        print(
            f"  RMSE : "
            f"{metrics['RMSE_seconds']:.2f} sec"
        )

        print(
            f"  R²   : "
            f"{metrics['R2']:.4f}"
        )

        fold_results.append(
            metrics
        )

    # ========================================================
    # CV SUMMARY
    # ========================================================

    cv_mae_values = [
        x["MAE_seconds"]
        for x in fold_results
    ]

    cv_rmse_values = [
        x["RMSE_seconds"]
        for x in fold_results
    ]

    cv_r2_values = [
        x["R2"]
        for x in fold_results
    ]

    cv_mae = np.mean(
        cv_mae_values
    )

    cv_rmse = np.mean(
        cv_rmse_values
    )

    cv_r2 = np.mean(
        cv_r2_values
    )

    cv_mae_std = np.std(
        cv_mae_values
    )

    cv_rmse_std = np.std(
        cv_rmse_values
    )

    cv_r2_std = np.std(
        cv_r2_values
    )

    # ========================================================
    # MEAN BASELINE
    # ========================================================

    print_header(
        f"{vehicle_name} -- "
        f"Baseline Comparison"
    )

    mean_duration = y.mean()

    mean_prediction = np.full(
        len(y),
        mean_duration,
    )

    baseline_metrics = calculate_metrics(
        y,
        mean_prediction,
    )

    print(
        "\nMean-duration baseline:"
    )

    print(
        f"  Mean duration : "
        f"{mean_duration:.2f} sec"
    )

    print(
        f"  MAE           : "
        f"{baseline_metrics['MAE_seconds']:.2f} sec"
    )

    print(
        f"  RMSE          : "
        f"{baseline_metrics['RMSE_seconds']:.2f} sec"
    )

    print(
        f"  R²            : "
        f"{baseline_metrics['R2']:.4f}"
    )

    # ========================================================
    # PHYSICS BASELINE
    # ========================================================

    # FHV trip_distance is estimated from pickup/drop-off taxi-zone
    # centroids, so this baseline is a reference calculation rather
    # than a comparison against recorded FHV road distance.
    if vehicle_name == "FHV":
        print(
            "\nNOTE: FHV trip_distance is estimated from taxi-zone "
            "centroids; the 15 mph baseline is therefore only a "
            "reference baseline."
        )

    physics_metrics, distance_column = (
        calculate_physics_baseline(
            df_valid,
            target_column,
        )
    )

    if physics_metrics is not None:

        print(
            "\n15 mph physics baseline:"
        )

        print(
            f"  Distance column: "
            f"{distance_column}"
        )

        print(
            f"  MAE  : "
            f"{physics_metrics['MAE_seconds']:.2f} sec"
        )

        print(
            f"  RMSE : "
            f"{physics_metrics['RMSE_seconds']:.2f} sec"
        )

        print(
            f"  R²   : "
            f"{physics_metrics['R2']:.4f}"
        )

    else:

        print(
            "\n15 mph physics baseline:"
        )

        print(
            "  Not available -- "
            "no usable distance column."
        )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    mae_improvement = (

        (
            baseline_metrics[
                "MAE_seconds"
            ]

            - cv_mae
        )

        / baseline_metrics[
            "MAE_seconds"
        ]

        * 100
    )

    rmse_improvement = (

        (
            baseline_metrics[
                "RMSE_seconds"
            ]

            - cv_rmse
        )

        / baseline_metrics[
            "RMSE_seconds"
        ]

        * 100
    )

    # ========================================================
    # VALIDATION DECISION
    # ========================================================

    model_better_than_baseline = (

        cv_mae
        < baseline_metrics[
            "MAE_seconds"
        ]
    )

    print_header(
        f"{vehicle_name} -- "
        f"Validation Result"
    )

    print(
        f"\nCross-validation MAE  : "
        f"{cv_mae:.2f} sec"
    )

    print(
        f"Mean baseline MAE     : "
        f"{baseline_metrics['MAE_seconds']:.2f} sec"
    )

    print(
        f"\nMAE improvement       : "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"RMSE improvement      : "
        f"{rmse_improvement:.2f}%"
    )

    print(
        f"\nAverage R²            : "
        f"{cv_r2:.4f}"
    )

    if model_better_than_baseline:

        validation_status = "PASS"

        print(
            "\nRESULT: PASS"
        )

        print(
            "The duration model performs "
            "better than the mean-duration baseline."
        )

    else:

        validation_status = "CHECK"

        print(
            "\nRESULT: CHECK"
        )

        print(
            "The duration model does not outperform "
            "the mean-duration baseline."
        )

    # ========================================================
    # RESULT DICTIONARY
    # ========================================================

    return {

        "vehicle":
            vehicle_name,

        "model_path":
            str(
                model_path.relative_to(
                    ROOT
                )
            ),

        "data_path":
            str(
                data_path.relative_to(
                    ROOT
                )
            ),

        "model_type":
            type(model).__name__,

        "target_column":
            target_column,

        "rows_validated":
            int(len(y)),

        "model_features":
            int(len(model_features)),

        "feature_names":
            model_features,

        "cv_folds":
            N_SPLITS,

        "cv_mae_seconds":
            float(cv_mae),

        "cv_mae_minutes":
            float(cv_mae / 60.0),

        "cv_mae_std_seconds":
            float(cv_mae_std),

        "cv_rmse_seconds":
            float(cv_rmse),

        "cv_rmse_minutes":
            float(cv_rmse / 60.0),

        "cv_rmse_std_seconds":
            float(cv_rmse_std),

        "cv_r2":
            float(cv_r2),

        "cv_r2_std":
            float(cv_r2_std),

        "mean_baseline_mae_seconds":
            float(
                baseline_metrics[
                    "MAE_seconds"
                ]
            ),

        "mean_baseline_mae_minutes":
            float(
                baseline_metrics[
                    "MAE_seconds"
                ] / 60.0
            ),

        "mean_baseline_rmse_seconds":
            float(
                baseline_metrics[
                    "RMSE_seconds"
                ]
            ),

        "mean_baseline_rmse_minutes":
            float(
                baseline_metrics[
                    "RMSE_seconds"
                ] / 60.0
            ),

        "mean_baseline_r2":
            float(
                baseline_metrics[
                    "R2"
                ]
            ),

        "mae_improvement_vs_mean_percent":
            float(
                mae_improvement
            ),

        "rmse_improvement_vs_mean_percent":
            float(
                rmse_improvement
            ),

        "physics_baseline_available":
            physics_metrics is not None,

        "physics_baseline_mae_seconds":
            (
                float(
                    physics_metrics[
                        "MAE_seconds"
                    ]
                )
                if physics_metrics is not None
                else None
            ),

        "physics_baseline_rmse_seconds":
            (
                float(
                    physics_metrics[
                        "RMSE_seconds"
                    ]
                )
                if physics_metrics is not None
                else None
            ),

        "physics_baseline_r2":
            (
                float(
                    physics_metrics[
                        "R2"
                    ]
                )
                if physics_metrics is not None
                else None
            ),

        "validation_status":
            validation_status,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "TRIP DURATION MODEL VALIDATION"
    )
    print(
        "YELLOW TAXI + FHV ONLY"
    )
    print("=" * 75)

    print(
        "\nProject root:"
    )

    print(
        f"  {ROOT}"
    )

    print(
        "\nModel locations:"
    )

    print(
        f"  Yellow: "
        f"{CONFIGS['Yellow Taxi']['model_path']}"
    )

    print(
        f"  FHV   : "
        f"{CONFIGS['FHV']['model_path']}"
    )

    results = []

    # ========================================================
    # YELLOW
    # ========================================================

    yellow_result = run_validation(
        "Yellow Taxi",
        CONFIGS["Yellow Taxi"],
    )

    if yellow_result is not None:

        results.append(
            yellow_result
        )

    # ========================================================
    # FHV
    # ========================================================

    fhv_result = run_validation(
        "FHV",
        CONFIGS["FHV"],
    )

    if fhv_result is not None:

        results.append(
            fhv_result
        )

    # ========================================================
    # NO RESULTS
    # ========================================================

    if not results:

        print()
        print("=" * 75)
        print(
            "NO MODELS WERE VALIDATED"
        )
        print("=" * 75)

        print(
            "\nExpected model files:"
        )

        print(
            f"  {OUTPUT_DIR / 'yellow_duration_model.joblib'}"
        )

        print(
            f"  {OUTPUT_DIR / 'fhv_duration_model.joblib'}"
        )

        return

    # ========================================================
    # SUMMARY
    # ========================================================

    print_header(
        "FINAL DURATION MODEL "
        "VALIDATION SUMMARY"
    )

    summary_rows = []

    for result in results:

        summary_rows.append({

            "Vehicle":
                result["vehicle"],

            "Model":
                result["model_type"],

            "Rows":
                result["rows_validated"],

            "MAE_seconds":
                round(
                    result[
                        "cv_mae_seconds"
                    ],
                    2,
                ),

            "MAE_minutes":
                round(
                    result[
                        "cv_mae_minutes"
                    ],
                    3,
                ),

            "RMSE_seconds":
                round(
                    result[
                        "cv_rmse_seconds"
                    ],
                    2,
                ),

            "RMSE_minutes":
                round(
                    result[
                        "cv_rmse_minutes"
                    ],
                    3,
                ),

            "R2":
                round(
                    result["cv_r2"],
                    4,
                ),

            "Baseline_MAE_seconds":
                round(
                    result[
                        "mean_baseline_mae_seconds"
                    ],
                    2,
                ),

            "MAE_improvement_percent":
                round(
                    result[
                        "mae_improvement_vs_mean_percent"
                    ],
                    2,
                ),

            "Status":
                result[
                    "validation_status"
                ],
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
    # SAVE CSV
    # ========================================================

    summary_csv = (
        VALIDATION_DIR
        / "yellow_fhv_duration_validation_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    print(
        "\nSummary CSV saved:"
    )

    print(
        f"  {summary_csv}"
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        VALIDATION_DIR
        / "yellow_fhv_duration_validation_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "JSON results saved:"
    )

    print(
        f"  {json_path}"
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        VALIDATION_DIR
        / "yellow_fhv_duration_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TRIP DURATION MODEL "
            "VALIDATION REPORT\n"
        )

        f.write(
            "YELLOW TAXI + FHV\n"
        )

        f.write(
            "=" * 75 + "\n\n"
        )

        for result in results:

            f.write(
                f"Vehicle: "
                f"{result['vehicle']}\n"
            )

            f.write(
                "-" * 75 + "\n"
            )

            f.write(
                f"Model: "
                f"{result['model_path']}\n"
            )

            f.write(
                f"Model type: "
                f"{result['model_type']}\n"
            )

            f.write(
                f"Data: "
                f"{result['data_path']}\n"
            )

            f.write(
                f"Target: "
                f"{result['target_column']}\n"
            )

            f.write(
                f"Rows validated: "
                f"{result['rows_validated']:,}\n"
            )

            f.write(
                f"Features: "
                f"{result['model_features']}\n"
            )

            f.write(
                "\nFeature names:\n"
            )

            for feature in result[
                "feature_names"
            ]:

                f.write(
                    f"  - {feature}\n"
                )

            f.write(
                "\n"
            )

            f.write(
                f"CV MAE: "
                f"{result['cv_mae_seconds']:.2f} seconds "
                f"({result['cv_mae_minutes']:.3f} minutes)\n"
            )

            f.write(
                f"CV MAE std: "
                f"{result['cv_mae_std_seconds']:.2f} seconds\n"
            )

            f.write(
                f"CV RMSE: "
                f"{result['cv_rmse_seconds']:.2f} seconds "
                f"({result['cv_rmse_minutes']:.3f} minutes)\n"
            )

            f.write(
                f"CV RMSE std: "
                f"{result['cv_rmse_std_seconds']:.2f} seconds\n"
            )

            f.write(
                f"CV R²: "
                f"{result['cv_r2']:.4f}\n"
            )

            f.write(
                f"CV R² std: "
                f"{result['cv_r2_std']:.4f}\n\n"
            )

            f.write(
                "Mean-duration baseline:\n"
            )

            f.write(
                f"  MAE: "
                f"{result['mean_baseline_mae_seconds']:.2f} seconds\n"
            )

            f.write(
                f"  RMSE: "
                f"{result['mean_baseline_rmse_seconds']:.2f} seconds\n"
            )

            f.write(
                f"  R²: "
                f"{result['mean_baseline_r2']:.4f}\n"
            )

            f.write(
                f"\nMAE improvement vs baseline: "
                f"{result['mae_improvement_vs_mean_percent']:.2f}%\n"
            )

            f.write(
                f"RMSE improvement vs baseline: "
                f"{result['rmse_improvement_vs_mean_percent']:.2f}%\n"
            )

            if result[
                "physics_baseline_available"
            ]:

                f.write(
                    "\n15 mph physics baseline:\n"
                )

                f.write(
                    f"  MAE: "
                    f"{result['physics_baseline_mae_seconds']:.2f} seconds\n"
                )

                f.write(
                    f"  RMSE: "
                    f"{result['physics_baseline_rmse_seconds']:.2f} seconds\n"
                )

                f.write(
                    f"  R²: "
                    f"{result['physics_baseline_r2']:.4f}\n"
                )

            else:

                f.write(
                    "\n15 mph physics baseline: "
                    "Not available\n"
                )

            f.write(
                f"\nValidation status: "
                f"{result['validation_status']}\n"
            )

            f.write(
                "\n\n"
            )

    print(
        "Text report saved:"
    )

    print(
        f"  {report_path}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 75)
    print(
        "VALIDATION COMPLETE"
    )
    print("=" * 75)

    print(
        "\nValidated vehicles:"
    )

    for result in results:

        print(
            f"  {result['vehicle']}: "
            f"{result['validation_status']}"
        )

    print(
        "\nValidation uses each saved model's own feature_names_in_ "
        "so Yellow and the new 8-feature FHV model are validated "
        "against the exact inputs they were trained on."
    )

    print(
        "\nOutput directory:"
    )

    print(
        f"  {VALIDATION_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()