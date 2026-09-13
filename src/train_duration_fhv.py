"""
Train FHV trip-duration prediction model using GradientBoostingRegressor.

Input:
    data/processed/fhv_duration_features.parquet

Outputs:
    outputs/fhv_duration_model.joblib
    outputs/fhv_duration_model_metrics.txt
    outputs/fhv_duration_predictions.parquet

Algorithm:
    sklearn.ensemble.GradientBoostingRegressor

Target:
    trip_duration_s

Important:
    The FHV feature builder creates a simple model input. Because standard
    FHV records do not contain recorded trip distance, trip_distance is
    estimated from pickup/drop-off taxi-zone centroids.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "fhv_duration_features.parquet"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "fhv_duration_model.joblib"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "fhv_duration_model_metrics.txt"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "fhv_duration_predictions.parquet"
)


# ============================================================================
# SETTINGS
# ============================================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# GradientBoostingRegressor is computationally expensive on millions of rows.
#
# Your old FHV model also used a 500,000-row training sample.
#
# We therefore use 500,000 rows here so the new model remains practical
# while still using a large representative sample.
# ---------------------------------------------------------------------------

MAX_ROWS = 500_000


# ============================================================================
# FEATURES
# ============================================================================

FEATURE_COLS = [
    "trip_distance",
    "PUlocationID",
    "DOlocationID",
    "SR_Flag",
    "pickup_hour",
    "pickup_dow",
    "is_weekend",
    "pickup_month",
]


# ============================================================================
# TARGET
# ============================================================================

TARGET_COL = "trip_duration_s"


# ============================================================================
# CREATE MODEL
# ============================================================================

def create_model():
    """
    Create the GradientBoostingRegressor.

    This is the same algorithm family used by the existing
    FHV duration approach, but trained with the simple distance and temporal features.
    """

    model = GradientBoostingRegressor(

        # Number of boosting trees
        n_estimators=200,

        # Maximum depth of each individual tree
        max_depth=4,

        # Contribution of each tree
        learning_rate=0.08,

        # Use 80% of rows for each boosting iteration
        subsample=0.8,

        # Minimum samples required to split an internal node
        min_samples_split=2,

        # Minimum samples required at a leaf
        min_samples_leaf=1,

        # Squared-error regression loss
        loss="squared_error",

        # Reproducibility
        random_state=RANDOM_STATE,

        # Allow sklearn to use the default behavior
        verbose=1,
    )

    return model


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():

    print()
    print("=" * 72)
    print("LOADING FHV DURATION FEATURES")
    print("=" * 72)

    print()

    print(
        "Loading:"
    )

    print(
        INPUT_FILE
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"""
FHV duration feature file was not found:

{INPUT_FILE}

Run this first:

python src/build_fhv_duration_features.py --months 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12
"""
        )

    df = pd.read_parquet(
        INPUT_FILE
    )

    print()

    print(
        f"Loaded {len(df):,} rows"
    )

    return df


# ============================================================================
# VALIDATE COLUMNS
# ============================================================================

def validate_columns(
    df
):

    print()
    print("=" * 72)
    print("VALIDATING DATASET COLUMNS")
    print("=" * 72)

    required_columns = (
        FEATURE_COLS
        +
        [TARGET_COL]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print()

        print(
            "Missing required columns:"
        )

        for column in missing_columns:

            print(
                f"  - {column}"
            )

        print()

        print(
            "Available columns:"
        )

        for column in df.columns:

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Required feature columns are missing."
        )

    print()

    print(
        "All required columns are present."
    )

    print()

    print(
        "Features:"
    )

    for feature in FEATURE_COLS:

        print(
            f"  - {feature}"
        )

    print()

    print(
        f"Target: {TARGET_COL}"
    )


# ============================================================================
# PREPARE DATA
# ============================================================================

def prepare_data(
    df
):

    print()
    print("=" * 72)
    print("PREPARING MODEL DATA")
    print("=" * 72)

    # ------------------------------------------------------------------------
    # Keep only model columns and datetime
    # ------------------------------------------------------------------------

    columns_to_keep = (
        FEATURE_COLS
        +
        [
            TARGET_COL,
            "pickup_datetime",
        ]
    )

    df = df[
        columns_to_keep
    ].copy()

    # ------------------------------------------------------------------------
    # Convert all features to numeric
    #
    # GradientBoostingRegressor requires numerical input.
    # ------------------------------------------------------------------------

    print()

    print(
        "Converting features to numeric..."
    )

    for column in FEATURE_COLS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ------------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------------

    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce"
    )

    # ------------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=FEATURE_COLS + [TARGET_COL]
    ).copy()

    print()

    print(
        "Rows after removing missing values: "
        f"{len(df):,} "
        f"({before - len(df):,} removed)"
    )

    # ------------------------------------------------------------------------
    # Duration sanity check
    # ------------------------------------------------------------------------

    before = len(df)

    df = df[
        (
            df[TARGET_COL]
            >= 60
        )
        &
        (
            df[TARGET_COL]
            <= 10800
        )
    ].copy()

    print(
        "Rows after duration sanity check: "
        f"{len(df):,} "
        f"({before - len(df):,} removed)"
    )

    # ------------------------------------------------------------------------
    # Trip-distance sanity check
    # ------------------------------------------------------------------------

    before = len(df)

    df = df[
        (df["trip_distance"] >= 0)
        &
        (df["trip_distance"] <= 100)
    ].copy()

    print(
        "Rows after trip-distance sanity check: "
        f"{len(df):,} "
        f"({before - len(df):,} removed)"
    )

    # ------------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------------

    if (
        MAX_ROWS is not None
        and len(df) > MAX_ROWS
    ):

        print()

        print(
            f"Dataset is larger than {MAX_ROWS:,} rows."
        )

        print(
            f"Randomly sampling {MAX_ROWS:,} rows..."
        )

        df = df.sample(
            n=MAX_ROWS,
            random_state=RANDOM_STATE
        ).copy()

    # ------------------------------------------------------------------------
    # Reset index
    # ------------------------------------------------------------------------

    df = (
        df
        .reset_index(drop=True)
    )

    print()

    print(
        f"Final rows used for modeling: "
        f"{len(df):,}"
    )

    return df


# ============================================================================
# TRAIN / TEST SPLIT
# ============================================================================

def split_data(
    df
):

    print()
    print("=" * 72)
    print("CREATING TRAIN / TEST SPLIT")
    print("=" * 72)

    X = df[
        FEATURE_COLS
    ].copy()

    y = df[
        TARGET_COL
    ].copy()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    print()

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows:  {len(X_test):,}"
    )

    print()

    print(
        f"Training percentage: "
        f"{len(X_train) / len(X) * 100:.1f}%"
    )

    print(
        f"Testing percentage: "
        f"{len(X_test) / len(X) * 100:.1f}%"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================================
# TRAIN MODEL
# ============================================================================

def train_model(
    X_train,
    y_train
):

    print()
    print("=" * 72)
    print("TRAINING GRADIENT BOOSTING REGRESSOR")
    print("=" * 72)

    print()

    print(
        "Algorithm:"
    )

    print(
        "GradientBoostingRegressor"
    )

    print()

    print(
        "Model parameters:"
    )

    print(
        "  n_estimators  = 200"
    )

    print(
        "  max_depth     = 4"
    )

    print(
        "  learning_rate = 0.08"
    )

    print(
        "  subsample     = 0.80"
    )

    print()

    print(
        f"Training rows: {len(X_train):,}"
    )

    print()

    model = create_model()

    start_time = time.time()

    print(
        "Starting training..."
    )

    print()

    model.fit(
        X_train,
        y_train
    )

    elapsed_seconds = (
        time.time()
        -
        start_time
    )

    print()

    print(
        "Training completed."
    )

    print(
        f"Training time: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )

    return model


# ============================================================================
# EVALUATE
# ============================================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print()
    print("=" * 72)
    print("EVALUATING FHV DURATION MODEL")
    print("=" * 72)

    print()

    print(
        "Generating predictions..."
    )

    predictions_seconds = (
        model.predict(
            X_test
        )
    )

    # ------------------------------------------------------------------------
    # Prevent negative duration predictions
    # ------------------------------------------------------------------------

    predictions_seconds = np.maximum(
        predictions_seconds,
        0
    )

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    mae_seconds = mean_absolute_error(
        y_test,
        predictions_seconds
    )

    rmse_seconds = np.sqrt(
        mean_squared_error(
            y_test,
            predictions_seconds
        )
    )

    r2 = r2_score(
        y_test,
        predictions_seconds
    )

    # ------------------------------------------------------------------------
    # Convert to minutes
    # ------------------------------------------------------------------------

    mae_minutes = (
        mae_seconds
        /
        60.0
    )

    rmse_minutes = (
        rmse_seconds
        /
        60.0
    )

    actual_minutes = (
        y_test.to_numpy()
        /
        60.0
    )

    predicted_minutes = (
        predictions_seconds
        /
        60.0
    )

    # ------------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------------

    print()

    print(
        "--- TEST SET PERFORMANCE ---"
    )

    print()

    print(
        f"MAE:  {mae_minutes:.3f} minutes"
    )

    print(
        f"RMSE: {rmse_minutes:.3f} minutes"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    print()

    print(
        "--- DURATION STATISTICS ---"
    )

    print()

    print(
        f"Actual mean duration: "
        f"{actual_minutes.mean():.2f} minutes"
    )

    print(
        f"Predicted mean duration: "
        f"{predicted_minutes.mean():.2f} minutes"
    )

    print()

    print(
        f"Actual median duration: "
        f"{np.median(actual_minutes):.2f} minutes"
    )

    print(
        f"Predicted median duration: "
        f"{np.median(predicted_minutes):.2f} minutes"
    )

    metrics = {

        "mae_seconds":
            float(mae_seconds),

        "mae_minutes":
            float(mae_minutes),

        "rmse_seconds":
            float(rmse_seconds),

        "rmse_minutes":
            float(rmse_minutes),

        "r2":
            float(r2),

        "actual_mean_minutes":
            float(actual_minutes.mean()),

        "predicted_mean_minutes":
            float(predicted_minutes.mean()),

        "actual_median_minutes":
            float(np.median(actual_minutes)),

        "predicted_median_minutes":
            float(np.median(predicted_minutes)),
    }

    return (
        metrics,
        predictions_seconds
    )


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def show_feature_importance(
    model
):

    print()
    print("=" * 72)
    print("FEATURE IMPORTANCE")
    print("=" * 72)

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,

            "importance":
                model.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print()

    print(
        importance_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Top 10 features:"
    )

    print()

    for index, row in (
        importance_df
        .head(10)
        .iterrows()
    ):

        print(
            f"{index + 1:2d}. "
            f"{row['feature']}: "
            f"{row['importance']:.4f}"
        )

    return importance_df


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(
    model
):

    print()
    print("=" * 72)
    print("SAVING MODEL")
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print()

    print(
        "Model saved:"
    )

    print(
        MODEL_FILE
    )

    print()

    size_mb = (
        MODEL_FILE.stat().st_size
        /
        1024
        /
        1024
    )

    print(
        f"Model size: {size_mb:.2f} MB"
    )


# ============================================================================
# SAVE METRICS
# ============================================================================

def save_metrics(
    metrics,
    rows_used
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FHV TRIP DURATION MODEL METRICS\n"
        )

        file.write(
            "=" * 60
            +
            "\n\n"
        )

        file.write(
            "Algorithm: GradientBoostingRegressor\n"
        )

        file.write(
            f"Input: {INPUT_FILE}\n"
        )

        file.write(
            f"Rows used: {rows_used:,}\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "MODEL PARAMETERS\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        file.write(
            "n_estimators: 200\n"
        )

        file.write(
            "max_depth: 4\n"
        )

        file.write(
            "learning_rate: 0.08\n"
        )

        file.write(
            "subsample: 0.80\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "TEST SET PERFORMANCE\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        file.write(
            f"MAE: "
            f"{metrics['mae_minutes']:.4f} minutes\n"
        )

        file.write(
            f"RMSE: "
            f"{metrics['rmse_minutes']:.4f} minutes\n"
        )

        file.write(
            f"R2: "
            f"{metrics['r2']:.6f}\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "DURATION STATISTICS\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        file.write(
            f"Actual mean: "
            f"{metrics['actual_mean_minutes']:.4f} minutes\n"
        )

        file.write(
            f"Predicted mean: "
            f"{metrics['predicted_mean_minutes']:.4f} minutes\n"
        )

        file.write(
            f"Actual median: "
            f"{metrics['actual_median_minutes']:.4f} minutes\n"
        )

        file.write(
            f"Predicted median: "
            f"{metrics['predicted_median_minutes']:.4f} minutes\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "FEATURES\n"
        )

        file.write(
            "-" * 60
            +
            "\n"
        )

        for feature in FEATURE_COLS:

            file.write(
                f"{feature}\n"
            )

    print()

    print(
        "Metrics saved:"
    )

    print(
        METRICS_FILE
    )


# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

def save_predictions(
    X_test,
    y_test,
    predictions_seconds
):

    print()
    print("=" * 72)
    print("SAVING TEST PREDICTIONS")
    print("=" * 72)

    predictions_df = X_test.copy()

    # ------------------------------------------------------------------------
    # Actual
    # ------------------------------------------------------------------------

    predictions_df[
        "actual_duration_s"
    ] = y_test.to_numpy()

    # ------------------------------------------------------------------------
    # Predicted
    # ------------------------------------------------------------------------

    predictions_df[
        "predicted_duration_s"
    ] = predictions_seconds

    # ------------------------------------------------------------------------
    # Minutes
    # ------------------------------------------------------------------------

    predictions_df[
        "actual_duration_minutes"
    ] = (
        predictions_df[
            "actual_duration_s"
        ]
        /
        60.0
    )

    predictions_df[
        "predicted_duration_minutes"
    ] = (
        predictions_df[
            "predicted_duration_s"
        ]
        /
        60.0
    )

    # ------------------------------------------------------------------------
    # Absolute error
    # ------------------------------------------------------------------------

    predictions_df[
        "absolute_error_minutes"
    ] = (
        predictions_df[
            "actual_duration_minutes"
        ]
        -
        predictions_df[
            "predicted_duration_minutes"
        ]
    ).abs()

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_df.to_parquet(
        PREDICTIONS_FILE,
        index=False,
        compression="zstd"
    )

    print()

    print(
        "Predictions saved:"
    )

    print(
        PREDICTIONS_FILE
    )

    print()

    size_mb = (
        PREDICTIONS_FILE.stat().st_size
        /
        1024
        /
        1024
    )

    print(
        f"File size: {size_mb:.2f} MB"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=" * 72
    )

    print(
        "IMPROVED FHV TRIP DURATION MODEL"
    )

    print(
        "=" * 72
    )

    print()

    print(
        "Algorithm:"
    )

    print(
        "GradientBoostingRegressor"
    )

    print()

    # =========================================================================
    # LOAD
    # =========================================================================

    df = load_data()

    # =========================================================================
    # VALIDATE
    # =========================================================================

    validate_columns(
        df
    )

    # =========================================================================
    # PREPARE
    # =========================================================================

    df = prepare_data(
        df
    )

    # =========================================================================
    # SPLIT
    # =========================================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        df
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    model = train_model(
        X_train,
        y_train
    )

    # =========================================================================
    # EVALUATE
    # =========================================================================

    (
        metrics,
        predictions_seconds
    ) = evaluate_model(
        model,
        X_test,
        y_test
    )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    importance_df = (
        show_feature_importance(
            model
        )
    )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    save_model(
        model
    )

    # =========================================================================
    # SAVE METRICS
    # =========================================================================

    save_metrics(
        metrics,
        len(df)
    )

    # =========================================================================
    # SAVE PREDICTIONS
    # =========================================================================

    save_predictions(
        X_test,
        y_test,
        predictions_seconds
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 72)
    print("FHV DURATION MODEL TRAINING COMPLETE")
    print("=" * 72)

    print()

    print(
        "Algorithm:"
    )

    print(
        "GradientBoostingRegressor"
    )

    print()

    print(
        "FINAL TEST PERFORMANCE"
    )

    print(
        "-" * 72
    )

    print(
        f"MAE:  "
        f"{metrics['mae_minutes']:.3f} minutes"
    )

    print(
        f"RMSE: "
        f"{metrics['rmse_minutes']:.3f} minutes"
    )

    print(
        f"R²:   "
        f"{metrics['r2']:.4f}"
    )

    print()

    print(
        "FILES CREATED"
    )

    print(
        "-" * 72
    )

    print(
        MODEL_FILE
    )

    print(
        METRICS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    print()

    print(
        "FHV GradientBoostingRegressor model is ready."
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()




























"""
Task 1 (FHV) — Trip Duration Prediction.

Predicts FHV trip duration using pickup/dropoff zones,
shared-ride flag, and calendar features.

A representative sample is used because GradientBoostingRegressor
is computationally expensive on very large datasets.

Usage:
    python src/train_duration_fhv.py
"""
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
"""