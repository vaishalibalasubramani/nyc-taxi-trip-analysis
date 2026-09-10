"""
Standalone project explanation / results report.

This script explains:
    - what the NYC TLC project does
    - Task 1 / Task 2 / Task 3
    - train/validation/test concept
    - duration and demand metrics
    - demand validation results
    - duration validation results
    - baseline comparisons
    - data-quality validation
    - same-zone diagnostic purpose

It reads the validation output files if they exist, so the numerical
results shown by the report come from the project's generated artifacts.

Run:
    python src/explain_project.py
"""

from pathlib import Path
import json
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
VALIDATION = OUTPUTS / "validation"


# =============================================================================
# HELPERS
# =============================================================================

def pct(x):
    """Format a numeric value as a percentage."""
    return f"{x:.2f}%" if pd.notna(x) else "N/A"


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)


def safe_float(value):
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =============================================================================
# 1. METRICS
# =============================================================================

def explain_metrics():
    print_header("1. METRICS EXPLAINED")

    print(
        """
MAE (Mean Absolute Error)
    Average absolute prediction error.

    For duration:
        Average number of seconds by which the prediction differs
        from the actual trip duration.

    For demand:
        Average number of trips by which the prediction differs
        from the actual demand.

    Lower is better.


RMSE (Root Mean Squared Error)
    Similar to MAE, but penalizes large errors more strongly.

    Lower is better.


R² (R-squared)
    Measures how much variation in the target is explained by the model.

    1.0 = perfect prediction
    0.0 = no better than a constant-mean predictor
    Negative = worse than the constant-mean predictor


Baseline
    A simple reference prediction used to determine whether the ML model
    provides useful predictive value rather than merely producing
    plausible-looking numbers.


Mean baseline
    Predicts using the average target value from the training/reference data.


Persistence baseline
    Predicts the current hour using the previous hour's demand.


Seasonal-naive baseline
    Predicts the current hour using demand from the same hour one week
    earlier.

    Lag = 168 hours.


TimeSeriesSplit
    Chronological cross-validation.

    Earlier observations are used for training and later observations
    are used for validation.

    This is appropriate for forecasting because future observations
    should not be used to train a model evaluated on an earlier period.
"""
    )


# =============================================================================
# 2. PROJECT TASKS
# =============================================================================

def explain_tasks():
    print_header("2. WHAT THE PROJECT DOES")

    print(
        """
Task 1 -- Trip Duration Prediction
    Predict the duration of an individual taxi trip from trip,
    time and route information.

    The prediction is evaluated against the actual trip duration.


Task 2 -- Citywide Taxi Demand Prediction
    Predict the total number of taxi trips for an hour across the city.


Task 3 -- Taxi Zone Demand Forecasting
    Predict hourly demand separately for taxi zones.


Final vehicle scope
    Yellow Taxi
    FHV (For-Hire Vehicle)


The project therefore contains six trained models:

    2 Trip Duration models
        - Yellow Taxi
        - FHV

    2 Citywide Demand models
        - Yellow Taxi
        - FHV

    2 Zone Demand models
        - Yellow Taxi
        - FHV
"""
    )


# =============================================================================
# 3. VALIDATION
# =============================================================================

def explain_validation():
    print_header("3. VALIDATIONS PERFORMED")

    print(
        """
A. Duration model validation

    - Saved duration models were loaded.
    - Model feature names were taken from the saved estimators.
    - A 5-fold validation run was used for diagnostic evaluation.
    - MAE, RMSE and R² were calculated.
    - Results were compared with simple baselines where available.
    - This checks general model behavior.

    Important:
        This diagnostic validation is separate from a final untouched
        December test-set evaluation.


B. Demand model validation

    - Task 2 and Task 3 were validated.
    - Yellow Taxi and FHV were both tested.
    - 5-fold TimeSeriesSplit was used.
    - Mean, persistence and seasonal-naive baselines were calculated.
    - The model was compared against these baselines.


C. Data-quality validation

    - Processed Parquet files were checked directly.
    - NULL values were checked.
    - Negative values were checked.
    - Range violations were checked.
    - Duplicate timestamps/keys were checked.
    - Calendar feature ranges were checked.
    - Zone-ID integrity was checked.
    - Date coverage was checked.


D. Same-zone duration diagnostic

    - Compares trips where pickup zone equals drop-off zone
      against trips crossing zones.
    - Examines short same-zone trips.
    - Examines same-zone trips by distance bucket.
    - The purpose is to identify localized error patterns rather than
      relying only on the overall MAE.
"""
    )


# =============================================================================
# DEMAND RESULTS
# =============================================================================

def load_demand_results():
    """
    Load the demand validation summary.

    The validation artifact may have slightly different columns depending
    on which version of the validation script produced it.
    """

    path = VALIDATION / "yellow_fhv_demand_validation_summary.csv"

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read demand validation CSV: {exc}")
        return None

    if df.empty:
        return None

    # Clean column names.
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    return df


def load_demand_report_text():
    """
    Load the text demand validation report if available.

    This is used as a fallback when the CSV does not contain MAE/RMSE/R².
    """

    path = VALIDATION / "yellow_fhv_demand_validation_report.txt"

    if not path.exists():
        return None

    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def extract_metric_from_report(report_text, task_name, metric_name):
    """
    Try to extract a metric from the text validation report.

    This is deliberately conservative. If the value cannot be found,
    None is returned rather than inventing a number.
    """

    if not report_text:
        return None

    # Locate the task section.
    task_pattern = re.escape(task_name)

    match = re.search(
        rf"{task_pattern}.*?(?=\n[A-Z][^\n]*:|\Z)",
        report_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        # Fall back to searching near the task name.
        position = report_text.lower().find(task_name.lower())

        if position < 0:
            return None

        section = report_text[position:position + 2500]
    else:
        section = match.group(0)

    metric_pattern = re.escape(metric_name)

    metric_match = re.search(
        rf"{metric_pattern}\s*[:=]\s*([-+]?\d*\.?\d+)",
        section,
        flags=re.IGNORECASE,
    )

    if not metric_match:
        return None

    return safe_float(metric_match.group(1))


def normalize_demand_columns(df):
    """
    Normalize common column-name variations.

    Example:
        R² -> R2
        r2 -> R2
    """

    rename_map = {}

    for col in df.columns:
        normalized = (
            str(col)
            .strip()
            .lower()
            .replace("²", "2")
            .replace(" ", "_")
        )

        if normalized == "task":
            rename_map[col] = "Task"

        elif normalized == "mae":
            rename_map[col] = "MAE"

        elif normalized == "rmse":
            rename_map[col] = "RMSE"

        elif normalized in ("r2", "r_2"):
            rename_map[col] = "R2"

        elif normalized == "mean_mae":
            rename_map[col] = "Mean_MAE"

        elif normalized == "persistence_mae":
            rename_map[col] = "Persistence_MAE"

        elif normalized in (
            "seasonal_naive_mae",
            "seasonal_naive",
        ):
            rename_map[col] = "Seasonal_Naive_MAE"

        elif normalized == "status":
            rename_map[col] = "Status"

    return df.rename(columns=rename_map)


def show_demand_results():
    print_header("4. DEMAND VALIDATION RESULTS")

    df = load_demand_results()

    if df is None:
        print("Demand validation summary not found.")
        print(
            "Run validate_demand_models.py first."
        )
        return

    df = normalize_demand_columns(df)

    # -------------------------------------------------------------------------
    # Display whatever columns actually exist.
    # -------------------------------------------------------------------------

    preferred_cols = [
        "Task",
        "MAE",
        "RMSE",
        "R2",
        "Mean_MAE",
        "Persistence_MAE",
        "Seasonal_Naive_MAE",
        "Status",
    ]

    available_cols = [
        col for col in preferred_cols
        if col in df.columns
    ]

    if available_cols:
        print(df[available_cols].to_string(index=False))
    else:
        print(df.to_string(index=False))

    # -------------------------------------------------------------------------
    # If MAE/RMSE/R2 are missing, try the text report.
    # -------------------------------------------------------------------------

    report_text = None

    if "MAE" not in df.columns:
        report_text = load_demand_report_text()

    print("\nInterpretation:")

    for _, row in df.iterrows():

        task = row.get("Task", "Unknown")

        mae = safe_float(row.get("MAE"))
        rmse = safe_float(row.get("RMSE"))
        r2 = safe_float(row.get("R2"))

        # Try text report fallback if CSV does not contain model metrics.
        if mae is None:
            mae = extract_metric_from_report(
                report_text,
                str(task),
                "MAE",
            )

        if rmse is None:
            rmse = extract_metric_from_report(
                report_text,
                str(task),
                "RMSE",
            )

        if r2 is None:
            r2 = extract_metric_from_report(
                report_text,
                str(task),
                "R2",
            )

        status = row.get("Status", "N/A")

        print(f"- {task}:")

        if mae is not None:
            print(f"    MAE  = {mae:.2f}")
        else:
            print("    MAE  = N/A")

        if rmse is not None:
            print(f"    RMSE = {rmse:.2f}")
        else:
            print("    RMSE = N/A")

        if r2 is not None:
            print(f"    R²   = {r2:.3f}")
        else:
            print("    R²   = N/A")

        print(f"    Status = {status}")

        persistence = safe_float(
            row.get("Persistence_MAE")
        )

        seasonal = safe_float(
            row.get("Seasonal_Naive_MAE")
        )

        mean_mae = safe_float(
            row.get("Mean_MAE")
        )

        if mean_mae is not None:
            print(
                f"    Mean baseline MAE = "
                f"{mean_mae:.2f}"
            )

        if persistence is not None:
            print(
                f"    Persistence baseline MAE = "
                f"{persistence:.2f}"
            )

        if seasonal is not None:
            print(
                f"    Seasonal-naive baseline MAE = "
                f"{seasonal:.2f}"
            )

        # -------------------------------------------------------------
        # Compare against baselines when sufficient information exists.
        # -------------------------------------------------------------

        baselines = [
            value
            for value in [
                mean_mae,
                persistence,
                seasonal,
            ]
            if value is not None
        ]

        if mae is not None and baselines:

            beaten = all(
                mae < baseline
                for baseline in baselines
            )

            if beaten:
                print(
                    "    Result: Model MAE beats all "
                    "available demand baselines."
                )
            else:
                print(
                    "    Result: Model MAE does not beat "
                    "all available demand baselines."
                )

        print()


# =============================================================================
# 5. DURATION RESULTS
# =============================================================================

def show_duration_results():
    print_header("5. DURATION VALIDATION RESULTS")

    json_path = (
        VALIDATION
        / "yellow_fhv_duration_validation_results.json"
    )

    csv_path = (
        VALIDATION
        / "yellow_fhv_duration_validation_summary.csv"
    )

    found = False

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    if json_path.exists():

        found = True

        try:
            data = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )

            print("\nJSON validation results:")

            if isinstance(data, dict):
                print(
                    json.dumps(
                        data,
                        indent=2,
                        default=str,
                    )
                )
            else:
                print(data)

        except Exception as exc:
            print(
                f"Could not read duration JSON: {exc}"
            )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    if csv_path.exists():

        found = True

        try:
            df = pd.read_csv(csv_path)

            print("\nCSV validation summary:")
            print(
                df.to_string(index=False)
            )

        except Exception as exc:
            print(
                f"Could not read duration CSV: {exc}"
            )

    if not found:
        print(
            "Duration validation artifact was not found."
        )
        print(
            "Run validate_duration_models.py first."
        )


# =============================================================================
# 6. DATA QUALITY
# =============================================================================

def show_quality():
    print_header("6. DATA QUALITY RESULTS")

    candidates = [
        OUTPUTS / "data_quality_yellow_fhv_report.txt",
        OUTPUTS / "data_quality_report.txt",
    ]

    for path in candidates:

        if path.exists():

            try:
                print(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception as exc:
                print(
                    f"Could not read data-quality report: "
                    f"{exc}"
                )

            return

    print(
        "Data-quality report not found."
    )

    print(
        "Run audit_data_quality.py first."
    )


# =============================================================================
# 7. MODEL PERFORMANCE AND ERROR INTERPRETATION
# =============================================================================

def explain_model_performance_and_errors():
    print_header(
        "7. HOW TO EXPLAIN MODEL PERFORMANCE AND ERRORS"
    )

    print(
        """
When presenting the project, do not rely on R² alone.


For duration:

    - MAE is the easiest metric to explain to a non-technical audience
      because it directly represents the average prediction error.

    - RMSE tells you whether large prediction mistakes are a problem.

    - R² tells you how much variation in trip duration is captured
      by the model.

    - Compare model error with simple baselines.

    - Investigate specific error groups such as:
        * same-zone trips
        * short-distance trips
        * peak hours
        * different vehicle types
        * other route/time segments


Same-zone diagnostic:

    The same-zone diagnostic is particularly useful because a model can
    have an acceptable overall MAE while still performing poorly for a
    specific trip group.

    For Yellow Taxi, same-zone trips have lower absolute MAE than
    different-zone trips, although percentage error is high for short trips.

    For FHV, same-zone trips show a notable overprediction pattern:
    the average predicted duration is substantially higher than the
    average actual duration.

    This is an error pattern worth discussing rather than hiding it.


For demand:

    - MAE tells you the average number of trips by which the forecast misses.

    - RMSE highlights unusually large demand misses.

    - R² shows how much hourly variation is explained.

    - The strongest practical evidence is that the ML model is compared
      against simple forecasting baselines such as persistence and
      seasonal-naive forecasting.


Important evaluation distinction:

    Cross-validation estimates generalization using multiple folds.

    A final held-out test set is still the cleanest final performance
    report because that data is not used during model development.

    Therefore, cross-validation results should be described as validation
    evidence rather than claiming that they are an untouched final test
    result.
"""
    )


# =============================================================================
# 8. FINAL PRESENTATION MESSAGE
# =============================================================================

def final_presentation_message():
    print_header("8. FINAL PRESENTATION MESSAGE")

    print(
        """
The project is an end-to-end NYC TLC 2025 analytics and
machine-learning system.

It combines:

    1. Data ingestion and preprocessing
    2. Feature engineering
    3. Trip duration prediction
    4. Citywide demand prediction
    5. Zone-level demand forecasting
    6. Model validation against baselines
    7. Independent data-quality auditing
    8. Same-zone error diagnostics
    9. Dashboard-based prediction and
       model-performance reporting


Final vehicle scope:

    Yellow Taxi
    FHV (For-Hire Vehicle)


Final model scope:

    Task 1:
        Yellow Taxi duration model
        FHV duration model

    Task 2:
        Yellow Taxi citywide demand model
        FHV citywide demand model

    Task 3:
        Yellow Taxi zone demand model
        FHV zone demand model


The dashboard presents the saved model outputs, while the validation
scripts provide evidence that the models were evaluated systematically.

The data-quality audit verifies that the processed datasets satisfy
the expected integrity and range checks.

The same-zone diagnostic provides additional evidence about where
duration-model errors occur and helps explain model limitations.
"""
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 85)
    print(
        "NYC TLC 2025 ML PROJECT -- PROJECT EXPLANATION"
    )
    print(
        "FINAL SCOPE: YELLOW TAXI + FHV"
    )
    print("=" * 85)

    # Keep section numbering in the same order as the report.
    explain_metrics()
    explain_tasks()
    explain_validation()
    show_demand_results()
    show_duration_results()
    show_quality()
    explain_model_performance_and_errors()
    final_presentation_message()

    print("\n")
    print("=" * 85)
    print("PROJECT EXPLANATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    main()