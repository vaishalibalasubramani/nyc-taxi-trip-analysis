"""
Generate interactive Plotly visualizations for Yellow Taxi.

Creates standalone HTML files in:

    outputs/plots/yellow/

Visualizations:

    Task 1 -- Trip Duration Prediction
        1. Actual vs Predicted Duration
        2. Feature Importance

    Task 2 -- Citywide Demand Prediction
        3. Actual vs Predicted Hourly Demand

    Task 3 -- Taxi Zone Demand Forecasting
        4. Top 15 Zones by Predicted Demand
        5. Predicted Demand by Borough Over Time

Run:

    python src/visualize_results_yellow.py

The script reads the already-generated Yellow Taxi models,
processed data, and prediction files.
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# Keep Yellow Taxi plots separate from FHV / Green / HVFHV
PLOTS_DIR = OUTPUTS_DIR / "plots" / "yellow"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_file(path, description):
    """
    Check whether an input file exists.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\n{description} was not found:\n"
            f"{path}\n\n"
            "Make sure the required Yellow Taxi "
            "pipeline/model has been run first."
        )


def get_duration_features(model, df):
    """
    Get the exact feature columns used by the saved
    Yellow Taxi duration model.

    This avoids hard-coding features such as trip_type,
    payment_type, etc.

    If the saved estimator contains feature_names_in_,
    those exact names are used.

    Otherwise a fallback list is used.
    """

    # -------------------------------------------------------------------------
    # Preferred method
    # -------------------------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        feature_cols = list(
            model.feature_names_in_
        )

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------

    else:

        feature_cols = [
            "trip_distance",
            "passenger_count",
            "PULocationID",
            "DOLocationID",
            "RatecodeID",
            "payment_type",
            "pickup_hour",
            "pickup_dow",
            "is_weekend",
            "pickup_month",
        ]

    # -------------------------------------------------------------------------
    # Check that every model feature exists in the dataset
    # -------------------------------------------------------------------------

    missing = [
        col
        for col in feature_cols
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nThe saved Yellow Taxi duration model "
            "expects these columns, but they are missing "
            "from yellow_trip_features.parquet:\n\n"
            f"Missing columns: {missing}\n\n"
            "Available columns:\n"
            f"{list(df.columns)}"
        )

    return feature_cols


# =============================================================================
# TASK 1 -- TRIP DURATION
# =============================================================================

def plot_duration_task():
    """
    Generate Yellow Taxi Task 1 visualizations.

    Outputs:

        yellow_task1_duration_actual_vs_predicted.html

        yellow_task1_duration_feature_importance.html
    """

    print("\nBuilding Yellow Taxi Task 1 plots...")

    model_path = (
        OUTPUTS_DIR /
        "yellow_duration_model.joblib"
    )

    feature_path = (
        PROCESSED_DIR /
        "yellow_trip_features.parquet"
    )

    # -------------------------------------------------------------------------
    # Check required files
    # -------------------------------------------------------------------------

    check_file(
        model_path,
        "Yellow Taxi duration model",
    )

    check_file(
        feature_path,
        "Yellow Taxi trip feature data",
    )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    model = joblib.load(
        model_path
    )

    # -------------------------------------------------------------------------
    # Load processed data
    # -------------------------------------------------------------------------

    df = pd.read_parquet(
        feature_path
    )

    print(
        f"  Loaded Yellow Taxi feature data: "
        f"{len(df):,} rows"
    )

    # -------------------------------------------------------------------------
    # Get exact model features
    # -------------------------------------------------------------------------

    feature_cols = get_duration_features(
        model,
        df,
    )

    print("  Model features:")

    for col in feature_cols:
        print(f"    - {col}")

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    required = feature_cols + [
        "trip_duration_s"
    ]

    # -------------------------------------------------------------------------
    # Remove missing values
    # -------------------------------------------------------------------------

    df = df.dropna(
        subset=required
    )

    if df.empty:

        raise ValueError(
            "No valid Yellow Taxi rows remain "
            "after removing NULL values."
        )

    print(
        f"  Rows available for visualization: "
        f"{len(df):,}"
    )

    # -------------------------------------------------------------------------
    # Sample data
    # -------------------------------------------------------------------------

    sample = df.sample(
        n=min(5000, len(df)),
        random_state=42,
    )

    print(
        f"  Plotting sample: "
        f"{len(sample):,} trips"
    )

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    predictions = model.predict(
        sample[feature_cols]
    )

    # =========================================================================
    # TASK 1 — PLOT 1
    # Actual vs Predicted Duration
    # =========================================================================

    fig = px.scatter(
        x=sample["trip_duration_s"],
        y=predictions,
        labels={
            "x": "Actual duration (seconds)",
            "y": "Predicted duration (seconds)",
        },
        title=(
            "Yellow Taxi — Task 1: "
            "Trip Duration Actual vs Predicted"
        ),
        opacity=0.4,
    )

    # -------------------------------------------------------------------------
    # Perfect prediction line
    # -------------------------------------------------------------------------

    max_actual = sample[
        "trip_duration_s"
    ].max()

    max_predicted = predictions.max()

    max_val = max(
        max_actual,
        max_predicted,
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(
                dash="dash",
                color="red",
            ),
            name="Perfect prediction",
        )
    )

    fig.update_layout(
        xaxis_title="Actual duration (seconds)",
        yaxis_title="Predicted duration (seconds)",
        hovermode="closest",
    )

    duration_plot_path = (
        PLOTS_DIR /
        "yellow_task1_duration_actual_vs_predicted.html"
    )

    fig.write_html(
        duration_plot_path
    )

    print(
        "  -> yellow_task1_duration_actual_vs_predicted.html"
    )

    # =========================================================================
    # TASK 1 — PLOT 2
    # Feature Importance
    # =========================================================================

    if hasattr(
        model,
        "feature_importances_"
    ):

        importances = pd.Series(
            model.feature_importances_,
            index=feature_cols,
        ).sort_values(
            ascending=True
        )

        fig2 = px.bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            labels={
                "x": "Feature importance",
                "y": "Feature",
            },
            title=(
                "Yellow Taxi — Task 1: "
                "Trip Duration Feature Importance"
            ),
        )

        fig2.update_layout(
            xaxis_title="Feature importance",
            yaxis_title="Feature",
        )

        importance_plot_path = (
            PLOTS_DIR /
            "yellow_task1_duration_feature_importance.html"
        )

        fig2.write_html(
            importance_plot_path
        )

        print(
            "  -> "
            "yellow_task1_duration_feature_importance.html"
        )

    else:

        print(
            "  -> Feature importance skipped: "
            "the saved model does not provide "
            "feature_importances_."
        )


# =============================================================================
# TASK 2 -- CITYWIDE DEMAND
# =============================================================================

def plot_demand_task():
    """
    Generate Yellow Taxi Task 2 visualization.

    Output:

        yellow_task2_demand_actual_vs_predicted.html
    """

    print("\nBuilding Yellow Taxi Task 2 plots...")

    prediction_path = (
        OUTPUTS_DIR /
        "yellow_demand_predictions.parquet"
    )

    # -------------------------------------------------------------------------
    # Check file
    # -------------------------------------------------------------------------

    check_file(
        prediction_path,
        "Yellow Taxi citywide demand predictions",
    )

    # -------------------------------------------------------------------------
    # Load predictions
    # -------------------------------------------------------------------------

    results = pd.read_parquet(
        prediction_path
    )

    print(
        f"  Loaded demand predictions: "
        f"{len(results):,} rows"
    )

    # -------------------------------------------------------------------------
    # Check required columns
    # -------------------------------------------------------------------------

    required_columns = [
        "actual",
        "predicted",
    ]

    missing = [
        col
        for col in required_columns
        if col not in results.columns
    ]

    if missing:

        raise ValueError(
            "Yellow Taxi demand prediction file "
            f"is missing: {missing}\n\n"
            f"Available columns: {list(results.columns)}"
        )

    # -------------------------------------------------------------------------
    # Determine x-axis
    # -------------------------------------------------------------------------

    if "pickup_hour_ts" in results.columns:

        x_values = results[
            "pickup_hour_ts"
        ]

        x_title = "Hour"

    elif "datetime" in results.columns:

        x_values = results[
            "datetime"
        ]

        x_title = "Hour"

    elif "timestamp" in results.columns:

        x_values = results[
            "timestamp"
        ]

        x_title = "Hour"

    else:

        x_values = results.index

        x_title = "Hour"

    # =========================================================================
    # Actual vs Predicted Demand
    # =========================================================================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=results["actual"],
            name="Actual",
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=results["predicted"],
            name="Predicted",
            mode="lines",
        )
    )

    fig.update_layout(
        title=(
            "Yellow Taxi — Task 2: "
            "Citywide Hourly Demand "
            "Actual vs Predicted"
        ),
        xaxis_title=x_title,
        yaxis_title="Trips",
        hovermode="x unified",
    )

    demand_plot_path = (
        PLOTS_DIR /
        "yellow_task2_demand_actual_vs_predicted.html"
    )

    fig.write_html(
        demand_plot_path
    )

    print(
        "  -> yellow_task2_demand_actual_vs_predicted.html"
    )


# =============================================================================
# TASK 3 -- ZONE DEMAND
# =============================================================================

def plot_zone_task():
    """
    Generate Yellow Taxi Task 3 visualizations.

    Outputs:

        yellow_task3_zone_top_demand.html

        yellow_task3_zone_borough_timeseries.html
    """

    print("\nBuilding Yellow Taxi Task 3 plots...")

    zone_data_path = (
        PROCESSED_DIR /
        "yellow_zone_hourly_demand.parquet"
    )

    prediction_path = (
        OUTPUTS_DIR /
        "yellow_zone_demand_predictions.parquet"
    )

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    check_file(
        zone_data_path,
        "Yellow Taxi zone hourly demand data",
    )

    check_file(
        prediction_path,
        "Yellow Taxi zone demand predictions",
    )

    # -------------------------------------------------------------------------
    # Load zone data
    # -------------------------------------------------------------------------

    zone_data = pd.read_parquet(
        zone_data_path
    )

    print(
        f"  Loaded zone data: "
        f"{len(zone_data):,} rows"
    )

    # -------------------------------------------------------------------------
    # Required zone metadata
    # -------------------------------------------------------------------------

    required_zone_columns = [
        "zone_id",
        "zone_name",
        "borough",
    ]

    missing_zone_columns = [
        col
        for col in required_zone_columns
        if col not in zone_data.columns
    ]

    if missing_zone_columns:

        raise ValueError(
            "Yellow Taxi zone hourly demand data "
            "is missing these columns:\n"
            f"{missing_zone_columns}\n\n"
            f"Available columns:\n"
            f"{list(zone_data.columns)}"
        )

    # -------------------------------------------------------------------------
    # Create zone metadata lookup
    # -------------------------------------------------------------------------

    zone_meta = (
        zone_data[
            required_zone_columns
        ]
        .drop_duplicates(
            "zone_id"
        )
    )

    # -------------------------------------------------------------------------
    # Load prediction results
    # -------------------------------------------------------------------------

    results = pd.read_parquet(
        prediction_path
    )

    print(
        f"  Loaded zone predictions: "
        f"{len(results):,} rows"
    )

    print(
        "  Prediction columns:"
    )

    print(
        f"    {list(results.columns)}"
    )

    # -------------------------------------------------------------------------
    # Required prediction columns
    # -------------------------------------------------------------------------

    if "zone_id" not in results.columns:

        raise ValueError(
            "Yellow Taxi zone predictions do not "
            "contain 'zone_id'.\n\n"
            f"Available columns:\n"
            f"{list(results.columns)}"
        )

    if "predicted" not in results.columns:

        raise ValueError(
            "Yellow Taxi zone predictions do not "
            "contain 'predicted'.\n\n"
            f"Available columns:\n"
            f"{list(results.columns)}"
        )

    # =========================================================================
    # SAFELY HANDLE ZONE METADATA
    # =========================================================================
    #
    # Some prediction files may already contain zone_name / borough.
    #
    # We therefore avoid blindly merging those columns.
    #
    # If zone_name or borough already exists, keep it.
    # Otherwise, obtain it from zone_meta.
    #
    # =========================================================================

    # -------------------------------------------------------------------------
    # Zone name
    # -------------------------------------------------------------------------

    if "zone_name" not in results.columns:

        zone_name_lookup = zone_meta[
            [
                "zone_id",
                "zone_name",
            ]
        ]

        results = results.merge(
            zone_name_lookup,
            on="zone_id",
            how="left",
        )

    # -------------------------------------------------------------------------
    # Borough
    # -------------------------------------------------------------------------

    if "borough" not in results.columns:

        borough_lookup = zone_meta[
            [
                "zone_id",
                "borough",
            ]
        ]

        results = results.merge(
            borough_lookup,
            on="zone_id",
            how="left",
        )

    # -------------------------------------------------------------------------
    # Handle any unusual _x / _y columns safely
    # -------------------------------------------------------------------------

    if "zone_name" not in results.columns:

        if "zone_name_x" in results.columns:

            results["zone_name"] = (
                results["zone_name_x"]
            )

        elif "zone_name_y" in results.columns:

            results["zone_name"] = (
                results["zone_name_y"]
            )

    if "borough" not in results.columns:

        if "borough_x" in results.columns:

            results["borough"] = (
                results["borough_x"]
            )

        elif "borough_y" in results.columns:

            results["borough"] = (
                results["borough_y"]
            )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    if "zone_name" not in results.columns:

        raise ValueError(
            "Unable to determine Yellow Taxi zone names.\n\n"
            "Available columns after metadata processing:\n"
            f"{list(results.columns)}"
        )

    if "borough" not in results.columns:

        raise ValueError(
            "Unable to determine Yellow Taxi borough names.\n\n"
            "Available columns after metadata processing:\n"
            f"{list(results.columns)}"
        )

    # -------------------------------------------------------------------------
    # Clean missing metadata
    # -------------------------------------------------------------------------

    results["zone_name"] = (
        results["zone_name"]
        .fillna(
            results["zone_id"].astype(str)
        )
    )

    results["borough"] = (
        results["borough"]
        .fillna("Unknown")
    )

    # =========================================================================
    # TASK 3 — PLOT 1
    # TOP 15 ZONES
    # =========================================================================

    top_zones = (
        results.groupby(
            [
                "zone_id",
                "zone_name",
            ]
        )["predicted"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        top_zones,
        x="predicted",
        y="zone_name",
        orientation="h",
        labels={
            "predicted":
                "Average predicted trips/hour",
            "zone_name":
                "Taxi zone",
        },
        title=(
            "Yellow Taxi — Task 3: "
            "Top 15 Zones by Predicted Demand"
        ),
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        )
    )

    top_zone_plot_path = (
        PLOTS_DIR /
        "yellow_task3_zone_top_demand.html"
    )

    fig.write_html(
        top_zone_plot_path
    )

    print(
        "  -> yellow_task3_zone_top_demand.html"
    )

    # =========================================================================
    # TASK 3 — PLOT 2
    # BOROUGH DEMAND OVER TIME
    # =========================================================================

    if "pickup_hour_ts" in results.columns:

        borough_ts = (
            results.groupby(
                [
                    "pickup_hour_ts",
                    "borough",
                ]
            )["predicted"]
            .sum()
            .reset_index()
        )

        fig2 = px.line(
            borough_ts,
            x="pickup_hour_ts",
            y="predicted",
            color="borough",
            title=(
                "Yellow Taxi — Task 3: "
                "Predicted Demand by Borough Over Time"
            ),
            labels={
                "predicted":
                    "Predicted trips/hour",
                "pickup_hour_ts":
                    "Hour",
                "borough":
                    "Borough",
            },
        )

        fig2.update_layout(
            hovermode="x unified"
        )

        borough_plot_path = (
            PLOTS_DIR /
            "yellow_task3_zone_borough_timeseries.html"
        )

        fig2.write_html(
            borough_plot_path
        )

        print(
            "  -> yellow_task3_zone_borough_timeseries.html"
        )

    else:

        print(
            "  -> Borough time-series plot skipped: "
            "'pickup_hour_ts' was not found in "
            "the zone prediction file."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("YELLOW TAXI — INTERACTIVE VISUALIZATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Output directory:"
    )
    print(
        PLOTS_DIR
    )

    # -------------------------------------------------------------------------
    # Run all tasks
    # -------------------------------------------------------------------------

    plot_duration_task()

    plot_demand_task()

    plot_zone_task()

    # -------------------------------------------------------------------------
    # Completion
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("YELLOW TAXI VISUALIZATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "All Yellow Taxi plots were saved to:"
    )
    print(
        PLOTS_DIR
    )

    print()
    print(
        "Generated files:"
    )

    html_files = sorted(
        PLOTS_DIR.glob("*.html")
    )

    if html_files:

        for path in html_files:
            print(
                f"  - {path.name}"
            )

    else:

        print(
            "  No HTML files were generated."
        )

    print()
    print(
        "Open any HTML file in your browser "
        "to view the interactive Plotly visualization."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()