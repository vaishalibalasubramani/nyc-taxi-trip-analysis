"""
FHV Taxi — Interactive Visualization

Generates Plotly HTML visualizations for:

Task 1: Trip Duration Prediction
    - Actual vs Predicted duration
    - Feature importance

Task 2: Citywide Demand Prediction
    - Actual vs Predicted hourly demand

Task 3: Taxi Zone Demand Forecasting
    - Top zones by demand
    - Borough-level time series
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px


# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

PLOT_DIR = OUTPUTS_DIR / "plots" / "fhv"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FILE PATHS
# =============================================================================

TRIP_FEATURES = (
    PROCESSED_DIR / "fhv_trip_features.parquet"
)

HOURLY_DEMAND = (
    PROCESSED_DIR / "fhv_hourly_demand.parquet"
)

ZONE_DEMAND = (
    PROCESSED_DIR / "fhv_zone_hourly_demand.parquet"
)

DURATION_MODEL = (
    OUTPUTS_DIR / "fhv_duration_model.joblib"
)

DURATION_PREDICTIONS = (
    OUTPUTS_DIR / "fhv_duration_predictions.parquet"
)

DEMAND_PREDICTIONS = (
    OUTPUTS_DIR / "fhv_demand_predictions.parquet"
)

ZONE_DEMAND_PREDICTIONS = (
    OUTPUTS_DIR / "fhv_zone_demand_predictions.parquet"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_column(df, candidates, required=True):
    """
    Find the first available column from a list of possible names.
    """

    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def get_model_features(model):
    """
    Get the feature names used by the trained model.
    """

    # Scikit-learn models
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # If model was saved as a dictionary/bundle
    if isinstance(model, dict):

        possible_keys = [
            "feature_names",
            "features",
            "feature_names_in_",
        ]

        for key in possible_keys:
            if key in model:
                return list(model[key])

    return []


# =============================================================================
# TASK 1 — TRIP DURATION
# =============================================================================

def plot_duration_task():

    print("Building FHV Task 1 plots...")

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    if not TRIP_FEATURES.exists():
        raise FileNotFoundError(
            f"Missing FHV feature file:\n{TRIP_FEATURES}"
        )

    if not DURATION_MODEL.exists():
        raise FileNotFoundError(
            f"Missing FHV duration model:\n{DURATION_MODEL}"
        )

    # -------------------------------------------------------------------------
    # Load data and model
    # -------------------------------------------------------------------------

    df = pd.read_parquet(TRIP_FEATURES)

    model = joblib.load(DURATION_MODEL)

    print(
        f"  Loaded FHV feature data: {len(df):,} rows"
    )

    # -------------------------------------------------------------------------
    # Get model features
    # -------------------------------------------------------------------------

    model_features = get_model_features(model)

    if not model_features:
        raise ValueError(
            "Could not determine the feature names used by "
            "the FHV duration model."
        )

    print("  Model features:")

    for feature in model_features:
        print(f"    - {feature}")

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # FHV uses trip_duration_s as the target.
    # -------------------------------------------------------------------------

    duration_col = find_column(
        df,
        [
            "trip_duration_s",
            "trip_duration",
            "duration_seconds",
            "duration",
            "trip_time",
        ],
    )

    print(
        f"  Duration column: {duration_col}"
    )

    # -------------------------------------------------------------------------
    # Check model features exist
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:

        print(
            "\n  ERROR: The following model features are missing:"
        )

        for feature in missing_features:
            print(f"    - {feature}")

        print(
            "\n  Available columns:"
        )

        for column in df.columns:
            print(f"    - {column}")

        raise KeyError(
            f"Missing model features: {missing_features}"
        )

    # -------------------------------------------------------------------------
    # Drop missing values
    # -------------------------------------------------------------------------

    required_columns = [
        duration_col
    ] + model_features

    df = df.dropna(
        subset=required_columns
    )

    print(
        f"  Rows available for visualization: {len(df):,}"
    )

    # -------------------------------------------------------------------------
    # Sample data
    # -------------------------------------------------------------------------

    sample_size = min(
        5000,
        len(df)
    )

    df_sample = df.sample(
        n=sample_size,
        random_state=42,
    )

    print(
        f"  Plotting sample: {sample_size:,} trips"
    )

    # -------------------------------------------------------------------------
    # Prepare model input
    # -------------------------------------------------------------------------

    X = df_sample[model_features]

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    predictions = model.predict(X)

    # -------------------------------------------------------------------------
    # Actual vs predicted
    # -------------------------------------------------------------------------

    plot_df = pd.DataFrame(
        {
            "Actual Duration (min)":
                df_sample[duration_col] / 60.0,

            "Predicted Duration (min)":
                predictions / 60.0,
        }
    )

    fig = px.scatter(
        plot_df,
        x="Actual Duration (min)",
        y="Predicted Duration (min)",
        title=(
            "FHV Taxi — Actual vs Predicted "
            "Trip Duration"
        ),
        opacity=0.45,
    )

    fig.update_layout(
        xaxis_title="Actual Duration (minutes)",
        yaxis_title="Predicted Duration (minutes)",
    )

    output_file = (
        PLOT_DIR
        / "fhv_task1_duration_actual_vs_predicted.html"
    )

    fig.write_html(
        output_file
    )

    print(
        f"  -> {output_file.name}"
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame(
            {
                "Feature": model_features,
                "Importance":
                    model.feature_importances_,
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=True,
            )
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                "FHV Taxi — "
                "Trip Duration Feature Importance"
            ),
        )

        output_file = (
            PLOT_DIR
            / "fhv_task1_duration_feature_importance.html"
        )

        fig.write_html(
            output_file
        )

        print(
            f"  -> {output_file.name}"
        )

    else:

        print(
            "  WARNING: Model does not expose "
            "feature_importances_."
        )


# =============================================================================
# TASK 2 — CITYWIDE DEMAND
# =============================================================================

def plot_demand_task():

    print("\nBuilding FHV Task 2 plots...")

    # -------------------------------------------------------------------------
    # Check file
    # -------------------------------------------------------------------------

    if not DEMAND_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing FHV demand predictions:\n"
            f"{DEMAND_PREDICTIONS}"
        )

    # -------------------------------------------------------------------------
    # Load predictions
    # -------------------------------------------------------------------------

    df = pd.read_parquet(
        DEMAND_PREDICTIONS
    )

    print(
        f"  Loaded demand predictions: {len(df):,} rows"
    )

    print(
        "  Prediction columns:"
    )

    print(
        f"    {list(df.columns)}"
    )

    # -------------------------------------------------------------------------
    # Detect timestamp
    #
    # FHV demand prediction files may contain only:
    #     actual, predicted
    #
    # In that case, recover the timestamps from the original hourly
    # demand dataset. The prediction file contains the test-period rows,
    # so we use the final N hourly timestamps in the same chronological
    # order.
    # -------------------------------------------------------------------------

    timestamp_col = find_column(
        df,
        [
            "pickup_hour_ts",
            "pickup_hour",
            "hour",
            "timestamp",
            "datetime",
            "pickup_datetime",
        ],
        required=False,
    )

    if timestamp_col is None:

        if not HOURLY_DEMAND.exists():
            raise FileNotFoundError(
                "The FHV demand prediction file does not contain a "
                "timestamp column, and the original hourly demand file "
                f"was not found:\n{HOURLY_DEMAND}"
            )

        hourly_df = pd.read_parquet(HOURLY_DEMAND)

        print(
            "  Timestamp column not present in prediction file."
        )
        print(
            "  Recovering timestamps from FHV hourly demand data..."
        )

        hourly_timestamp_col = find_column(
            hourly_df,
            [
                "pickup_hour_ts",
                "pickup_hour",
                "hour",
                "timestamp",
                "datetime",
                "pickup_datetime",
            ],
        )

        hourly_df[hourly_timestamp_col] = pd.to_datetime(
            hourly_df[hourly_timestamp_col]
        )

        hourly_df = hourly_df.sort_values(
            hourly_timestamp_col
        ).reset_index(drop=True)

        if len(hourly_df) < len(df):
            raise ValueError(
                "The FHV hourly demand dataset has fewer rows than "
                "the demand prediction file, so timestamps cannot "
                "be safely recovered."
            )

        # The FHV demand prediction file contains the chronological
        # test-period predictions. Recover the matching final N timestamps.
        recovered_timestamps = (
            hourly_df[hourly_timestamp_col]
            .tail(len(df))
            .reset_index(drop=True)
        )

        df = df.reset_index(drop=True)

        df["__visualization_timestamp"] = recovered_timestamps

        timestamp_col = "__visualization_timestamp"

        print(
            f"  Recovered {len(df):,} hourly timestamps."
        )

    # -------------------------------------------------------------------------
    # Detect actual demand
    # -------------------------------------------------------------------------

    actual_col = find_column(
        df,
        [
            "actual",
            "y_true",
            "actual_demand",
            "demand",
        ],
    )

    # -------------------------------------------------------------------------
    # Detect predicted demand
    # -------------------------------------------------------------------------

    predicted_col = find_column(
        df,
        [
            "predicted",
            "y_pred",
            "predicted_demand",
        ],
    )

    print(
        f"  Timestamp column: {timestamp_col}"
    )

    print(
        f"  Actual column: {actual_col}"
    )

    print(
        f"  Predicted column: {predicted_col}"
    )

    # -------------------------------------------------------------------------
    # Prepare dataframe
    # -------------------------------------------------------------------------

    plot_df = df[
        [
            timestamp_col,
            actual_col,
            predicted_col,
        ]
    ].copy()

    plot_df[timestamp_col] = pd.to_datetime(
        plot_df[timestamp_col]
    )

    # -------------------------------------------------------------------------
    # Create line chart
    # -------------------------------------------------------------------------

    fig = px.line(
        plot_df,
        x=timestamp_col,
        y=[
            actual_col,
            predicted_col,
        ],
        title=(
            "FHV Taxi — Citywide Hourly Demand: "
            "Actual vs Predicted"
        ),
        labels={
            "value": "Trips",
            "variable": "Series",
            timestamp_col: "Time",
        },
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Trips",
    )

    output_file = (
        PLOT_DIR
        / "fhv_task2_demand_actual_vs_predicted.html"
    )

    fig.write_html(
        output_file
    )

    print(
        f"  -> {output_file.name}"
    )


# =============================================================================
# TASK 3 — ZONE DEMAND
# =============================================================================

def plot_zone_demand_task():

    print("\nBuilding FHV Task 3 plots...")

    # -------------------------------------------------------------------------
    # Check file
    # -------------------------------------------------------------------------

    if not ZONE_DEMAND_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Missing FHV zone demand predictions:\n"
            f"{ZONE_DEMAND_PREDICTIONS}"
        )

    # -------------------------------------------------------------------------
    # Load predictions
    # -------------------------------------------------------------------------

    df = pd.read_parquet(
        ZONE_DEMAND_PREDICTIONS
    )

    print(
        f"  Loaded zone predictions: {len(df):,} rows"
    )

    print(
        "  Prediction columns:"
    )

    print(
        f"    {list(df.columns)}"
    )

    # -------------------------------------------------------------------------
    # Detect timestamp
    # -------------------------------------------------------------------------

    timestamp_col = find_column(
        df,
        [
            "pickup_hour_ts",
            "pickup_hour",
            "hour",
            "timestamp",
            "datetime",
        ],
    )

    # -------------------------------------------------------------------------
    # Detect zone ID
    # -------------------------------------------------------------------------

    zone_id_col = find_column(
        df,
        [
            "zone_id",
            "PUlocationID",
            "PULocationID",
            "pickup_zone_id",
        ],
    )

    # -------------------------------------------------------------------------
    # Detect optional zone name
    # -------------------------------------------------------------------------

    zone_name_col = find_column(
        df,
        [
            "zone_name",
            "Zone",
            "pickup_zone",
            "zone",
        ],
        required=False,
    )

    # -------------------------------------------------------------------------
    # Detect optional borough
    # -------------------------------------------------------------------------

    borough_col = find_column(
        df,
        [
            "borough",
            "Borough",
        ],
        required=False,
    )

    # -------------------------------------------------------------------------
    # Detect actual demand
    # -------------------------------------------------------------------------

    actual_col = find_column(
        df,
        [
            "actual",
            "y_true",
            "actual_demand",
            "demand",
        ],
    )

    # -------------------------------------------------------------------------
    # Detect predicted demand
    # -------------------------------------------------------------------------

    predicted_col = find_column(
        df,
        [
            "predicted",
            "y_pred",
            "predicted_demand",
        ],
    )

    print(
        f"  Timestamp column: {timestamp_col}"
    )

    print(
        f"  Zone ID column: {zone_id_col}"
    )

    print(
        f"  Zone name column: {zone_name_col}"
    )

    print(
        f"  Borough column: {borough_col}"
    )

    print(
        f"  Actual column: {actual_col}"
    )

    print(
        f"  Predicted column: {predicted_col}"
    )

    # -------------------------------------------------------------------------
    # Prepare dataframe
    # -------------------------------------------------------------------------

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col]
    )

    # -------------------------------------------------------------------------
    # TOP 15 ZONES
    # -------------------------------------------------------------------------

    group_columns = [
        zone_id_col
    ]

    if zone_name_col:
        group_columns.append(
            zone_name_col
        )

    zone_summary = (
        df.groupby(
            group_columns,
            as_index=False,
        )[actual_col]
        .sum()
    )

    zone_summary = (
        zone_summary
        .sort_values(
            actual_col,
            ascending=False,
        )
        .head(15)
    )

    # -------------------------------------------------------------------------
    # Create readable zone label
    # -------------------------------------------------------------------------

    if zone_name_col:

        zone_summary["zone_label"] = (
            zone_summary[zone_name_col]
            .astype(str)
            + " (#"
            + zone_summary[zone_id_col]
            .astype(str)
            + ")"
        )

    else:

        zone_summary["zone_label"] = (
            "Zone #"
            + zone_summary[zone_id_col]
            .astype(str)
        )

    # -------------------------------------------------------------------------
    # Plot top zones
    # -------------------------------------------------------------------------

    fig = px.bar(
        zone_summary.sort_values(
            actual_col,
            ascending=True,
        ),
        x=actual_col,
        y="zone_label",
        orientation="h",
        title=(
            "FHV Taxi — "
            "Top 15 Pickup Zones by Demand"
        ),
        labels={
            actual_col: "Total Trips",
            "zone_label": "Taxi Zone",
        },
    )

    fig.update_layout(
        xaxis_title="Total Trips",
        yaxis_title="Taxi Zone",
    )

    output_file = (
        PLOT_DIR
        / "fhv_task3_zone_top_demand.html"
    )

    fig.write_html(
        output_file
    )

    print(
        f"  -> {output_file.name}"
    )

    # -------------------------------------------------------------------------
    # BOROUGH TIME SERIES
    # -------------------------------------------------------------------------

    if borough_col:

        borough_df = (
            df.groupby(
                [
                    timestamp_col,
                    borough_col,
                ],
                as_index=False,
            )[actual_col]
            .sum()
        )

        fig = px.line(
            borough_df,
            x=timestamp_col,
            y=actual_col,
            color=borough_col,
            title=(
                "FHV Taxi — "
                "Demand by Borough Over Time"
            ),
            labels={
                timestamp_col: "Time",
                actual_col: "Trips",
                borough_col: "Borough",
            },
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Number of Trips",
        )

        output_file = (
            PLOT_DIR
            / "fhv_task3_zone_borough_timeseries.html"
        )

        fig.write_html(
            output_file
        )

        print(
            f"  -> {output_file.name}"
        )

    else:

        print(
            "  WARNING: Borough information is not "
            "available in the prediction file."
        )

        print(
            "  Borough time-series plot skipped."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("FHV TAXI — INTERACTIVE VISUALIZATION")
    print("=" * 80)

    print(
        "\nOutput directory:"
    )

    print(
        PLOT_DIR
    )

    print()

    # -------------------------------------------------------------------------
    # Task 1
    # -------------------------------------------------------------------------

    plot_duration_task()

    # -------------------------------------------------------------------------
    # Task 2
    # -------------------------------------------------------------------------

    plot_demand_task()

    # -------------------------------------------------------------------------
    # Task 3
    # -------------------------------------------------------------------------

    plot_zone_demand_task()

    # -------------------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FHV TAXI VISUALIZATION COMPLETE")
    print("=" * 80)

    print(
        "\nAll FHV Taxi plots were saved to:"
    )

    print(
        PLOT_DIR
    )

    print(
        "\nGenerated files:"
    )

    for file in sorted(
        PLOT_DIR.glob("*.html")
    ):
        print(
            f"  - {file.name}"
        )

    print(
        "\nOpen any HTML file in your browser to "
        "view the interactive Plotly visualization."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()