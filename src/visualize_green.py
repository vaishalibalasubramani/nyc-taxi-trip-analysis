"""
Generate interactive Plotly visualizations for all three tasks. Saves each
as a standalone .html file in outputs/plots/ -- double-click any of them to
open in your browser, no server needed.

Run this AFTER all three training scripts, since it reads their saved
outputs.

Usage:
    python src/visualize.py
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

FEATURE_COLS_DURATION = [
    "trip_distance", "passenger_count", "PULocationID", "DOLocationID",
    "trip_type", "payment_type", "pickup_hour", "pickup_dow",
    "is_weekend", "pickup_month",
]


def plot_duration_task():
    print("Building Task 1 plots (trip duration)...")
    model = joblib.load(OUTPUTS_DIR / "duration_model.joblib")
    df = pd.read_parquet(PROCESSED_DIR / "trip_features.parquet").dropna(
        subset=FEATURE_COLS_DURATION + ["trip_duration_s"]
    )

    # Sample for plotting -- no need to render 500K points
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    preds = model.predict(sample[FEATURE_COLS_DURATION])

    fig = px.scatter(
        x=sample["trip_duration_s"], y=preds,
        labels={"x": "Actual duration (s)", "y": "Predicted duration (s)"},
        title="Task 1: Trip Duration -- Actual vs Predicted (5,000-trip sample)",
        opacity=0.4,
    )
    max_val = max(sample["trip_duration_s"].max(), preds.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        line=dict(dash="dash", color="red"), name="Perfect prediction",
    ))
    fig.write_html(PLOTS_DIR / "task1_duration_actual_vs_predicted.html")

    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS_DURATION
    ).sort_values(ascending=True)
    fig2 = px.bar(
        x=importances.values, y=importances.index, orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        title="Task 1: Trip Duration -- Feature Importance",
    )
    fig2.write_html(PLOTS_DIR / "task1_duration_feature_importance.html")
    print("  -> task1_duration_actual_vs_predicted.html")
    print("  -> task1_duration_feature_importance.html")


def plot_demand_task():
    print("Building Task 2 plots (citywide demand)...")
    results = pd.read_parquet(OUTPUTS_DIR / "demand_predictions.parquet")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=results["actual"], name="Actual", mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=results.index, y=results["predicted"], name="Predicted", mode="lines",
    ))
    fig.update_layout(
        title="Task 2: Citywide Hourly Demand -- Actual vs Predicted (holdout period)",
        xaxis_title="Hour", yaxis_title="Trips",
    )
    fig.write_html(PLOTS_DIR / "task2_demand_actual_vs_predicted.html")
    print("  -> task2_demand_actual_vs_predicted.html")


def plot_zone_task():
    print("Building Task 3 plots (zone demand)...")
    zone_lookup_cols = ["pickup_hour_ts", "zone_id"]
    zone_meta = (
        pd.read_parquet(PROCESSED_DIR / "zone_hourly_demand.parquet")
        [["zone_id", "zone_name", "borough"]]
        .drop_duplicates("zone_id")
    )

    results = pd.read_parquet(OUTPUTS_DIR / "zone_demand_predictions.parquet")
    results = results.merge(zone_meta, on="zone_id", how="left")

    # Top 15 zones by average predicted demand
    top_zones = (
        results.groupby(["zone_id", "zone_name"])["predicted"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    fig = px.bar(
        top_zones, x="predicted", y="zone_name", orientation="h",
        labels={"predicted": "Avg predicted trips/hour", "zone_name": "Zone"},
        title="Task 3: Top 15 Zones by Predicted Demand (holdout period average)",
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig.write_html(PLOTS_DIR / "task3_zone_top_demand.html")

    # Demand by borough over time (aggregated hourly)
    borough_ts = (
        results.groupby(["pickup_hour_ts", "borough"])["predicted"]
        .sum()
        .reset_index()
    )
    fig2 = px.line(
        borough_ts, x="pickup_hour_ts", y="predicted", color="borough",
        title="Task 3: Predicted Demand by Borough Over Time (holdout period)",
        labels={"predicted": "Predicted trips/hour", "pickup_hour_ts": "Hour"},
    )
    fig2.write_html(PLOTS_DIR / "task3_zone_borough_timeseries.html")
    print("  -> task3_zone_top_demand.html")
    print("  -> task3_zone_borough_timeseries.html")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_duration_task()
    plot_demand_task()
    plot_zone_task()
    print(f"\nAll plots saved to {PLOTS_DIR}")
    print("Open any .html file in your browser to view it interactively.")


if __name__ == "__main__":
    main()
