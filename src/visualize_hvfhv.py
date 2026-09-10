"""
Visualize HVFHV (Uber/Lyft/Via/Juno) model results.

Reads what train_duration_hvfhv.py, train_demand_hvfhv.py, and
train_zone_demand_hvfhv.py already saved to outputs/ and data/processed/,
and writes a set of PNG charts to outputs/figures/.

Usage:
    python src/visualize_hvfhv.py

Notes:
    - Task 1 (duration) currently only saves a metrics .txt + feature
      importances -- no per-row predictions file -- so this script plots
      feature importances + printed metrics for Task 1, and full
      actual-vs-predicted charts for Tasks 2 and 3, which DO save
      predictions parquet files.
    - If you want an actual-vs-predicted scatter for Task 1 too, add this
      right after `preds = model.predict(X_test)` in train_duration_hvfhv.py:

        results = pd.DataFrame({"actual": y_test.values, "predicted": preds})
        results.to_parquet(OUTPUTS_DIR / "hvfhv_duration_predictions.parquet")

      Rerun that script once, and this file will pick the predictions up
      automatically (see load_duration_predictions() below).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def savefig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {path}")


def parse_duration_metrics(path):
    """Parse the plain-text metrics file written by train_duration_hvfhv.py."""
    text = path.read_text()
    lines = text.splitlines()

    metrics = {}
    importances = {}
    in_importances = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Feature importances"):
            in_importances = True
            continue
        if not in_importances:
            if line.startswith("MAE:"):
                metrics["MAE"] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("RMSE:"):
                metrics["RMSE"] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("R2:"):
                metrics["R2"] = float(line.split(":")[1].strip())
        else:
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                name, val = parts
                try:
                    importances[name] = float(val)
                except ValueError:
                    pass
    return metrics, pd.Series(importances).sort_values(ascending=False)


def load_duration_predictions():
    """Optional -- only exists if you added the predictions.to_parquet() line
    described in the module docstring."""
    path = OUTPUTS_DIR / "hvfhv_duration_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# ---------------------------------------------------------------------------
# Task 1 -- Duration
# ---------------------------------------------------------------------------

def plot_task1_duration():
    print("\n[Task 1] Duration")
    metrics_path = OUTPUTS_DIR / "hvfhv_duration_model_metrics.txt"
    if not metrics_path.exists():
        print(f"  skipping -- {metrics_path} not found. Run train_duration_hvfhv.py first.")
        return

    metrics, importances = parse_duration_metrics(metrics_path)

    # Feature importances
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importances.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(
        f"HVFHV Duration Model -- Feature Importances\n"
        f"MAE {metrics.get('MAE', float('nan')):.1f}s | "
        f"RMSE {metrics.get('RMSE', float('nan')):.1f}s | "
        f"R\u00b2 {metrics.get('R2', float('nan')):.3f}"
    )
    savefig(fig, "hvfhv_task1_feature_importances.png")

    preds = load_duration_predictions()
    if preds is None:
        print("  (no per-row predictions file found -- add the "
              "to_parquet() line in train_duration_hvfhv.py for an "
              "actual-vs-predicted scatter)")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(preds["actual"], preds["predicted"], s=4, alpha=0.15, color="#4C72B0")
    lims = [0, max(preds["actual"].max(), preds["predicted"].max())]
    ax.plot(lims, lims, color="crimson", linewidth=1, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual duration (s)")
    ax.set_ylabel("Predicted duration (s)")
    ax.set_title("HVFHV Duration -- Actual vs Predicted")
    ax.legend()
    savefig(fig, "hvfhv_task1_actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# Task 2 -- Citywide hourly demand
# ---------------------------------------------------------------------------

def plot_task2_demand():
    print("\n[Task 2] Citywide demand")
    path = OUTPUTS_DIR / "hvfhv_demand_predictions.parquet"
    if not path.exists():
        print(f"  skipping -- {path} not found. Run train_demand_hvfhv.py first.")
        return

    df = pd.read_parquet(path)
    # index is the pickup_hour_ts (see train_demand_hvfhv.py: results is
    # created with `index=y_test.index`)
    if df.index.name != "pickup_hour_ts" and "pickup_hour_ts" in df.columns:
        df = df.set_index("pickup_hour_ts")
    df = df.sort_index()

    mae = (df["actual"] - df["predicted"]).abs().mean()
    r2 = 1 - ((df["actual"] - df["predicted"]) ** 2).sum() / (
        (df["actual"] - df["actual"].mean()) ** 2
    ).sum()

    # Time series: actual vs predicted over the holdout period
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["actual"], label="Actual", color="#333333", linewidth=1)
    ax.plot(df.index, df["predicted"], label="Predicted", color="#DD8452", linewidth=1)
    ax.set_ylabel("Trips / hour")
    ax.set_title(f"HVFHV Citywide Hourly Demand -- Holdout Period\nMAE {mae:.0f} trips/hr | R\u00b2 {r2:.3f}")
    ax.legend()
    fig.autofmt_xdate()
    savefig(fig, "hvfhv_task2_timeseries.png")

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["actual"], df["predicted"], s=10, alpha=0.4, color="#DD8452")
    lims = [0, max(df["actual"].max(), df["predicted"].max())]
    ax.plot(lims, lims, color="crimson", linewidth=1, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual trips/hour")
    ax.set_ylabel("Predicted trips/hour")
    ax.set_title("HVFHV Citywide Demand -- Actual vs Predicted")
    ax.legend()
    savefig(fig, "hvfhv_task2_actual_vs_predicted.png")

    # Residuals by hour-of-day (is error systematic at certain hours?)
    resid = (df["predicted"] - df["actual"])
    by_hour = resid.groupby(df.index.hour).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_hour.plot.bar(ax=ax, color=np.where(by_hour >= 0, "#DD8452", "#4C72B0"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean residual (predicted - actual)")
    ax.set_title("HVFHV Citywide Demand -- Mean Error by Hour of Day")
    savefig(fig, "hvfhv_task2_residuals_by_hour.png")


# ---------------------------------------------------------------------------
# Task 3 -- Zone-hourly demand
# ---------------------------------------------------------------------------

def plot_task3_zone_demand():
    print("\n[Task 3] Zone demand")
    pred_path = OUTPUTS_DIR / "hvfhv_zone_demand_predictions.parquet"
    if not pred_path.exists():
        print(f"  skipping -- {pred_path} not found. Run train_zone_demand_hvfhv.py first.")
        return

    preds = pd.read_parquet(pred_path)  # columns: pickup_hour_ts, zone_id, actual, predicted

    mae = (preds["actual"] - preds["predicted"]).abs().mean()
    r2 = 1 - ((preds["actual"] - preds["predicted"]) ** 2).sum() / (
        (preds["actual"] - preds["actual"].mean()) ** 2
    ).sum()

    # Overall scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(preds["actual"], preds["predicted"], s=6, alpha=0.2, color="#55A868")
    lims = [0, max(preds["actual"].max(), preds["predicted"].max())]
    ax.plot(lims, lims, color="crimson", linewidth=1, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual trips/zone/hour")
    ax.set_ylabel("Predicted trips/zone/hour")
    ax.set_title(f"HVFHV Zone Demand -- Actual vs Predicted\nMAE {mae:.2f} | R\u00b2 {r2:.3f}")
    ax.legend()
    savefig(fig, "hvfhv_task3_actual_vs_predicted.png")

    # Bring in zone/borough names from the processed data (predictions file
    # only has zone_id, not the name/borough)
    zone_lookup_path = PROCESSED_DIR / "hvfhv_zone_hourly_demand.parquet"
    zone_names = None
    if zone_lookup_path.exists():
        zdf = pd.read_parquet(zone_lookup_path)
        zone_names = zdf[["zone_id", "zone_name", "borough"]].drop_duplicates("zone_id")
        preds = preds.merge(zone_names, on="zone_id", how="left")

    # Top 15 busiest zones by total actual volume in the holdout window
    totals = preds.groupby("zone_id")["actual"].sum().sort_values(ascending=False).head(15)
    labels = totals.index.astype(str)
    if zone_names is not None:
        name_map = zone_names.set_index("zone_id")["zone_name"]
        labels = [f"{name_map.get(z, z)} ({z})" for z in totals.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels[::-1], totals.values[::-1], color="#55A868")
    ax.set_xlabel("Total actual trips (holdout period)")
    ax.set_title("HVFHV -- Top 15 Busiest Zones (Holdout Period)")
    savefig(fig, "hvfhv_task3_top_zones.png")

    # Per-borough MAE, if borough info is available
    if zone_names is not None and "borough" in preds.columns:
        by_borough = (
            preds.assign(abs_err=(preds["actual"] - preds["predicted"]).abs())
            .groupby("borough")["abs_err"]
            .mean()
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        by_borough.plot.barh(ax=ax, color="#8172B2")
        ax.set_xlabel("Mean absolute error (trips/zone/hour)")
        ax.set_title("HVFHV Zone Demand -- MAE by Borough")
        savefig(fig, "hvfhv_task3_mae_by_borough.png")


# ---------------------------------------------------------------------------

def main():
    print("Writing HVFHV visualizations to outputs/figures/ ...")
    plot_task1_duration()
    plot_task2_demand()
    plot_task3_zone_demand()
    print("\nDone.")


if __name__ == "__main__":
    main()
