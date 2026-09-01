import joblib
import pandas as pd
from pathlib import Path


FEATURE_COLUMNS = [
    "trip_distance",
    "passenger_count",
    "pickup_location_id",
    "dropoff_location_id",
    "vendor_id",
    "rate_code_id",
    "pickup_hour",
    "pickup_day",
    "pickup_month"
]


def load_model():
    """Load the trained Random Forest model."""

    model_path = Path("outputs/models/random_forest_baseline.joblib")

    return joblib.load(model_path)


def predict_trip_duration(model, trip_data):
    """Predict taxi trip duration in minutes."""

    trip_data = trip_data[FEATURE_COLUMNS]

    prediction = model.predict(trip_data)

    return prediction


if __name__ == "__main__":

    model = load_model()

        # Test with multiple taxi trips
    trips = pd.DataFrame([
        {
            "trip_distance": 3.5,
            "passenger_count": 1,
            "pickup_location_id": 100,
            "dropoff_location_id": 200,
            "vendor_id": 1,
            "rate_code_id": 1,
            "pickup_hour": 18,
            "pickup_day": 3,
            "pickup_month": 1
        },
        {
            "trip_distance": 1.2,
            "passenger_count": 2,
            "pickup_location_id": 50,
            "dropoff_location_id": 80,
            "vendor_id": 1,
            "rate_code_id": 1,
            "pickup_hour": 9,
            "pickup_day": 2,
            "pickup_month": 1
        },
        {
            "trip_distance": 8.0,
            "passenger_count": 3,
            "pickup_location_id": 150,
            "dropoff_location_id": 250,
            "vendor_id": 2,
            "rate_code_id": 1,
            "pickup_hour": 22,
            "pickup_day": 5,
            "pickup_month": 1
        }
    ])

    # Make predictions
predictions = predict_trip_duration(model, trips)

# Add predictions to the DataFrame
trips["predicted_duration_minutes"] = predictions

# Display results
print("\nPrediction Results")
print("==================")
print(trips)

# Create output directory
output_dir = Path("outputs/predictions")
output_dir.mkdir(parents=True, exist_ok=True)

# Save predictions
output_path = output_dir / "sample_predictions.csv"

trips.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
# Prediction summary
print("\nPrediction Summary")
print("===================")
print(f"Number of trips: {len(trips)}")
print(
    f"Average predicted duration: "
    f"{trips['predicted_duration_minutes'].mean():.2f} minutes"
)
print(
    f"Minimum predicted duration: "
    f"{trips['predicted_duration_minutes'].min():.2f} minutes"
)
print(
    f"Maximum predicted duration: "
    f"{trips['predicted_duration_minutes'].max():.2f} minutes"
)