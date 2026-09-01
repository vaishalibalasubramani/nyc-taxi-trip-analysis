from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np


def evaluate_model(y_test, y_pred):
    """
    Calculate regression model evaluation metrics.
    """

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "mae_minutes": round(mae, 4),
        "rmse_minutes": round(rmse, 4),
        "r2_score": round(r2, 4)
    }