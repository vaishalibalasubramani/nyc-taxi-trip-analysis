import matplotlib.pyplot as plt


def plot_feature_importance(feature_importance):
    """
    Plot Random Forest feature importance.
    """

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance["feature"],
        feature_importance["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plot actual vs predicted trip duration.
    """

    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.3
    )

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Trip Duration (minutes)")
    plt.ylabel("Predicted Trip Duration (minutes)")
    plt.title("Actual vs Predicted Trip Duration")

    plt.tight_layout()
    plt.show()