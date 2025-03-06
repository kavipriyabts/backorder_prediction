from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, 
    recall_score, balanced_accuracy_score, roc_auc_score
)
from source.entity.artifact_entity import ClassificationMetricArtifact

def calculate_metric(model, x, y) -> ClassificationMetricArtifact:
    """
    Calculates classification metrics for a given model and input features.

    Args:
        model (estimator): The classification model to evaluate.
        x (array-like): Input features.
        y (array-like): True labels.

    Returns:
        ClassificationMetricArtifact: An artifact containing calculated metrics.
    """
    try:
        yhat = model.predict(x)
        yhat_proba = model.predict_proba(x)[:, 1]  # Get predicted probabilities for ROC AUC

        classification_metric = ClassificationMetricArtifact(
            f1_score=f1_score(y, yhat),
            recall_score=recall_score(y, yhat),
            precision_score=precision_score(y, yhat),
            balanced_accuracy_score=balanced_accuracy_score(y, yhat),
            roc_auc_score=roc_auc_score(y, yhat_proba),  # Use predicted probabilities
        )

        return classification_metric

    except Exception as e:
        raise ValueError(f"Error calculating metrics: {str(e)}")

# Uncomment and implement the total_cost function if needed
# def total_cost(y_true, y_pred):
#     """
#     This function takes y_true, y_predicted, and prints Total cost due to misclassification.
#     Args:
#         y_true (array-like): True labels.
#         y_pred (array-like): Predicted labels.
#     Returns:
#         float: Total cost due to misclassification.
#     """
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     cost = 10 * fp + 500 * fn
#     return cost