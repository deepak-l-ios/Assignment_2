import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(model, X, y):
    """Return the six assignment metrics; works for binary and multi-class targets."""
    y_pred = model.predict(X)
    n_classes = len(np.unique(y))
    average = "binary" if n_classes == 2 else "weighted"

    proba = model.predict_proba(X)
    if n_classes == 2:
        auc = roc_auc_score(y, proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y, y_pred, average=average, zero_division=0),
        "F1": f1_score(y, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
