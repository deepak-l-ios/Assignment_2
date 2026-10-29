"""Tune the five classifiers with cross-validation on the training split, save them and print test-set metrics."""
import json
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from evaluation import compute_metrics
from preprocessing import build_preprocessor

CV_FOLDS = 5
SEARCH_SCORING = "roc_auc"


def build_search_space():
    """Return {model name: (estimator, parameter grid)}."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            {
                "clf__max_depth": [3, 5, 8, 12, None],
                "clf__min_samples_leaf": [1, 5, 10],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 9, 15, 25], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 1.0]},
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=config.RANDOM_STATE),
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01],
                "clf__class_weight": [None, "balanced"],
            },
        ),
    }


def main():
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    X_train, y_train = train_df.drop(columns=config.TARGET_COL), train_df[config.TARGET_COL]
    X_test, y_test = test_df.drop(columns=config.TARGET_COL), test_df[config.TARGET_COL]

    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    rows, best_params = {}, {}
    for name, (estimator, grid) in build_search_space().items():
        started = time.time()
        pipeline = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipeline, grid, scoring=SEARCH_SCORING, cv=folds, n_jobs=-1)
        search.fit(X_train, y_train)

        joblib.dump(search.best_estimator_, config.MODEL_DIR / config.MODEL_FILES[name])
        rows[name] = compute_metrics(search.best_estimator_, X_test, y_test)
        best_params[name] = {
            "params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
            f"cv_{SEARCH_SCORING}": round(float(search.best_score_), 4),
        }
        print(f"{name}: cv {SEARCH_SCORING} {search.best_score_:.4f} "
              f"{search.best_params_} ({time.time() - started:.0f}s)")

    with open(config.RESULTS_DIR / "best_params.json", "w") as fh:
        json.dump(best_params, fh, indent=2)

    table = pd.DataFrame(rows).T.round(4)
    print()
    print(table.to_markdown())


if __name__ == "__main__":
    main()
