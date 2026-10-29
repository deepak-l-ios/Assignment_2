"""Train the five classifiers on the training split, save them and print test-set metrics."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from evaluation import compute_metrics
from preprocessing import build_preprocessor


def build_models():
    def pipe(clf):
        return Pipeline([("prep", build_preprocessor()), ("clf", clf)])

    return {
        "Logistic Regression": pipe(LogisticRegression(max_iter=1000)),
        "Decision Tree": pipe(DecisionTreeClassifier(random_state=config.RANDOM_STATE)),
        "KNN": pipe(KNeighborsClassifier()),
        "Naive Bayes": pipe(GaussianNB()),
        "Support Vector Machine": pipe(SVC(probability=True, random_state=config.RANDOM_STATE)),
    }


def main():
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    X_train, y_train = train_df.drop(columns=config.TARGET_COL), train_df[config.TARGET_COL]
    X_test, y_test = test_df.drop(columns=config.TARGET_COL), test_df[config.TARGET_COL]

    config.MODEL_DIR.mkdir(exist_ok=True)
    rows = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, config.MODEL_DIR / config.MODEL_FILES[name])
        rows[name] = compute_metrics(model, X_test, y_test)

    table = pd.DataFrame(rows).T.round(4)
    print(table.to_markdown())


if __name__ == "__main__":
    main()
