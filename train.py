"""Train the five classifiers, save them, write test_data.csv and print the metrics table."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
    df = pd.read_csv(config.DATA_PATH)
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET_COL],
        random_state=config.RANDOM_STATE,
    )
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

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
