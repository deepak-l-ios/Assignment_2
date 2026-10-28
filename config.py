from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"  # TODO: place the full raw dataset here
TEST_DATA_PATH = BASE_DIR / "test_data.csv"
MODEL_DIR = BASE_DIR / "model"

TARGET_COL = "target"  # TODO: set to the dataset's label column
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
