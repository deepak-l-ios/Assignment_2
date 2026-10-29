from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
TRAIN_DATA_PATH = BASE_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = BASE_DIR / "test_data.csv"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

TARGET_COL = "class"
TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_COLS = ["state", "area_code", "international_plan", "voice_mail_plan"]
# identifier plus *_charge columns that are exact multiples of the *_minutes columns
DROPPED_COLS = [
    "phone_number",
    "total_day_charge",
    "total_eve_charge",
    "total_night_charge",
    "total_intl_charge",
]
NUMERIC_COLS = [
    "account_length",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_eve_minutes",
    "total_eve_calls",
    "total_night_minutes",
    "total_night_calls",
    "total_intl_minutes",
    "total_intl_calls",
    "number_customer_service_calls",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
