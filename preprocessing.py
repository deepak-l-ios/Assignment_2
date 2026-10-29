from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def build_preprocessor():
    """Scale numeric columns and one-hot encode categorical ones; any other column is ignored."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # dense output because GaussianNB cannot take sparse input
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_COLS),
            ("cat", categorical, config.CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
