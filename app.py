import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

import config
from evaluation import compute_metrics

REQUIRED_FEATURES = config.NUMERIC_COLS + config.CATEGORICAL_COLS
CLASS_NAMES = ["Stayed", "Churned"]

st.set_page_config(page_title="Churn model checker", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(config.MODEL_DIR / config.MODEL_FILES[name])


@st.cache_data
def load_bundled_test_data():
    return pd.read_csv(config.TEST_DATA_PATH)


def check_dataset(df):
    """Return an error message if the data cannot be scored, else None."""
    if config.TARGET_COL not in df.columns:
        return f"Target column `{config.TARGET_COL}` (1 = churned, 0 = stayed) is missing."
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        return "Missing feature columns: " + ", ".join(f"`{c}`" for c in missing)
    if not set(df[config.TARGET_COL].dropna().unique()) <= {0, 1}:
        return f"`{config.TARGET_COL}` must contain only 0 and 1."
    if df[config.TARGET_COL].nunique() < 2:
        return "The target column has only one class, so AUC cannot be computed."
    return None


def predict_all(model, X):
    return model.predict(X), model.predict_proba(X)[:, 1]


def confusion_figure(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        annot=True, fmt="d", cmap="Purples", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def roc_figure(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.plot(fpr, tpr, color="tab:purple")
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.tight_layout()
    return fig


st.title("Telecom churn - model checker")
st.write(
    "Score a held-out customer file with one of five pre-trained classifiers. "
    "Nothing is retrained: the saved pipelines are loaded and applied as they are."
)

with st.sidebar:
    st.header("Inputs")
    model_name = st.selectbox("Classifier", list(config.MODEL_FILES))
    uploaded = st.file_uploader("Test data (CSV)", type="csv")
    use_bundled = st.checkbox("Use the bundled test_data.csv", value=uploaded is None)
    st.caption(
        f"The file needs the target column `{config.TARGET_COL}` and these features: "
        + ", ".join(REQUIRED_FEATURES)
    )

if uploaded is not None and not use_bundled:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_bundled:
    data = load_bundled_test_data()
    source = "bundled test_data.csv"
else:
    st.info("Upload a CSV or tick the checkbox to use the bundled test file.")
    st.stop()

problem = check_dataset(data)
if problem:
    st.error(problem)
    st.stop()

features, target = data[REQUIRED_FEATURES], data[config.TARGET_COL].astype(int)
st.caption(f"Scoring **{source}**: {len(data):,} rows, {target.mean():.1%} churned.")

tab_eval, tab_compare, tab_data = st.tabs(["Evaluate", "Compare all models", "Data"])

with tab_eval:
    model = load_model(model_name)
    predicted, scores = predict_all(model, features)
    metrics = compute_metrics(model, features, target)

    st.subheader(model_name)
    for col, (label, value) in zip(st.columns(len(metrics)), metrics.items()):
        col.metric(label, f"{value:.4f}")

    left, middle, right = st.columns([1, 1, 1.2])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(confusion_figure(target, predicted))
    with middle:
        st.markdown("**ROC curve**")
        st.pyplot(roc_figure(target, scores))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            target, predicted, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3))

with tab_compare:
    table = pd.DataFrame(
        {name: compute_metrics(load_model(name), features, target) for name in config.MODEL_FILES}
    ).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#d9c8f5"))
    st.caption("Best value per metric is highlighted.")
    st.bar_chart(table.T)

with tab_data:
    st.dataframe(data.head(20))
    st.write("Class counts")
    st.bar_chart(target.map(dict(enumerate(CLASS_NAMES))).value_counts())
