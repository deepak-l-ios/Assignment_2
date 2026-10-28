import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

import config
from evaluation import compute_metrics

st.set_page_config(page_title="Classifier Evaluation", layout="wide")
st.title("Classification Model Evaluation")
st.caption("Upload a test CSV (features + target column) and evaluate a pre-trained model.")


@st.cache_resource
def load_model(name):
    return joblib.load(config.MODEL_DIR / config.MODEL_FILES[name])


model_name = st.sidebar.selectbox("Model", list(config.MODEL_FILES))
uploaded = st.sidebar.file_uploader("Test data (CSV)", type="csv")

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)
if config.TARGET_COL not in df.columns:
    st.error(f"Target column '{config.TARGET_COL}' not found in the uploaded file.")
    st.stop()

X, y = df.drop(columns=config.TARGET_COL), df[config.TARGET_COL]
model = load_model(model_name)

try:
    metrics = compute_metrics(model, X, y)
except Exception as exc:
    st.error(f"Could not evaluate: {exc}")
    st.stop()

st.subheader(f"Metrics - {model_name}")
cols = st.columns(len(metrics))
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
y_pred = model.predict(X)

with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))

with st.expander("Preview uploaded data"):
    st.dataframe(df.head())
