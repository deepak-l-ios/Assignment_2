# ML Assignment 2 - Classification Models and Streamlit Deployment

## a. Problem statement

Telecom operators lose revenue when customers leave (churn), and keeping a customer is cheaper than acquiring a new one. Given a customer's account details, plan subscriptions, call usage and customer service history, the task is to predict whether the customer will churn (binary classification: 1 = churned, 0 = stayed). Five classifiers are trained on the same data and compared on six metrics; the best one is exposed through a Streamlit app.

## b. Dataset description

| Item | Value |
|---|---|
| Source | OpenML `churn` (ID 40701), public licence: https://www.openml.org/d/40701 |
| Instances | 5,000 |
| Features | 20 (plus the target) |
| Target | `class` (1 = churned, 0 = stayed) |
| Task type | Binary classification |
| Class distribution | 4,293 stayed (85.9%), 707 churned (14.1%) |
| Missing values / duplicates | none / none |

Feature groups:

| Group | Columns |
|---|---|
| Account | `state`, `account_length`, `area_code`, `phone_number` (identifier, dropped) |
| Plans | `international_plan`, `voice_mail_plan`, `number_vmail_messages` |
| Day / evening / night usage | `total_*_minutes`, `total_*_calls`, `total_*_charge` |
| International usage | `total_intl_minutes`, `total_intl_calls`, `total_intl_charge` |
| Support | `number_customer_service_calls` |

Key findings from the exploratory analysis ([notebooks/eda.ipynb](notebooks/eda.ipynb)):

- Churn is much higher with an international plan (42% vs 11%) and with 4 or more customer service calls (44-65% vs about 11%).
- Customers without a voice mail plan churn more (16.5% vs 7.7%); area code carries almost no signal.
- Each `*_charge` column is perfectly correlated with its `*_minutes` column.
- Because of the 14% churn rate, accuracy alone is misleading, so AUC, F1 and MCC are emphasised.

Preprocessing ([preprocessing.py](preprocessing.py), built into every saved model so uploaded raw CSVs need no manual preparation):

- Dropped: `phone_number` (identifier) and the four `*_charge` columns (exact multiples of the matching `*_minutes` columns).
- Numeric (11 columns): median imputation, then standard scaling.
- Categorical (`state`, `area_code`, `international_plan`, `voice_mail_plan`): most-frequent imputation, then one-hot encoding (unknown categories ignored), giving 69 model inputs in total.
- Train/test split ([split_data.py](split_data.py)): stratified 80/20 with a fixed seed. The 4,000-row training file ([data/train_data.csv](data/train_data.csv)) is the only data used for fitting; the 1,000-row [test_data.csv](test_data.csv) (141 churners, 14.1%) is never seen during training and is what the app evaluates on.

## c. GitHub repository

https://github.com/deepak-l-ios/Assignment_2

Streamlit app: TODO (add link after deployment)

## d. Models used

### Training and tuning

Each model is a scikit-learn pipeline (preprocessing + classifier) tuned with `GridSearchCV` on the training split only: 5-fold stratified cross-validation, scored on ROC AUC (threshold-independent, so suitable for the 14% churn rate). The tuned pipelines are saved in [model/](model/) and the selected hyperparameters in [results/best_params.json](results/best_params.json).

| Model | Parameters searched |
|---|---|
| Logistic Regression | `C` (0.01-10), `class_weight` |
| Decision Tree | `max_depth` (3-12, none), `min_samples_leaf`, `class_weight` |
| KNN | `n_neighbors` (3-25), `weights` |
| Naive Bayes (Gaussian) | `var_smoothing` (1e-9 to 1) |
| Support Vector Machine (RBF) | `C` (0.1-10), `gamma`, `class_weight` |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner** | |

## Project structure

```
.
├── app.py              # Streamlit app
├── config.py           # paths, column roles, split settings
├── preprocessing.py    # shared preprocessing (scaling, one-hot encoding)
├── evaluation.py       # metric computation
├── split_data.py       # stratified 80/20 split into train and test files
├── train.py            # tunes models on the train split, saves them, prints test metrics
├── requirements.txt
├── test_data.csv       # held-out test split (1,000 rows, used by the app)
├── notebooks/eda.ipynb # exploratory data analysis
├── data/dataset.csv    # full raw dataset
├── data/train_data.csv # training split (4,000 rows)
├── model/              # saved .joblib pipelines (one per classifier)
└── results/            # selected hyperparameters
```

## Setup and usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. data/dataset.csv holds the full dataset; TARGET_COL is set in config.py
# 2. split into train_data.csv and test_data.csv (fixed seed, already committed)
python split_data.py

# 3. train, save models and print test-set metrics
python train.py

# 4. run the app
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud: select this repo, branch `master`, main file `app.py`.
