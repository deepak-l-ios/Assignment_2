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

Evaluated on the held-out [test_data.csv](test_data.csv) (1,000 rows); also saved in [results/metrics.csv](results/metrics.csv). Precision, recall and F1 are for the churn class at the default 0.5 threshold.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7780 | 0.8260 | 0.3618 | 0.7518 | 0.4885 | 0.4084 |
| Decision Tree | 0.9370 | 0.8781 | 0.8611 | 0.6596 | 0.7470 | 0.7200 |
| KNN | 0.8690 | 0.8593 | 0.9167 | 0.0780 | 0.1438 | 0.2456 |
| Naive Bayes | 0.8630 | 0.8365 | 0.5217 | 0.3404 | 0.4120 | 0.3482 |
| Support Vector Machine | 0.9120 | 0.9103 | 0.8533 | 0.4539 | 0.5926 | 0.5828 |

### Observations

The test set has 1,000 customers, 141 of them churners, so predicting "stayed" for everyone would already score 85.9% accuracy. Accuracy is therefore read together with recall, F1 and MCC. Counts below are for the churn class at the default 0.5 threshold.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Catches the most churners (recall 0.752, 106 of 141) but pays for it in false alarms: precision is only 0.362 (about 187 loyal customers flagged), which pulls accuracy down to 0.778. The tuned `class_weight="balanced"` setting trades precision for recall. AUC 0.826 and MCC 0.408 show a linear boundary captures part of the pattern but misses interactions such as international plan combined with heavy day usage. A reasonable choice only if missing a churner is much costlier than a false alarm. |
| Decision Tree | Best balance between precision (0.861) and recall (0.660): 93 churners found with only about 15 false alarms. Highest accuracy (0.937), F1 (0.747) and MCC (0.720) of all five. A shallow tree (depth 5, at least 10 samples per leaf) was enough, which matches the EDA: a few clear rules (international plan, 4 or more customer service calls, high daytime minutes) drive churn. Its AUC (0.878) is lower than SVM's because a tree outputs coarse leaf probabilities, so it ranks customers less finely. |
| KNN | Very conservative: precision 0.917 but recall only 0.078 (11 of 141 churners found), so F1 is 0.144 and MCC 0.246. Accuracy (0.869) is barely above the 0.859 majority-class baseline. With 69 one-hot and scaled inputs and only 14% churners, the 25 nearest neighbours are almost always loyal customers, so few predictions cross 0.5. AUC (0.859) is much better than its F1, showing the ranking is useful and the problem is the threshold, not the model. Weakest model at the default threshold. |
| Naive Bayes | Middle of the pack: recall 0.340, precision 0.522, F1 0.412, MCC 0.348, AUC 0.837. The Gaussian and independence assumptions fit poorly: one-hot columns are binary, and minutes/calls are correlated. Heavy smoothing (`var_smoothing` = 0.1) was needed to get a usable model, and untuned it was far worse (AUC about 0.63). Fast and simple, but it does not match the tree or SVM here. |
| Support Vector Machine | Best ranker: highest AUC (0.910), also the highest cross-validated AUC (0.905), with strong precision (0.853) and accuracy (0.912). Recall is moderate (0.454, 64 of 141), so it misses more churners than the tree and its F1 (0.593) and MCC (0.583) are lower. Its probability scores are good, but the 0.5 cut-off is too strict for the 14% positive class; a lower threshold would likely lift recall at some cost in precision. |
| **Overall winner** | **Decision Tree.** It gives the best accuracy (0.937), F1 (0.747) and MCC (0.720), and the best precision/recall balance, using simple, explainable rules. The SVM is a close second and is the best by AUC (0.910 vs 0.878), so it would be the better pick if scores were ranked or the threshold tuned. Naive Bayes, Logistic Regression and KNN trail. The ranking rests on one test split with 141 churners, so small gaps (for example between the tree and the SVM on cross-validated AUC, 0.901 vs 0.905) should not be over-interpreted. |

## Streamlit app

[app.py](app.py) loads the saved pipelines from `model/` and scores a CSV without retraining.

- **Inputs (sidebar):** CSV upload (tes (evaluate, compare, data tabs)t data only), a classifier dropdown, and a checkbox to fall back to the bundled `test_data.csv` when nothing is uploaded (an uploaded file always takes priority). The file must contain the target column `class` and all 15 feature columns; problems are reported with a clear message.
- **Evaluate tab:** Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model, plus a confusion matrix, ROC curve and classification report.
- **Compare all models tab:** the six metrics for all five models on the same file, with the best value per metric highlighted, and a bar chart.
- **Data tab:** a preview of the file and its class counts.

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
└── results/            # selected hyperparameters and test metrics
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

Deployed on Streamlit Community Cloud: repository `deepak-l-ios/Assignment_2`, branch `main`, main file `app.py`, Python 3.13 (advanced settings). `requirements.txt` pins the scikit-learn, NumPy, SciPy, pandas and joblib versions used to train the saved models so they load unchanged in the cloud.
