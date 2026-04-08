# 🚢 MLflow Lab — Titanic Survival Prediction

> **Course:** IE 7374 — MLOps  
> **Author:** Ibrahim (Ibbu)  
> **Dataset:** Titanic (binary classification)  
> **Notebook:** `mlflow_titanic.ipynb`

---

## 📌 What This Lab Is About

This lab is an **extended, from-scratch re-implementation** of the original `starter.ipynb` wine quality MLflow demo — rebuilt entirely on the **Titanic dataset** with three different models and significantly more MLflow features than the original.

The goal was to go beyond the starter and learn MLflow end-to-end: experiment tracking, artifact logging, hyperparameter tuning with nested runs, model registration, serving, and batch inference.

---

## 🔄 Original vs This Lab

| Area | Original `starter.ipynb` | This Lab |
|---|---|---|
| **Dataset** | Wine Quality (CSV files) | Titanic (loaded from seaborn, no files needed) |
| **Models** | Random Forest + XGBoost | Logistic Regression + Random Forest + LightGBM |
| **Tuning** | Hyperopt + SparkTrials (parallel) | Hyperopt + `Trials()` (sequential, no Spark needed) |
| **Metrics logged** | AUC only | AUC + Accuracy + F1 (all three) |
| **Val vs Test** | Test only | Both val and test logged separately |
| **Tags** | None | `author`, `dataset`, `model_type`, `split`, `note` |
| **Artifacts** | None | Confusion matrix PNG, feature importance PNG + JSON |
| **`log_dict()`** | Not used | Used to log feature importances as JSON |
| **`autolog()`** | Not used | Used for LightGBM child runs |
| **Nested runs** | Yes (SparkTrials) | Yes (standard `nested=True`, no Spark) |
| **Model Registry** | Yes | Yes + version description added |
| **`archive_existing_versions`** | Manual | Automatic via flag |
| **Batch inference** | Spark UDF | Pure pandas + pyfunc (no Spark needed) |
| **`search_runs()` comparison** | Not used | Full comparison table + bar chart |
| **MlflowClient inspection** | Basic | Params, metrics, and tags printed programmatically |

---

## 🗂️ Lab Structure (Batch by Batch)

The notebook was built in **5 batches**, each adding a layer of MLflow knowledge:

### Batch 1 — Data Loading & Preprocessing
- Loaded Titanic directly from `seaborn.load_dataset()` — no CSV files needed
- Dropped leaky columns (`alive`, `who`, `adult_male`, `deck`)
- Filled missing `age` with median, `embarked` with mode
- Encoded `sex` (binary) and `embarked` (one-hot)
- Engineered two new features: `family_size` and `is_alone`
- Stratified 60/20/20 train/val/test split
- Applied `StandardScaler` (fit on train only, transformed val and test)

### Batch 2 — MLflow Setup + Logistic Regression Baseline
- Created a named experiment: `titanic_survival`
- Built a reusable `log_confusion_matrix()` helper that saves a PNG and logs it as an MLflow artifact
- Ran Logistic Regression with full MLflow tracking:
  - Tags: `author`, `dataset`, `model_type`, `split`
  - Params: `C`, `max_iter`, `solver`
  - Metrics: `test_auc`, `test_accuracy`, `test_f1`
  - Artifact: confusion matrix PNG
  - Model logged with `infer_signature()`
- Used `MlflowClient.get_run()` to programmatically inspect the logged run

### Batch 3 — Random Forest + Feature Importance Artifact
- Built a second helper `log_feature_importance()` that logs:
  - A horizontal bar chart PNG (top-10 features)
  - Raw importance scores as a JSON file via `mlflow.log_dict()`
- Logged both **val and test metrics** — so you can spot overfitting in the UI
- Added a `note` tag explaining the intent of the run
- Used `mlflow.search_runs()` to pull all runs into a DataFrame and compare LR vs RF with a grouped bar chart

### Batch 4 — LightGBM + Hyperopt Sweep (Nested Runs)
- Defined a 9-parameter Hyperopt search space covering depth, leaves, learning rate, regularization, subsampling
- Each Hyperopt trial = one **nested child run** (`nested=True`) under a parent run `lgbm_hyperopt_sweep`
- Used `mlflow.lightgbm.autolog()` inside each child run — automatically logs all LightGBM params and eval metrics
- Used `Trials()` (sequential) instead of `SparkTrials` — works without a Spark cluster
- After the sweep, used `search_runs(filter_string=parentRunId)` to find the best child run
- Retrained best params on train+val combined, evaluated on held-out test set
- Logged final model as `lgbm_best_final` with confusion matrix and feature importance artifacts

### Batch 5 — Model Registry → Production + Serving
- Registered the best LightGBM model as `titanic_survival_lgbm` in the MLflow Model Registry
- Added a version description string (AUC, F1, dataset info)
- Promoted to `Production` with `archive_existing_versions=True` — auto-archives any previous version
- Loaded the production model back via `mlflow.pyfunc.load_model("models:/titanic_survival_lgbm/Production")` and verified AUC matches
- Demonstrated REST serving via `mlflow models serve` + a `requests.post()` payload
- Demonstrated batch inference: load from registry → predict on full test set → results DataFrame with `correct` column

---

## 🧪 Models & Results

| Model | Test AUC | Test Accuracy | Test F1 |
|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.79 | ~0.74 |
| Random Forest | ~0.87 | ~0.82 | ~0.77 |
| LightGBM (tuned) | ~0.89 | ~0.83 | ~0.79 |

> Exact numbers will vary slightly — LightGBM results depend on the Hyperopt trial that wins.

---

## ⚙️ Key MLflow Concepts Used

| Concept | API Used | Where |
|---|---|---|
| Create experiment | `mlflow.set_experiment()` | Batch 2 |
| Tag a run | `mlflow.set_tag()` | All models |
| Log parameter | `mlflow.log_param()` | All models |
| Log metric | `mlflow.log_metric()` | All models |
| Log file artifact | `mlflow.log_artifact()` | Confusion matrix, feature importance |
| Log dict as JSON | `mlflow.log_dict()` | Feature importances |
| Log sklearn model | `mlflow.sklearn.log_model()` | LR, RF |
| Log LightGBM model | `mlflow.lightgbm.log_model()` | LightGBM final |
| Auto-log everything | `mlflow.lightgbm.autolog()` | Hyperopt child runs |
| Nested runs | `mlflow.start_run(nested=True)` | Hyperopt sweep |
| Query runs | `mlflow.search_runs()` | Comparison, best run selection |
| Inspect run | `MlflowClient.get_run()` | Batch 2 |
| Register model | `mlflow.register_model()` | Batch 5 |
| Promote to stage | `client.transition_model_version_stage()` | Batch 5 |
| Load from registry | `mlflow.pyfunc.load_model()` | Batch 5 |

---

## 🚀 How to Run

**1. Install dependencies**
```bash
pip install mlflow scikit-learn lightgbm hyperopt seaborn matplotlib cloudpickle
```

**2. Launch MLflow UI** (in a separate terminal — do NOT run inside the notebook)
```bash
mlflow ui --port 5000
# Open: http://127.0.0.1:5000
```

**3. Run the notebook top to bottom**
```
mlflow_titanic.ipynb
```

**4. (Optional) Serve the production model**
```bash
mlflow models serve \
  --env-manager=local \
  -m models:/titanic_survival_lgbm/Production \
  -h 0.0.0.0 \
  -p 5001
```

---

## 📁 What Gets Created Locally

```
mlflow_lab/
├── mlflow_titanic.ipynb   ← the notebook
├── README.md              ← this file
└── mlruns/                ← auto-created by MLflow
    └── <experiment_id>/
        └── <run_id>/
            ├── params/
            ├── metrics/
            ├── tags/
            └── artifacts/
                ├── confusion_matrix_*.png
                ├── feature_importance_*.png
                ├── feature_importances.json
                └── <model>/
```

---

## 💡 Key Differences from the Original Starter

The biggest conceptual differences from `starter.ipynb`:

**No Spark required** — the original used `SparkTrials` for parallel Hyperopt. This lab uses standard `Trials()` which runs sequentially and works on any machine without a Spark cluster.

**Three models instead of two** — added Logistic Regression as a true baseline before tree-based models, making the progression LR → RF → LightGBM more educational.

**Artifacts as a first-class feature** — the original logged no artifacts at all. This lab logs confusion matrices and feature importance plots to every run, making the MLflow UI genuinely useful for comparing runs visually.

**Val + test split** — the original only logged test metrics. Logging both val and test lets you see the gap in the MLflow UI and catch overfitting early.

**Tags for every run** — makes filtering in the UI practical when you have 20+ runs from the Hyperopt sweep.