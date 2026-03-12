# ELK Stack Lab — ML Experiment Tracking for Credit Card Fraud Detection

**Author:** Ibrahim Mohammed  
**Course:** IE 7374 MLOps — Prof. Ramin Mohammadi  
**Date:** March 12, 2026

---

## Overview

This lab extends the standard ELK (Elasticsearch, Logstash, Kibana) setup lab by building a **real ML experiment tracking pipeline**. Instead of ingesting generic log data, this project trains **four classification models** on a synthetic credit card fraud detection dataset and streams all experiment metrics — training times, evaluation scores, hyperparameters, feature importances, and confusion matrix results — into the ELK stack for visualization and comparison in Kibana.

### What's Different From the Original Lab

The original lab focuses on installing and configuring the ELK stack with basic log ingestion. This implementation goes further by:

- **Training 4 ML models** (SVM, RandomForest, XGBoost, LightGBM) with GridSearchCV hyperparameter tuning instead of using a single pre-built model
- **Handling class imbalance** using SMOTE oversampling on a highly imbalanced fraud dataset (only 5.94% fraud cases)
- **Structured JSON logging** — every stage of the ML pipeline (dataset creation, preprocessing, training, evaluation, model saving) emits structured JSON events that Logstash can parse directly
- **Rich metrics tracking** — logs accuracy, precision, recall, F1, ROC-AUC, average precision, Matthews Correlation Coefficient (MCC), confusion matrix components, cross-validation scores, feature importances, and training times
- **End-to-end pipeline** — the training script writes to `training.log`, Logstash reads and transforms the data, Elasticsearch indexes it, and Kibana visualizes model comparisons

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────┐
│  model.py    │────▶│ training.log │────▶│   Logstash    │────▶│  Elastic │
│ (Train 4     │     │ (JSON lines) │     │ (Parse/Filter)│     │  search  │
│  models)     │     └──────────────┘     └───────────────┘     └────┬─────┘
└──────────────┘                                                     │
                                                                     ▼
                                                               ┌──────────┐
                                                               │  Kibana  │
                                                               │(Dashboards)│
                                                               └──────────┘
```

---

## Project Structure

```
ELK-LAB/
├── model.py                 # ML training script (SVM, RF, XGBoost, LightGBM)
├── logstash.conf            # Logstash pipeline configuration
├── training.log             # JSON experiment logs (auto-generated)
├── readme.md                # This file
├── requirements.txt         # Python dependencies
└── saved_models/            # Serialized best models (.pkl files)
    ├── svm_best.pkl
    ├── randomforest_best.pkl
    ├── xgboost_best.pkl
    └── lightgbm_best.pkl
```

---

## Prerequisites

- **Windows 11** with WSL2 and Ubuntu 24.04 installed
- **Python 3.12+** (comes with Ubuntu 24.04)
- **Elasticsearch 9.3.1** downloaded and extracted in `~/elasticsearch-9.3.1`
- **Logstash 9.3.1** downloaded and extracted in `~/logstash-9.3.1`
- **Kibana 9.3.1** downloaded and configured

---

## Step-by-Step Setup and Execution

### Step 1: Install WSL2 and Ubuntu

Open PowerShell as Administrator:

```powershell
wsl --install -d ubuntu-24.04
```

Create a Unix user and password when prompted. Open the Ubuntu terminal from the Start Menu or by typing `wsl` in PowerShell.

### Step 2: Install Java (Required for ELK Stack)

```bash
sudo apt update
sudo apt install openjdk-17-jre-headless -y
java --version
```

### Step 3: Download and Start Elasticsearch

```bash
cd ~
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.3.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-9.3.1-linux-x86_64.tar.gz
cd elasticsearch-9.3.1
bin/elasticsearch
```

Wait for the cluster health to turn GREEN. Note the auto-generated password for the `elastic` user from the first startup output. Verify in a new terminal:

```bash
curl -k -u elastic:<YOUR_PASSWORD> https://localhost:9200
```

### Step 4: Download and Start Kibana

```bash
cd ~
wget https://artifacts.elastic.co/downloads/kibana/kibana-9.3.1-linux-x86_64.tar.gz
tar -xzf kibana-9.3.1-linux-x86_64.tar.gz
cd kibana-9.3.1
bin/kibana
```

Access Kibana at `https://localhost:5601` in your browser.

### Step 5: Download Logstash

```bash
cd ~
wget https://artifacts.elastic.co/downloads/logstash/logstash-9.3.1-linux-x86_64.tar.gz
tar -xzf logstash-9.3.1-linux-x86_64.tar.gz
```

### Step 6: Install Python Dependencies

```bash
pip3 install scikit-learn xgboost lightgbm imbalanced-learn pandas numpy joblib --break-system-packages
```

### Step 7: Run the ML Training Script

Navigate to the project directory (WSL can access Windows files via `/mnt/c/`):

```bash
cd /mnt/c/Users/<your-username>/Downloads/MLOPS-LAB/ELK-LAB
python3 model.py
```

This will:
- Generate a synthetic fraud dataset (10,000 samples, 5.94% fraud rate)
- Apply SMOTE to balance the training set
- Train SVM, RandomForest, XGBoost, and LightGBM with GridSearchCV
- Log all metrics as JSON lines to `training.log`
- Save the best model for each algorithm to `saved_models/`

### Step 8: Update Logstash Config

Edit `logstash.conf` and update:
- The `path` in the input section to match your WSL file path
- The `password` in the output section to your Elasticsearch password

**Important:** The file path must use WSL format (`/mnt/c/...`), not Windows format (`C:\...`).

### Step 9: Start Logstash

```bash
cd ~/logstash-9.3.1
bin/logstash -f /mnt/c/Users/<your-username>/Downloads/MLOPS-LAB/ELK-LAB/logstash.conf
```

You should see the JSON events printed in rubydebug format as Logstash ingests them.

### Step 10: Visualize in Kibana

1. Open `https://localhost:5601`
2. Go to **Stack Management → Data Views**
3. Create a new data view with pattern: `ml-experiments-*`
4. Go to **Discover** to explore all logged events
5. Go to **Dashboard** to create visualizations:
   - Bar chart comparing F1 scores across models
   - Table with full metrics comparison
   - Pie chart of confusion matrix breakdown per model

---

## Dataset Details

The training script generates a synthetic credit card fraud dataset using `sklearn.make_classification`:

| Property | Value |
|---|---|
| Total Samples | 10,000 |
| Features | 20 |
| Fraud Cases | 594 (5.94%) |
| Legitimate Cases | 9,406 (94.06%) |
| Train/Test Split | 80/20 with stratification |
| SMOTE Applied | Yes (train set: 8,000 → 15,050) |
| Scaling | StandardScaler |

Features simulate real fraud detection attributes like transaction amount, frequency, distance from home, merchant risk score, card age, and velocity metrics.

---

## Models and Results

All models were trained with **3-fold GridSearchCV** optimizing for F1 score, then evaluated on the held-out test set.

### SVM (Support Vector Machine)

| Metric | Value |
|---|---|
| Best Params | C=10.0, kernel=rbf, gamma=scale |
| Accuracy | 0.9555 |
| Precision | 0.6596 |
| Recall | 0.5210 |
| F1 Score | 0.5822 |
| ROC-AUC | 0.8901 |
| MCC | 0.5633 |
| Training Time | 206.5s |
| CV F1 Mean | 0.9886 |

### RandomForest

| Metric | Value |
|---|---|
| Best Params | n_estimators=200, max_depth=None, min_samples_split=2 |
| Accuracy | 0.9605 |
| Precision | 0.7000 |
| Recall | 0.5882 |
| F1 Score | 0.6393 |
| ROC-AUC | 0.8867 |
| MCC | 0.6212 |
| Training Time | 46.9s |
| CV F1 Mean | 0.9777 |

### XGBoost

| Metric | Value |
|---|---|
| Best Params | learning_rate=0.1, max_depth=8, n_estimators=200, scale_pos_weight=1 |
| Accuracy | 0.9630 |
| Precision | 0.7027 |
| Recall | 0.6555 |
| F1 Score | 0.6783 |
| ROC-AUC | 0.8882 |
| MCC | 0.6591 |
| Training Time | 16.2s |
| CV F1 Mean | 0.9835 |

### LightGBM

*Training was initiated but results were not captured in the log for this run. Re-run `model.py` to get complete results.*

### Model Comparison Summary

| Model | F1 Score | ROC-AUC | Recall | MCC | Training Time |
|---|---|---|---|---|---|
| **XGBoost** | **0.6783** | 0.8882 | **0.6555** | **0.6591** | **16.2s** |
| RandomForest | 0.6393 | 0.8867 | 0.5882 | 0.6212 | 46.9s |
| SVM | 0.5822 | **0.8901** | 0.5210 | 0.5633 | 206.5s |

**XGBoost** achieved the best overall performance with the highest F1 score (0.6783), best recall (0.6555), and fastest training time (16.2s). SVM had the highest ROC-AUC but was the slowest to train and had the lowest F1 score.

---

## Logged Event Types

The training script emits structured JSON events at every stage of the pipeline:

| Event Type | Description |
|---|---|
| `experiment_started` | Models list, dataset name, Python version |
| `dataset_created` | Sample counts, fraud ratio, feature names |
| `preprocessing_complete` | Train/test sizes, SMOTE details |
| `training_started` | Model name, hyperparameter grid |
| `training_complete` | All evaluation metrics and confusion matrix |
| `cross_validation` | 5-fold CV F1 mean, std, and per-fold scores |
| `feature_importance` | Top 10 features by importance (tree models) |
| `model_saved` | File path and size of serialized model |
| `experiment_summary` | Best model and full comparison |
| `experiment_complete` | Total time and event count |

---

## Top Features (by Importance)

From the tree-based models, the most important features for fraud detection were:

1. **time_since_last_tx** (feature 6) — Highest importance across both RandomForest (0.121) and XGBoost (0.186)
2. **feat_20** (feature 19) — Second most important in both models
3. **max_tx_24h** (feature 3) — Maximum transaction value in 24 hours
4. **feat_13** (feature 12) — Consistently important across models
5. **distance_from_last_tx** (feature 5) — Geographic anomaly indicator

---

## Troubleshooting

**Logstash can't find training.log:**  
Make sure the path in `logstash.conf` uses WSL format: `/mnt/c/Users/...` not `C:\Users\...`

**Windows line ending issues:**  
Run `dos2unix logstash.conf` to convert line endings.

**Elasticsearch not responding:**  
Restart it with `cd ~/elasticsearch-9.3.1 && bin/elasticsearch`. Wait for GREEN status.

**`ssl_certificate_verification` error in Logstash 9.x:**  
Replace `ssl_certificate_verification => false` with `ssl_verification_mode => "none"` in `logstash.conf`.

**pip3 not found:**  
Install with `sudo apt install python3-pip -y`. Use `--break-system-packages` flag on Ubuntu 24.04.

**Logstash hangs on startup:**  
Usually means Elasticsearch is not running. Start Elasticsearch first, verify with `curl`, then start Logstash.

---

## Technologies Used

- **Elasticsearch 9.3.1** — Search and analytics engine for storing experiment metrics
- **Logstash 9.3.1** — Data pipeline for ingesting JSON training logs
- **Kibana 9.3.1** — Visualization and dashboarding for model comparison
- **Python 3.12** — ML training and JSON log generation
- **scikit-learn** — SVM, RandomForest, preprocessing, evaluation metrics
- **XGBoost** — Gradient boosted trees
- **LightGBM** — Light gradient boosting
- **imbalanced-learn** — SMOTE oversampling
- **WSL2 + Ubuntu 24.04** — Linux environment on Windows

---

## References

- [Original ELK Lab — Prof. Ramin Mohammadi MLOps Course](https://github.com/raminmohammadi/MLOps/tree/main/Labs/ELK_Labs/Lab1_Setup_Windows_WSL_Ubuntu)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Logstash File Input Plugin](https://www.elastic.co/guide/en/logstash/current/plugins-inputs-file.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
