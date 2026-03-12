# GitHub Actions Labs — MLOPS-LAB

This repository contains two GitHub Actions labs demonstrating CI/CD pipelines for Python projects, from basic unit testing to automated ML model training and cloud storage.

**Repo:** [ibrahim-02/MLOPS-LAB](https://github.com/ibrahim-02/MLOPS-LAB)  
**Labs location:** `Github_Labs/`

---

## What I Did Differently from the Reference Lab

The reference lab used a **Random Forest Classifier** trained on the **Iris dataset** — a simple 3-class flower classification problem with only 4 features.

For this lab, I swapped both the model and the dataset:

- **Dataset:** Used the **Wine dataset** (178 samples, 13 features) instead of Iris — a more feature-rich dataset requiring the model to distinguish between 3 wine cultivars based on chemical properties like alcohol content, flavanoids, and color intensity.
- **Model:** Used a **Gradient Boosting Classifier** instead of Random Forest. Gradient Boosting builds trees sequentially where each tree corrects the errors of the previous one, making it generally more accurate but slower to train than Random Forest which builds trees in parallel.
- **Hyperparameters:** Configured `n_estimators=100`, `learning_rate=0.1`, and `max_depth=3` — the learning rate controls how much each tree contributes to the final prediction, which is a tuning parameter not present in the reference Random Forest setup.


---

## Lab 01 — Automated Testing with Pytest & Unittest

### What I Did

Built a simple **calculator application** and set up two automated CI/CD workflows that trigger on every push to `main`:

1. **Pytest workflow** (`pytest.yml`) — runs tests and generates a JUnit XML report, uploads it as a build artifact, and notifies on pass/fail.
2. **Unittest workflow** (`unittest.yml`) — runs Python's built-in `unittest` framework against the calculator test suite.

Both workflows are triggered on:
- Push to `main` or `releases/**` branches
- Issue opened/labeled events (Pytest workflow)

### Files

```
Github_Labs/Lab_01/
├── calculator.py          # Calculator logic
├── requirements.txt       # Dependencies
└── test/
    └── test_uni.py        # Unittest test file
```

### How to Re-run Lab 01

1. Clone the repo:
   ```bash
   git clone https://github.com/ibrahim-02/MLOPS-LAB.git
   cd MLOPS-LAB
   ```

2. Push any change to `main` — both workflows trigger automatically.

3. To view results:
   - Go to the repo on GitHub
   - Click the **Actions** tab
   - Select **"Testing with Pytest"** or **"Python Unittests"**
   - Click the latest run to see logs and download the test report artifact

---

## Lab 02 — ML Model Training & Upload to Google Cloud Storage

### What I Did

Built an end-to-end automated ML pipeline using GitHub Actions that:

1. Loads the **Wine dataset** from scikit-learn
2. Splits the data into train/test sets (80/20)
3. Trains a **Gradient Boosting Classifier** (`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`)
4. Evaluates the model and prints accuracy + classification report
5. Uploads the trained model to a **Google Cloud Storage (GCS)** bucket with a timestamp-based filename: `trained_models/wine_gbc_YYYYMMDD-HHMMSS.joblib`

The workflow runs automatically every day at midnight and can also be triggered manually.

### Files

```
Github_Labs/lab_02/
├── train_and_save_model.py    # Full training + GCS upload pipeline
├── requirements.txt            # Dependencies
└── .github/
    └── workflows/
        └── train-and-upload.yml   # GitHub Actions workflow
```

### Prerequisites

Before running this lab, you need:

- A **Google Cloud Platform** account with a GCS bucket (`model_training_1001`)
- A **GCP Service Account** with Storage Object Admin permissions
- The service account key stored as a GitHub Secret named `GCP_SA_KEY`

#### Setting up the GitHub Secret

1. In your GCP console, create a service account and download the JSON key
2. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Name: `GCP_SA_KEY`
5. Value: paste the entire contents of your GCP service account JSON key

### How to Re-run Lab 02

**Option 1: Manual trigger**
1. Go to [ibrahim-02/MLOPS-LAB](https://github.com/ibrahim-02/MLOPS-LAB)
2. Click the **Actions** tab
3. Select **"Train and Save Wine Model to GCS"** from the left sidebar
4. Click **"Run workflow"** → **"Run workflow"** (green button)

**Option 2: Automatic**  
The workflow runs automatically every day at midnight UTC via cron schedule.

**Option 3: Push a change**  
Any push that modifies files in the repo will not trigger this workflow — use manual trigger or wait for the cron schedule.

### How to Verify the Run Succeeded

1. Go to the **Actions** tab in the repo
2. Click the latest run of **"Train and Save Wine Model to GCS"**
3. Expand each step to see logs — look for:
   - `✅ Model successfully uploaded to gs://model_training_1001/trained_models/wine_gbc_...`
4. Verify in GCS: go to your bucket `model_training_1001` → `trained_models/` folder — the `.joblib` file should be there

### Requirements

```
scikit-learn
pandas
joblib
google-cloud-storage
```

---

## Repository Structure

```
MLOPS-LAB/
└── Github_Labs/
    ├── Lab_01/
    │   ├── calculator.py
    │   ├── requirements.txt
    │   └── test/
    │       └── test_uni.py
    └── lab_02/
        ├── train_and_save_model.py
        └── requirements.txt
.github/
└── workflows/
    ├── pytest.yml
    ├── unittest.yml
    └── train-and-upload.yml
```

---



## Key Concepts Demonstrated

- **CI/CD with GitHub Actions** — automated pipelines triggered by push, schedule, or manual dispatch
- **Pytest & Unittest** — two approaches to automated Python testing
- **GCP Authentication** — using service account keys stored as GitHub Secrets
- **Model versioning** — timestamp-based naming for uploaded model artifacts
- **Pip dependency caching** — speeding up workflow runs by caching installed packages
