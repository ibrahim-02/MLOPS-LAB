# Lab: Wine Classification Pipeline with Apache Airflow

## Overview

This lab demonstrates how to build and orchestrate a machine learning pipeline using **Apache Airflow** and **Docker**. The pipeline trains a **Gradient Boosting Classifier** on the UCI Wine dataset and evaluates its performance — all managed as a DAG (Directed Acyclic Graph) in Airflow.

### What I Did Differently

Instead of using the default Iris or a simple demo dataset, I used the **Wine dataset** from `sklearn.datasets` and trained it using a **GradientBoostingClassifier** (rather than a simpler model like Logistic Regression or Decision Tree). This allowed me to explore ensemble learning within an orchestrated ML pipeline context.


## Pipeline Architecture

The pipeline consists of 4 sequential tasks orchestrated by Airflow:

```
load_data → split_data → train_model → predict_and_evaluate
```

| Task | Description |
|---|---|
| `load_data` | Loads the Wine dataset from sklearn and serializes it via base64/pickle for XCom |
| `split_data` | Performs an 80/20 stratified train/test split |
| `train_model` | Trains a `GradientBoostingClassifier` and saves the model to disk |
| `predict_and_evaluate` | Loads the saved model, runs predictions, and prints accuracy + classification report |

> **Note on XCom serialization:** Airflow's XCom passes data between tasks as JSON. Since NumPy arrays aren't JSON-serializable, all arrays are serialized using `pickle` and then base64-encoded before being pushed to XCom, and decoded on the receiving end.

---

## ML Details

- **Dataset:** UCI Wine (178 samples, 13 features, 3 classes)
- **Model:** `GradientBoostingClassifier`
  - `n_estimators=100`
  - `learning_rate=0.1`
  - `max_depth=3`
  - `random_state=42`
- **Split:** 80% train / 20% test (stratified)
- **Metrics:** Accuracy score + full classification report (precision, recall, F1 per class)

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least **4 GB RAM**, **2 CPUs**, and **10 GB disk space** allocated to Docker
- Python 3.8+ (only needed if running `lab.py` locally)

---

## How to Run

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### Step 2: Set the Airflow UID (Linux only)

```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

On macOS/Windows, you can skip this or create a `.env` file manually:

```
AIRFLOW_UID=50000
```

### Step 3: Initialize Airflow

```bash
docker compose up airflow-init
```

This sets up the Airflow metadata database and creates the default admin user (`airflow2` / `airflow2`).

### Step 4: Start All Services

```bash
docker compose up -d
```

This starts the following containers:
- `airflow-webserver` — the Airflow UI
- `airflow-scheduler` — triggers DAG runs
- `airflow-worker` — executes tasks (Celery)
- `postgres` — metadata database
- `redis` — Celery message broker

### Step 5: Access the Airflow UI

Open your browser and go to: [http://localhost:8080](http://localhost:8080)

Login with:
- **Username:** `airflow2`
- **Password:** `airflow2`

### Step 6: Trigger the DAG

1. In the Airflow UI, find the DAG named **`wine_classification_pipeline`**
2. Toggle it **ON** (enable it)
3. Click the **▶ Trigger DAG** button
4. Watch the tasks run in the Graph or Grid view

### Step 7: View Results

Click on the `predict_and_evaluate` task → **Logs** to see:
- Model accuracy
- Per-class classification report (precision, recall, F1)

---

## Running Locally (Without Airflow)

You can also run the ML pipeline directly using Python:

```bash
pip install scikit-learn numpy
python src/lab.py
```

This runs all four steps sequentially and saves the model to `src/model/model.pkl`.

---

## Stopping the Lab

```bash
docker compose down
```

To also remove stored data and volumes:

```bash
docker compose down --volumes --remove-orphans
```

---

## Troubleshooting

**Docker memory warning:** If you see a warning about insufficient memory, increase Docker's memory limit in Docker Desktop → Settings → Resources (minimum 4 GB recommended).

**DAG not appearing:** Make sure your `dags/` folder is correctly mounted and the DAG file has no syntax errors. Check scheduler logs with:
```bash
docker compose logs airflow-scheduler
```

**Model file not found:** Ensure the `model/` directory is writable inside the container. The `train_model` task creates it automatically, but volume permissions may need adjustment on Linux.
