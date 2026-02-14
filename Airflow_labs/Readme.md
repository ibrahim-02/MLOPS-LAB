# Wine Quality Classification Pipeline with Apache Airflow

A production-grade MLOps pipeline that orchestrates wine quality classification using Apache Airflow, scikit-learn's GradientBoostingClassifier, and Docker containerization.

## 📋 Project Overview

This project demonstrates end-to-end ML workflow orchestration using Apache Airflow. The pipeline loads the wine quality dataset, performs train-test splitting, trains a gradient boosting classifier, and evaluates model performance - all orchestrated as a DAG (Directed Acyclic Graph) with proper task dependencies and XCom-based data passing.

**Key Features:**
- ✅ Modular ML pipeline with 4 distinct Airflow tasks
- ✅ Proper XCom data serialization using base64-encoded pickle
- ✅ Docker-based deployment for environment consistency
- ✅ Model persistence with performance metrics logging
- ✅ Configurable DAG with retry logic and monitoring

## 🏗️ Architecture

```
wine_classification_pipeline
    │
    ├─> load_data           # Load wine dataset from sklearn
    │
    ├─> split_data          # Train-test split (80-20)
    │
    ├─> train_model         # Train GradientBoostingClassifier
    │
    └─> predict_and_evaluate # Generate predictions & metrics
```

**Task Dependencies:** Linear pipeline with sequential execution

## 📁 Project Structure

```
Airflow_labs/
├── .env                      # Airflow UID configuration
├── docker-compose.yaml       # Airflow services orchestration
├── dags/
│   ├── airflow.py           # Main DAG definition
│   └── src/
│       ├── __init__.py
│       ├── lab.py           # Core ML functions
│       └── model/
│           └── model.pkl    # Trained model (generated at runtime)
├── logs/                    # Airflow execution logs
├── config/                  # Airflow configuration
└── plugins/                 # Custom Airflow plugins
```

## 🚀 Setup & Installation

### Prerequisites
- Docker Desktop (v20.10+)
- Docker Compose (v2.0+)
- 4GB+ RAM available for Docker
- Git

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd Airflow_labs
```

### Step 2: Configure Environment
```bash
# Create .env file with your user ID (Linux/Mac)
echo -e "AIRFLOW_UID=$(id -u)" > .env

# For Windows PowerShell
"AIRFLOW_UID=50000" | Out-File -FilePath .env -Encoding utf8
```

### Step 3: Update Docker Compose (Important!)

Ensure your `docker-compose.yaml` includes scikit-learn installation:

```yaml
environment:
  &airflow-common-env
  _PIP_ADDITIONAL_REQUIREMENTS: 'scikit-learn'
  AIRFLOW__CORE__LOAD_EXAMPLES: 'false'  # Optional: hide example DAGs
```

### Step 4: Initialize Airflow
```bash
# Initialize database and create admin user
docker-compose up airflow-init

# Start all services in detached mode
docker-compose up -d
```

### Step 5: Verify Services
```bash
# Check all containers are running
docker ps

# Expected services:
# - postgres (database)
# - redis (message broker)
# - airflow-webserver
# - airflow-scheduler
# - airflow-worker
# - airflow-triggerer
```

## 🎯 Running the Pipeline

### Access Airflow Web UI
1. Navigate to: `http://localhost:8080`
2. Login credentials:
   - **Username:** `airflow`
   - **Password:** `airflow`

### Execute the DAG
1. Find `wine_classification_pipeline` in the DAGs list
2. Click the toggle to **unpause** the DAG
3. Click the **play button** (▶) to trigger a manual run
4. Monitor execution in the **Graph** or **Grid** view

### View Task Logs
- Click on any task box in the Graph view
- Select **Log** to see detailed execution output
- Check model performance metrics in the `predict_and_evaluate` task logs

## 🔧 Technical Implementation Details

### XCom Data Passing Strategy

**Challenge:** Airflow's Jinja templating converts Python objects to strings when using `op_kwargs`, breaking dictionary operations.

**Solution:** Wrapper functions with direct `ti.xcom_pull()` calls to preserve native Python types:

```python
def _train_model(**context):
    ti = context["ti"]
    split = ti.xcom_pull(task_ids="split_data")  # Returns native dict
    return train_model(split)
```

### Data Serialization

All functions in `lab.py` use base64-encoded pickle for XCom compatibility:

```python
import base64
import pickle

# Serialize for XCom
data_encoded = base64.b64encode(pickle.dumps(data)).decode("utf-8")
return {"data": data_encoded}

# Deserialize from XCom
data = pickle.loads(base64.b64decode(xcom_dict["data"]))
```

### Model Training Details
- **Algorithm:** GradientBoostingClassifier
- **Dataset:** scikit-learn wine quality dataset
- **Train-Test Split:** 80-20
- **Model Storage:** Persistent pickle file in `dags/src/model/`

## 📊 Expected Output

Upon successful execution, the `predict_and_evaluate` task logs will show:

```
Model Accuracy: 0.XX
Classification Report:
              precision    recall  f1-score   support
...
Confusion Matrix:
[[XX XX XX]
 [XX XX XX]
 [XX XX XX]]
```

## 🛠️ Customization Options

### Change DAG Schedule
Edit `airflow.py`:
```python
schedule_interval="0 0 * * *"  # Run daily at midnight
```

### Modify Model Hyperparameters
Edit `lab.py`:
```python
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)
```

### Adjust Train-Test Split
Edit `lab.py`:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # 70-30 split
)
```

## 🐛 Troubleshooting

### DAG Not Appearing in UI
- Verify file is in `dags/` folder
- Check for Python syntax errors: `docker exec -it <scheduler-container> python /opt/airflow/dags/airflow.py`
- Refresh the Airflow UI (can take 30-60 seconds)

### Import Errors
- Ensure `dags/src/__init__.py` exists (can be empty)
- Verify scikit-learn is installed: `docker exec -it <scheduler-container> pip list | grep scikit`

### Task Failures
- Check task logs in Airflow UI for specific error messages
- Verify model directory exists: `mkdir -p dags/src/model`
- Ensure proper file permissions: `chmod -R 777 dags/`

### Docker Issues
```bash
# Stop all services
docker-compose down

# Remove volumes (fresh start)
docker-compose down -v

# Reinitialize
docker-compose up airflow-init
docker-compose up -d
```

## 🔄 Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (database, logs)
docker-compose down -v

# Remove generated model
rm dags/src/model/model.pkl
```

## 🆕 Key Improvements Over Base Lab

1. **Proper XCom Handling:** Implemented wrapper functions to avoid Jinja template string serialization issues
2. **Production-Ready Config:** Disabled example DAGs and added proper retry logic
3. **Modular Design:** Separated DAG orchestration (`airflow.py`) from ML logic (`lab.py`)
4. **Comprehensive Documentation:** Detailed setup, troubleshooting, and customization guide
5. **Clean Docker Setup:** Streamlined compose file with minimal required services

## 📚 Learning Outcomes

- **Airflow Fundamentals:** DAG creation, task dependencies, XCom data passing
- **MLOps Practices:** Model training orchestration, versioning, and logging
- **Docker Orchestration:** Multi-container deployment with Docker Compose
- **Production Considerations:** Error handling, retry logic, monitoring

## 📝 References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [scikit-learn GradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [Docker Compose Docs](https://docs.docker.com/compose/)

---

**Author:** Ibrahim  
**Course:** MS Data Analytics Engineering, Northeastern University  
**Semester:** Spring 2026
