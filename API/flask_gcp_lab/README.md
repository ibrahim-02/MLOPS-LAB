
### 1. Built the ML Model
- Used sWine dataset  (178 samples, 13 features, 3 classes)
- Trained GradientBoostingClassifier
- Model accuracy: ~97%
- Saved as `model/model.pkl` using joblib

### 2. Created Flask API

**File:** `src/main.py`

Created a Flask REST API with one endpoint:
- **Route:** `POST /predict`
- **Input:** JSON with 13 wine chemical features
- **Output:** Predicted wine class (class_0, class_1, or class_2)

**Key code:**
```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract 13 features
    features = [alcohol, malic_acid, ash, ...]
    prediction = predict_wine(features)
    return jsonify({'prediction': pred_label})
```

### 3. Dockerized the Application

**File:** `Dockerfile`
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
RUN python src/train.py
EXPOSE $PORT
ENV PYTHONPATH=/app
CMD ["sh", "-c", "python src/main.py"]
```

**What this does:**
- Uses Python 3.9 base image
- Installs dependencies
- Trains model during build
- Exposes port for Cloud Run
- Runs Flask app

### 4. Deployed to Google Cloud Run

**Commands used:**
```bash
# Set up GCP
gcloud config set project flask-gcp-lab
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Deploy
gcloud run deploy wine-classifier \
  --source . \
  --region us-east1 \
  --allow-unauthenticated \
  --port 8080
```

**Deployment Result:**
- Service URL: `https://wine-classifier-160071191552.us-east1.run.app`
- Status: Successfully deployed ✅
- Container automatically scales based on traffic

### 5. Tested the Deployed API

**Local test:** `test_api.py`
```python
url = 'https://wine-classifier-160071191552.us-east1.run.app/predict'
payload = {13 wine features...}
response = requests.post(url, json=payload)
```

**Result:** 
```
Status: 200
Body: {"prediction":"class_1"}
```

### 6. Created Streamlit Frontend

Built UI with sliders for 13 features that connects to deployed Cloud Run API.

**Running:** `python -m streamlit run streamlit_app.py`

## Challenges Faced

1. **Permission Errors:** Had to grant proper IAM roles to Cloud Build service account for Artifact Registry
   - Fixed by: `gcloud projects add-iam-policy-binding` with correct service account



## Key Learnings

1. **Flask API Development:** Created REST endpoint that accepts JSON and returns predictions
2. **Docker Containerization:** Packaged entire ML application into portable container
3. **GCP Cloud Run:** Deployed serverless container with automatic scaling
4. **IAM Permissions:** Understood service accounts and role bindings in GCP
5. **End-to-End ML Deployment:** Connected training → API → deployment → frontend

## Files Submitted
```
flask_gcp_lab/
├── src/
│   ├── train.py
│   ├── predict.py
│   └── main.py
├── Dockerfile
├── requirements.txt
├── test_api.py
├── streamlit_app.py
└── README.md
```

`

## Conclusion

Successfully deployed a machine learning model as a production-ready Flask API on Google Cloud Run. The API is publicly accessible, automatically scales, and integrates with a Streamlit frontend for user interaction.

**Live API:** https://wine-classifier-160071191552.us-east1.run.app/predict
