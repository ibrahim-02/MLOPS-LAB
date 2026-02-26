import pandas as pd
import joblib
import io
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine
from google.cloud import storage


def download_data():
    """Loads the Wine dataset and returns X as a DataFrame and y as a Series."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {list(wine.target_names)}")
    return X, y, list(wine.target_names)


def preprocess_data(X, y):
    """Performs train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    """Trains a GradientBoostingClassifier."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluates the model and prints metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\nModel accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    return acc


def save_model_to_gcs(model, bucket_name, blob_name):
    """Saves model directly to GCS using an in-memory buffer."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        blob.upload_from_file(buffer, content_type="application/octet-stream")
        print(f"✅ Model successfully uploaded to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"❌ Failed to upload model: {e}")


def main():
    bucket_name = os.getenv("GCS_BUCKET_NAME", "your-gcs-bucket-name")

    X, y, target_names = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, target_names)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    blob_name = f"trained_models/wine_gbc_{timestamp}.joblib"

    save_model_to_gcs(model, bucket_name, blob_name)


if __name__ == "__main__":
    main()