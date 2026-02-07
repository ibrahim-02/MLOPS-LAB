import os
import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_training():
    # 1. Load dataset (Wine)
    wine = load_wine()
    X = wine.data   # shape (178, 13)
    y = wine.target # 0,1,2 classes
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {wine.target_names}")
    
    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Train a model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nModel accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    
    # 5. Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    print("\nSaved trained model to model/model.pkl")

if __name__ == "__main__":
    run_training()