"""
ML Experiment Tracker with ELK Stack Integration
=================================================
Trains SVM, RandomForest, XGBoost, LightGBM on Credit Card Fraud Detection
Logs all experiment metrics as JSON → Logstash → Elasticsearch → Kibana

Author: Ibrahim
Course: IE 7374 MLOps (Prof. Ramin Mohammadi)
"""

import json
import time
import os
import uuid
import socket
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, matthews_corrcoef
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
EXPERIMENT_NAME = "credit_fraud_detection"
RUN_ID = str(uuid.uuid4())[:8]
MODEL_DIR = "saved_models"

# Write directly to training.log in the same directory as model.py
LOGSTASH_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training.log")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Logger Setup (JSON lines for Logstash)
# ─────────────────────────────────────────────
class JsonLogger:
    """Logs experiment events as JSON lines for Logstash file input."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.events = []
    
    def log(self, event_type, data):
        record = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_name": EXPERIMENT_NAME,
            "run_id": RUN_ID,
            "hostname": socket.gethostname(),
            "event_type": event_type,
            **data
        }
        self.events.append(record)
        # Append as JSON line
        with open(self.filepath, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  [LOG] {event_type}: {json.dumps(data, indent=2)}")
    
    def get_all_events(self):
        return self.events

logger = JsonLogger(LOGSTASH_LOG)

# ─────────────────────────────────────────────
# Step 1: Generate Synthetic Fraud Dataset
# ─────────────────────────────────────────────
def create_fraud_dataset():
    """Create an imbalanced fraud detection dataset."""
    print("\n" + "="*60)
    print("STEP 1: Generating Synthetic Fraud Dataset")
    print("="*60)
    
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        weights=[0.95, 0.05],  # 95% legit, 5% fraud → imbalanced
        random_state=42,
        flip_y=0.02
    )
    
    feature_names = [
        "tx_amount", "tx_frequency", "avg_tx_value", "max_tx_24h",
        "distance_from_home", "distance_from_last_tx", "time_since_last_tx",
        "merchant_risk_score", "card_age_days", "num_declined_24h",
        "avg_balance_30d", "velocity_1h", "feat_13", "feat_14",
        "feat_15", "feat_16", "feat_17", "feat_18", "feat_19", "feat_20"
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df["is_fraud"] = y
    
    dataset_info = {
        "total_samples": len(df),
        "num_features": X.shape[1],
        "fraud_count": int(y.sum()),
        "legit_count": int(len(y) - y.sum()),
        "fraud_ratio": round(float(y.mean()), 4),
        "feature_names": feature_names
    }
    
    logger.log("dataset_created", dataset_info)
    print(f"  Samples: {dataset_info['total_samples']}")
    print(f"  Fraud: {dataset_info['fraud_count']} ({dataset_info['fraud_ratio']*100:.1f}%)")
    print(f"  Legit: {dataset_info['legit_count']}")
    
    return df, feature_names

# ─────────────────────────────────────────────
# Step 2: Preprocessing + SMOTE
# ─────────────────────────────────────────────
def preprocess(df, feature_names):
    """Split, scale, and apply SMOTE for class balance."""
    print("\n" + "="*60)
    print("STEP 2: Preprocessing + SMOTE Oversampling")
    print("="*60)
    
    X = df[feature_names].values
    y = df["is_fraud"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    preprocess_info = {
        "train_size_original": len(y_train),
        "train_size_after_smote": len(y_train_resampled),
        "test_size": len(y_test),
        "smote_applied": True,
        "scaling": "StandardScaler",
        "train_fraud_before_smote": int(y_train.sum()),
        "train_fraud_after_smote": int(y_train_resampled.sum()),
    }
    
    logger.log("preprocessing_complete", preprocess_info)
    print(f"  Train (original): {preprocess_info['train_size_original']}")
    print(f"  Train (after SMOTE): {preprocess_info['train_size_after_smote']}")
    print(f"  Test: {preprocess_info['test_size']}")
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler

# ─────────────────────────────────────────────
# Step 3: Model Definitions with Hyperparams
# ─────────────────────────────────────────────
def get_models():
    """Define models with hyperparameter grids for experimentation."""
    models = {
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "class_weight": ["balanced", None]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42, 
                eval_metric="logloss",
                use_label_encoder=False
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.1],
                "scale_pos_weight": [1, 10]
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, verbose=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [6, 10, -1],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [31, 50],
                "is_unbalance": [True, False]
            }
        }
    }
    return models

# ─────────────────────────────────────────────
# Step 4: Train + Evaluate + Log
# ─────────────────────────────────────────────
def train_and_evaluate(model_name, model_config, X_train, X_test, y_train, y_test):
    """Train model with GridSearchCV, evaluate, and log everything."""
    print(f"\n{'─'*60}")
    print(f"  Training: {model_name}")
    print(f"{'─'*60}")
    
    # Log training start
    logger.log("training_started", {
        "model_name": model_name,
        "param_grid": {k: [str(v) for v in vals] for k, vals in model_config["params"].items()},
    })
    
    start_time = time.time()
    
    # GridSearchCV
    grid_search = GridSearchCV(
        model_config["model"],
        model_config["params"],
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    train_time = round(time.time() - start_time, 3)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "model_name": model_name,
        "best_params": {k: str(v) for k, v in grid_search.best_params_.items()},
        "training_time_sec": train_time,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "cv_best_f1": round(grid_search.best_score_, 4),
        "total_fits": grid_search.cv_results_["mean_test_score"].shape[0] * 3,
    }
    
    # Log metrics
    logger.log("training_complete", metrics)
    
    # Cross-val scores on best model
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="f1")
    cv_info = {
        "model_name": model_name,
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "cv_f1_scores": [round(s, 4) for s in cv_scores.tolist()],
    }
    logger.log("cross_validation", cv_info)
    
    # Feature importance (tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        top_features = sorted(
            zip(range(len(importances)), importances),
            key=lambda x: x[1], reverse=True
        )[:10]
        fi_info = {
            "model_name": model_name,
            "top_10_features": [
                {"feature_index": int(idx), "importance": round(float(imp), 4)}
                for idx, imp in top_features
            ]
        }
        logger.log("feature_importance", fi_info)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_best.pkl")
    joblib.dump(best_model, model_path)
    logger.log("model_saved", {
        "model_name": model_name,
        "model_path": model_path,
        "file_size_kb": round(os.path.getsize(model_path) / 1024, 2)
    })
    
    # Print summary
    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1 Score:  {metrics['f1_score']}")
    print(f"  ROC AUC:   {metrics['roc_auc']}")
    print(f"  MCC:       {metrics['mcc']}")
    print(f"  Time:      {train_time}s")
    print(f"  Best Params: {grid_search.best_params_}")
    
    return best_model, metrics

# ─────────────────────────────────────────────
# Step 5: Model Comparison Summary
# ─────────────────────────────────────────────
def compare_models(all_results):
    """Generate and log a comparison summary."""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison = []
    for name, metrics in all_results.items():
        row = {
            "model_name": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "mcc": metrics["mcc"],
            "training_time_sec": metrics["training_time_sec"],
        }
        comparison.append(row)
    
    # Sort by F1
    comparison.sort(key=lambda x: x["f1_score"], reverse=True)
    
    print(f"\n  {'Model':<15} {'F1':>8} {'AUC':>8} {'Recall':>8} {'MCC':>8} {'Time(s)':>8}")
    print(f"  {'─'*55}")
    for r in comparison:
        print(f"  {r['model_name']:<15} {r['f1_score']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['recall']:>8.4f} {r['mcc']:>8.4f} {r['training_time_sec']:>8.1f}")
    
    best = comparison[0]
    logger.log("experiment_summary", {
        "best_model": best["model_name"],
        "best_f1": best["f1_score"],
        "best_auc": best["roc_auc"],
        "all_models_comparison": comparison,
        "total_models_trained": len(comparison),
    })
    
    print(f"\n  Best Model: {best['model_name']} (F1={best['f1_score']}, AUC={best['roc_auc']})")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ML EXPERIMENT TRACKER + ELK STACK INTEGRATION")
    print("  Models: SVM, RandomForest, XGBoost, LightGBM")
    print("  Dataset: Credit Card Fraud Detection (Synthetic)")
    print(f"  Run ID: {RUN_ID}")
    print("=" * 60)
    
    logger.log("experiment_started", {
        "models": ["SVM", "RandomForest", "XGBoost", "LightGBM"],
        "dataset": "synthetic_credit_fraud",
        "python_version": f"{__import__('sys').version}",
    })
    
    # Create dataset
    df, feature_names = create_fraud_dataset()
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df, feature_names)
    
    # Train all models
    models = get_models()
    all_results = {}
    trained_models = {}
    
    for model_name, model_config in models.items():
        best_model, metrics = train_and_evaluate(
            model_name, model_config, X_train, X_test, y_train, y_test
        )
        all_results[model_name] = metrics
        trained_models[model_name] = best_model
    
    # Compare
    compare_models(all_results)
    
    logger.log("experiment_complete", {
        "total_training_time_sec": round(sum(m["training_time_sec"] for m in all_results.values()), 3),
        "total_events_logged": len(logger.get_all_events()),
        "log_file": LOGSTASH_LOG,
    })
    
    print(f"\n  Models saved to: {MODEL_DIR}/")
    print(f"  Experiment logs: {LOGSTASH_LOG}")
    print(f"  Total events logged: {len(logger.get_all_events())}")
    print(f"\n  Next: Start Logstash to ingest into Elasticsearch!")

if __name__ == "__main__":
    main()