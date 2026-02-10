import os
import base64
import pickle
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    """
    Loads the Wine dataset and returns base64-encoded serialized data.
    Returns:
        dict: Base64-encoded X, y, and target_names (all JSON-safe for XCom).
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    target_names = list(wine.target_names)

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {target_names}")

    X_b64 = base64.b64encode(pickle.dumps(X)).decode("ascii")
    y_b64 = base64.b64encode(pickle.dumps(y)).decode("ascii")

    return {
        "X": X_b64,
        "y": y_b64,
        "target_names": target_names,
    }


def split_data(data: dict, test_size: float = 0.2, random_state: int = 42):
    """
    Deserializes data, performs train/test split, and returns
    base64-encoded splits (JSON-safe for XCom).
    """
    X = pickle.loads(base64.b64decode(data["X"]))
    y = pickle.loads(base64.b64decode(data["y"]))
    target_names = data["target_names"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    return {
        "X_train": base64.b64encode(pickle.dumps(X_train)).decode("ascii"),
        "X_test": base64.b64encode(pickle.dumps(X_test)).decode("ascii"),
        "y_train": base64.b64encode(pickle.dumps(y_train)).decode("ascii"),
        "y_test": base64.b64encode(pickle.dumps(y_test)).decode("ascii"),
        "target_names": target_names,
    }


def train_model(
    split: dict,
    filename: str = "model.pkl",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42,
):
    """
    Trains a GradientBoostingClassifier on the training split and saves
    the model to disk.
    Returns:
        str: The path where the model was saved.
    """
    X_train = pickle.loads(base64.b64decode(split["X_train"]))
    y_train = pickle.loads(base64.b64decode(split["y_train"]))

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    output_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved trained model to {model_path}")
    return model_path


def predict_and_evaluate(split: dict, model_path: str):
    """
    Loads the saved model, predicts on the test set, and prints
    evaluation metrics.
    Returns:
        dict: accuracy (float) and report (str) — both JSON-safe.
    """
    X_test = pickle.loads(base64.b64decode(split["X_test"]))
    y_test = pickle.loads(base64.b64decode(split["y_test"]))
    target_names = split["target_names"]

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"\nModel accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(report)

    return {"accuracy": float(acc), "report": report}


# ── Local test runner ────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    split = split_data(data)
    model_path = train_model(split)
    results = predict_and_evaluate(split, model_path)