import numpy as np
import joblib
import os
from train import run_training

# Load the trained model
model = joblib.load("model/model.pkl")

def predict_wine(input_features):
    """
    Predict wine class from input features.
    
    Args:
        input_features: list or array of 13 wine features
    
    Returns:
        int: predicted class (0, 1, or 2)
    """
    input_data = np.array([input_features])
    prediction = model.predict(input_data)
    return int(prediction[0])

if __name__ == "__main__":
    if os.path.exists("model/model.pkl"):
        print("Model loaded successfully\n")
    else:
        print("Model not found. Training new model...\n")
        os.makedirs("model", exist_ok=True)
        run_training()
        print("\nModel trained. Now making predictions...\n")
    
    # Example prediction with 13 features
    sample_wine = [13.2, 2.77, 2.51, 18.5, 96.0, 2.20, 2.53, 
                   0.26, 1.56, 5.0, 1.05, 3.33, 820.0]
    
    print("Example prediction:")
    result = predict_wine(sample_wine)
    print(f"Predicted wine class: {result}")