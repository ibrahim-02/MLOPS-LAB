from flask import Flask, request, jsonify
from predict import predict_wine
import os

app = Flask(__name__)

# Map numeric model output to human-readable class
label_map = {
    0: "class_0",
    1: "class_1",
    2: "class_2"
}
@app.route('/', methods=['GET'])
def home():
    return 'Wine Classifier API is running!'
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    alcohol = float(data['alcohol'])
    malic_acid = float(data['malic_acid'])
    ash = float(data['ash'])
    alcalinity_of_ash = float(data['alcalinity_of_ash'])
    magnesium = float(data['magnesium'])
    total_phenols = float(data['total_phenols'])
    flavanoids = float(data['flavanoids'])
    nonflavanoid_phenols = float(data['nonflavanoid_phenols'])
    proanthocyanins = float(data['proanthocyanins'])
    color_intensity = float(data['color_intensity'])
    hue = float(data['hue'])
    od280_od315_of_diluted_wines = float(data['od280_od315_of_diluted_wines'])
    proline = float(data['proline'])
    
    print(f"Received features: alcohol={alcohol}, malic_acid={malic_acid}, ...")
    
    # Prepare features as list for prediction
    features = [
        alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
        total_phenols, flavanoids, nonflavanoid_phenols,
        proanthocyanins, color_intensity, hue,
        od280_od315_of_diluted_wines, proline
    ]
    
    # Call model
    prediction = predict_wine(features)
    
    # Convert numeric class → label string for frontend
    try:
        pred_int = int(prediction)
        pred_label = label_map.get(pred_int, str(pred_int))
    except Exception:
        pred_label = str(prediction)
    
    return jsonify({'prediction': pred_label})

if __name__ == '__main__':
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
