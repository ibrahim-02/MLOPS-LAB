import requests
import json

url = 'https://wine-classifier-160071191552.us-east1.run.app/predict'

payload = {
    'alcohol': 13.2,
    'malic_acid': 2.77,
    'ash': 2.51,
    'alcalinity_of_ash': 18.5,
    'magnesium': 96.0,
    'total_phenols': 2.20,
    'flavanoids': 2.53,
    'nonflavanoid_phenols': 0.26,
    'proanthocyanins': 1.56,
    'color_intensity': 5.0,
    'hue': 1.05,
    'od280_od315_of_diluted_wines': 3.33,
    'proline': 820.0
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print("Status:", response.status_code)
print("Body:", response.text)

if response.status_code == 200:
    try:
        result = response.json()
        prediction = result['prediction']
        wine_class = result['class']
        print(f'Predicted wine class: {wine_class} (class {prediction})')
    except Exception as e:
        print("Could not parse JSON:", e)
else:
    print('Error:', response.status_code)