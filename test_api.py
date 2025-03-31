import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Size (sq ft)": [2000],
    "Bedrooms": [3]
}

response = requests.post(url, json=data)
print(response.json())  # Expected: {"Predicted Price ($)": [...] }
