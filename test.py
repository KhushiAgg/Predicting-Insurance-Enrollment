import requests

url = "http://localhost:8000/predict"

sample_input = {
    "employee_id":20001,
    "age": 34,
    "gender": "Male",
    "marital_status": "Married",
    "salary": 75000,
    "employment_type": "Full-time",
    "region": "West",
    "has_dependents": True,
    "tenure_years": 4.5
}

response = requests.post(url, json=sample_input)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())