# Insurance Enrollment Prediction API

A modular machine learning pipeline and FastAPI service to predict whether an employee will enroll in a corporate insurance program based on demographic and employment details.

---

## Project Structure

```
ml_insurance_enrollment/
├── data/                         # Input dataset (employee_data.csv)
├── models/                       # Trained model, scaler, encoders (pkl files)
├── src/                          # Modular ML components
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── main.py                       # FastAPI app
├── test_client.py                # Sample test script
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (optional)

> Skip if you've already trained and have the `.pkl` files.

```bash
python src/model_training.py
```

### 3. Start the FastAPI server

```bash
uvicorn app:app --reload
```

Visit the docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Sample Input (JSON)

```json
{
  "employee_id": 10003,
  "age": 36,
  "gender": "Male",
  "marital_status": "Divorced",
  "salary": 74145.66,
  "employment_type": "Part-time",
  "region": "Midwest",
  "has_dependents": false,
  "tenure_years": 3.8
}
```

---

## Sample Output
Status Code: 200
Response JSON:
  {
   'enrolled': 1, 
   'probability': 0.97
  }

---

## Tech Stack

- Python 3.8+
- FastAPI + Uvicorn
- Scikit-learn
- Pandas
- Joblib
