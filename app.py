from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import add_features
from src.data_preprocessing import preprocess_data

app = FastAPI()

# Load model and artifacts
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

class EmployeeInput(BaseModel):
    employee_id: int
    age: int
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: bool
    tenure_years: float

@app.post("/predict")
def predict_enrollment(data: EmployeeInput):
    try:
        df = pd.DataFrame([data.dict()])
        df["has_dependents"] = df["has_dependents"].astype(int)

        df = add_features(df)
        df["enrolled"] = 0  # dummy target column

        # Safe encoding: if unseen, assign -1
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        X = df.drop("enrolled", axis=1)
        # X = df
        # Ensure all columns are numeric now
        assert X.select_dtypes(include='number').shape[1] == X.shape[1], "Non-numeric columns present"

        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        return {"enrolled": int(prediction), "probability": round(prob, 4)}

    except Exception as e:
        print("Prediction error:", str(e))
        return {"error": str(e)}
