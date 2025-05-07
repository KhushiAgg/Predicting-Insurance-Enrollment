import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import add_features

def train_model():
    df = load_data('data/employee_data.csv')
    df = add_features(df)
    X, y, scaler, label_encoders = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and preprocessing artifacts
    joblib.dump(model, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    return model, X_test, y_test

train_model()