import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.copy()
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.drop('enrolled', errors='ignore')
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separate features and target
    X = df.drop('enrolled', axis=1)
    y = df['enrolled']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoders
