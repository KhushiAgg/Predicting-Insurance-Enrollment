import pandas as pd

def add_features(df):
    df = df.copy()
    df['age_bucket'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=False)
    df['tenure_bucket'] = pd.cut(df['tenure_years'], bins=[0, 2, 5, 10, 20, 40], labels=False)
    return df