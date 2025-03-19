import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df.fillna(df.median(), inplace=True)  # Handle missing values
    X = df.iloc[:, :-1].values  # Features (Spectral Reflectance)
    y = df.iloc[:, -1].values   # Target (DON concentration)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
