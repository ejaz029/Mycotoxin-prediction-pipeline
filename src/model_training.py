import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle  

# ✅ Set dataset path
dataset_path = "data/don_synthetic_data_v2.csv"

# ✅ Check if dataset exists before loading
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}. Please check the file path.")

# ✅ Load dataset
data = pd.read_csv(dataset_path)
print("✅ Dataset loaded successfully!")

# ✅ Handle missing values (if any)
data.fillna(data.median(numeric_only=True), inplace=True)

# ✅ Drop non-numeric columns (e.g., 'hsi_id' if it exists)
non_numeric_columns = data.select_dtypes(exclude=["number"]).columns
if len(non_numeric_columns) > 0:
    print(f"⚠️ Dropping non-numeric columns: {list(non_numeric_columns)}")
    data.drop(columns=non_numeric_columns, inplace=True)

# ✅ Ensure target column exists
target_column = "vomitoxin_ppb"  # Corrected target variable
if target_column not in data.columns:
    raise ValueError(f"❌ Target variable '{target_column}' not found in dataset. Available columns: {list(data.columns)}")

# ✅ Separate features (X) and target (y)
X = data.drop(columns=[target_column])  # Features
y = data[target_column]  # Target

def train_model(X, y):
    """Train an XGBoost regression model and save it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Enable categorical handling explicitly
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, enable_categorical=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ✅ Model evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Model Performance:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # ✅ Save the trained model to 'model.pkl'
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    print("✅ Model saved successfully as 'model.pkl'")

    return model

# ✅ Train and save the model
if __name__ == "__main__":
    train_model(X, y)
