import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
# ✅ Load test dataset
dataset_path = "data/don_synthetic_data_v2.csv"

# ✅ Check if dataset exists
if not dataset_path or not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Test dataset not found: {dataset_path}. Please check the file path.")

# ✅ Load dataset
data = pd.read_csv(dataset_path)
print("✅ Test dataset loaded successfully!")

# ✅ Handle missing values
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
X_test = data.drop(columns=[target_column])
y_test = data[target_column]

# ✅ Load trained model
model_path = "model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}. Train the model first.")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

print("✅ Model loaded successfully!")

# ✅ Make predictions
y_pred = model.predict(X_test)

# ✅ Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Test Performance:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
