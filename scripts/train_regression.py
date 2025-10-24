import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib, os

df = pd.read_csv("results/summary.csv")

# Create lagged feature: today's temp → tomorrow's temp
df["tomorrow"] = df["temp_max_c"].shift(-1)
df = df.dropna()

X = df[["temp_max_c"]].values
y = df["tomorrow"].values

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Baseline: "tomorrow = today"
baseline_pred = X_test.flatten()  # naive baseline
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = baseline_mse ** 0.5

print("📊 MODEL PERFORMANCE")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")
print("")
print("🧠 BASELINE (naive 'tomorrow = today')")
print(f"RMSE: {baseline_rmse:.3f}")
print("")
print("✅ Better than baseline?" , "YES ✅" if rmse < baseline_rmse else "NO ❌")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/temp_regressor.joblib")
print("\n💾 Model saved to models/temp_regressor.joblib")
