import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib, os

df = pd.read_csv("results/summary.csv")

# --- Create new features ---
df["tomorrow"] = df["temp_max_c"].shift(-1)
df["temp_range"] = df["temp_max_c"] - df["temp_min_c"]
df["lag1"] = df["temp_max_c"].shift(1)    # yesterday
df["lag2"] = df["temp_max_c"].shift(2)    # two days ago
df["ma3"] = df["temp_max_c"].rolling(3).mean()  # 3-day moving average

df = df.dropna()

# Define feature matrix X and label y
features = [
    "temp_max_c",
    "temp_min_c",
    "temp_range",
    "lag1",
    "lag2",
    "ma3",
]
X = df[features].values
y = df["tomorrow"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

# Baseline: naive "tomorrow = today"
baseline_pred = df["temp_max_c"].shift(1).dropna().values[-len(y_test):]
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = baseline_mse**0.5

print(" MODEL PERFORMANCE (Multi-feature)")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")
print("\n🧠 BASELINE (naive)")
print(f"RMSE: {baseline_rmse:.3f}")
print("\n Better than baseline?", "YES ✅" if rmse < baseline_rmse else "NO ❌")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/temp_regressor_multi.joblib")
print("\n💾 Saved model: models/temp_regressor_multi.joblib")
