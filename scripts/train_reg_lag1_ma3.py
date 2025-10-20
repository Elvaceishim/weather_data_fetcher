import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_IN = "results/daily.csv"

if not os.path.exists(DATA_IN):
    raise SystemExit("results/daily.csv missing. Run `make export-daily` first.")

df = pd.read_csv(DATA_IN)
df["tomorrow"] = df["temp_max_c"].shift(-1)
df["lag1"] = df["temp_max_c"].shift(1)
df["ma3"] = df["temp_max_c"].rolling(3).mean()
df = df.dropna()

features_lag1 = ["lag1"]
features_lag1_ma3 = ["lag1", "ma3"]

X_lag1 = df[features_lag1].values
X_lag1_ma3 = df[features_lag1_ma3].values
y = df["tomorrow"].values

split_kwargs = dict(test_size=0.3, shuffle=False)
(
    X1_train,
    X1_test,
    X2_train,
    X2_test,
    y_train,
    y_test,
) = train_test_split(X_lag1, X_lag1_ma3, y, **split_kwargs)

model_lag1 = LinearRegression().fit(X1_train, y_train)
pred_lag1 = model_lag1.predict(X1_test)
rmse_lag1 = mean_squared_error(y_test, pred_lag1) ** 0.5
r2_lag1 = r2_score(y_test, pred_lag1)

model_lag1_ma3 = LinearRegression().fit(X2_train, y_train)
pred_lag1_ma3 = model_lag1_ma3.predict(X2_test)
rmse_lag1_ma3 = mean_squared_error(y_test, pred_lag1_ma3) ** 0.5
r2_lag1_ma3 = r2_score(y_test, pred_lag1_ma3)

baseline = df["temp_max_c"].shift(1).dropna().values[-len(y_test):]
baseline_rmse = mean_squared_error(y_test, baseline) ** 0.5

print("lag1 only  → RMSE:", round(rmse_lag1, 3), "R²:", round(r2_lag1, 3))
print("lag1+ma3   → RMSE:", round(rmse_lag1_ma3, 3), "R²:", round(r2_lag1_ma3, 3))
print("baseline   → RMSE:", round(baseline_rmse, 3))

improved_rmse = rmse_lag1_ma3 < rmse_lag1
improved_r2 = r2_lag1_ma3 > r2_lag1

if improved_rmse and improved_r2:
    print("\n✅ Keep ma3: both RMSE down and R² up versus lag1-only.")
else:
    print("\nℹ️  Drop ma3 for now; try again once more data is available.")
