import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("results/daily.csv")
df["tomorrow"] = df["temp_max_c"].shift(-1)
df["lag1"] = df["temp_max_c"].shift(1)
df = df.dropna()

X = df[["lag1"]].values
y = df["tomorrow"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.3, shuffle=False)

m = LinearRegression().fit(X_train, y_train)
pred = m.predict(X_test)

rmse = mean_squared_error(y_test, pred)**0.5
from sklearn.metrics import r2_score
r2 = r2_score(y_test, pred)

baseline = X_test.flatten()
base_rmse = mean_squared_error(y_test, baseline)**0.5

print("lag1 only → RMSE:", round(rmse,3), "R²:", 
round(r2,3))
print("baseline   → RMSE:", round(base_rmse,3))

