import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load your hourly data
df = pd.read_csv("results/hourly.csv")
df = df.dropna().reset_index(drop=True)

# Features and target
features = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
X = df[features].values
y = (df["precip_mm"].shift(-6) > 0).astype(int)  # rain in next 6h
y = y[:-6]
X = X[:-6]

# Time-series CV setup
tscv = TimeSeriesSplit(n_splits=5)

f1_scores = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    score = f1_score(y_test, preds)
    f1_scores.append(score)
    print(f"Fold {fold+1} F1: {score:.3f}")

print("\nAverage F1 across folds:", np.mean(f1_scores).round(3))
