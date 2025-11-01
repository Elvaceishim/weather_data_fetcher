
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

H = 12
EVENT_MM = 1.0
HOURLY = Path("results/hourly.csv")

# --- label builder ---
def make_labels(df, horizon=H, event_mm=EVENT_MM):
    prec = df["precip_mm"].values
    y = np.zeros(len(df), dtype=int)
    for i in range(len(prec) - horizon):
        y[i] = 1 if np.nansum(prec[i+1:i+1+horizon]) >= event_mm else 0
    return y[:-horizon]

# --- feature builder ---
def build_features(df):
    base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
    for c in base:
        df[f"d_{c}"] = df[c].diff()
        df[f"ma3_{c}"] = df[c].rolling(3).mean()
    df = df.dropna().reset_index(drop=True)
    return df[base + [f"d_{c}" for c in base] + [f"ma3_{c}" for c in base]]

# --- tuning function ---
def tune_xgb(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    grid = [
        dict(learning_rate=0.05, max_depth=3, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=3),
        dict(learning_rate=0.05, max_depth=5, n_estimators=800, subsample=0.8, colsample_bytree=0.8, min_child_weight=3),
        dict(learning_rate=0.1,  max_depth=6, n_estimators=600, subsample=0.9, colsample_bytree=0.9, min_child_weight=1),
        dict(learning_rate=0.03, max_depth=4, n_estimators=1000, subsample=0.7, colsample_bytree=0.7, min_child_weight=5),
    ]
    results = []
    for params in grid:
        fold_f1 = []
        for tr, te in tscv.split(X):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            model = XGBClassifier(
                **params,
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                random_state=42,
            )
            model.fit(X_tr, y_tr)
            p = model.predict(X_te)
            _, _, F1, _ = precision_recall_fscore_support(y_te, p, average="binary", zero_division=0)
            fold_f1.append(F1)
        avg_f1 = float(np.mean(fold_f1))
        results.append((params, avg_f1))
        print(f"{params} → mean F1 = {avg_f1:.3f}")
    best = max(results, key=lambda x: x[1])
    print("\n🏆 Best params:", best[0], f"→ mean F1={best[1]:.3f}")
    Path("models").mkdir(exist_ok=True)
    with open("models/xgb_tuned.json", "w") as f:
        json.dump(dict(params=best[0], mean_f1=best[1]), f, indent=2)

def main():
    if not HOURLY.exists():
        raise FileNotFoundError("results/hourly.csv not found. Run: make hourly PAST_DAYS=90")
    df = pd.read_csv(HOURLY, parse_dates=["time"])
    y_all = make_labels(df)
    Xdf = df.iloc[:-H].copy()
    Xdf = build_features(Xdf)

    n = len(Xdf)
    y = y_all[-n:]
    X = Xdf.values.astype(np.float32)

    if len(X) != len(y):
        raise RuntimeError(f"Misaligned shapes: X={len(X)} rows vs y={len(y)} labels")

    tune_xgb(X, y)

if __name__ == "__main__":
    main()
