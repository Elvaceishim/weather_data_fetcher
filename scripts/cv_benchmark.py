
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

warnings.filterwarnings("ignore")

H = 12         
EVENT_MM = 1.0 

HOURLY = Path("results/hourly.csv")
META   = Path("models/rain_model_meta.json")

# -----------------------------
# Feature builder (same as CLI/trainer)
# -----------------------------
def rebuild_features_like_training(df: pd.DataFrame, features_from_meta: list) -> pd.DataFrame:
    required = {"time","temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hourly data missing columns: {sorted(missing)}")

    base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
    for c in base:
        df[f"d_{c}"]   = df[c].diff()
        df[f"ma3_{c}"] = df[c].rolling(3).mean()

    for c in ["pressure","humidity","cloudcover","temp_c"]:
        df[f"d3_{c}"] = df[c] - df[c].shift(3)

    df["dew_proxy"]     = df["temp_c"] - (df["humidity"] / 5.0)
    df["d_dew_proxy"]   = df["dew_proxy"].diff()
    df["ma3_dew_proxy"] = df["dew_proxy"].rolling(3).mean()

    df["rain_sum_3h"]   = df["precip_mm"].rolling(3).sum()
    df["rain_sum_6h"]   = df["precip_mm"].rolling(6).sum()
    df["rain_sum_12h"]  = df["precip_mm"].rolling(12).sum()
    df["rain_sum_24h"]  = df["precip_mm"].rolling(24).sum()
    df["rain_max_6h"]   = df["precip_mm"].rolling(6).max()
    df["rain_max_12h"]  = df["precip_mm"].rolling(12).max()

    is_raining = (df["precip_mm"] > 0).astype(int)
    dry = (~(is_raining.astype(bool))).astype(int)
    df["dry_streak_h"] = (dry.groupby((dry != dry.shift()).cumsum()).cumcount() + 1) * dry
    df["dry_streak_h"] = df["dry_streak_h"].where(dry == 1, 0)

    wet = is_raining
    df["wet_streak_h"] = (wet.groupby((wet != wet.shift()).cumsum()).cumcount() + 1) * wet
    df["wet_streak_h"] = df["wet_streak_h"].where(wet == 1, 0)

    df["hour"] = df["time"].dt.hour
    df["dow"]  = df["time"].dt.dayofweek
    df["doy"]  = df["time"].dt.dayofyear
    df["hoy"]  = (df["doy"] - 1) * 24 + df["hour"]

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)
    df["hoy_sin"]  = np.sin(2*np.pi*df["hoy"]/(365.25*24))
    df["hoy_cos"]  = np.cos(2*np.pi*df["hoy"]/(365.25*24))

    df["hum_x_cloud"]   = df["humidity"] * df["cloudcover"]
    df["wind_x_cloud"]  = df["wind_speed"] * df["cloudcover"]
    df["press_drop_3h"] = -df["d3_pressure"]
    df["press_drop_6h"] = df["pressure"].shift(6) - df["pressure"]

    df = df.dropna().reset_index(drop=True)

    if features_from_meta:
        missing_feats = [c for c in features_from_meta if c not in df.columns]
        if missing_feats:
            raise ValueError(f"Missing features expected by model: {missing_feats}")
        return df[features_from_meta]

    feat = (
        base +
        [f"d_{c}" for c in base] +
        [f"ma3_{c}" for c in base] +
        [f"d3_{c}" for c in ["pressure","humidity","cloudcover","temp_c"]] +
        ["dew_proxy","d_dew_proxy","ma3_dew_proxy",
         "rain_sum_3h","rain_sum_6h","rain_sum_12h","rain_sum_24h","rain_max_6h","rain_max_12h",
         "dry_streak_h","wet_streak_h",
         "hour_sin","hour_cos","dow_sin","dow_cos","hoy_sin","hoy_cos",
         "hum_x_cloud","wind_x_cloud","press_drop_3h","press_drop_6h"]
    )
    return df[feat]

# -----------------------------
# Label builder: ≥ EVENT_MM in next H hours
# -----------------------------
def make_labels(df: pd.DataFrame, horizon=H, event_mm=EVENT_MM):
    prec = df["precip_mm"].values
    y = np.zeros(len(df), dtype=int)
    for i in range(len(prec) - horizon):
        y[i] = 1 if np.nansum(prec[i+1:i+1+horizon]) >= event_mm else 0
    y = y[:-horizon]
    return y

# -----------------------------
# Models to compare
# -----------------------------
def build_models():
    models = {}

    # Logistic + StandardScaler
    models["logreg_standard"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    # Logistic + RobustScaler (outlier-robust)
    models["logreg_robust"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
        )
    except Exception as e:
        print(f"[warn] XGBoost unavailable: {e}")
    return models

def evaluate_fold(model, X_train, y_train, X_test, y_test, val_frac=0.15):
    n = len(X_train)
    v = max(int(n * val_frac), 1)
    X_tr, y_tr = X_train[:-v], y_train[:-v]
    X_val, y_val = X_train[-v:], y_train[-v:]

    model.fit(X_tr, y_tr)

    # Probability on val to pick threshold
    if hasattr(model, "predict_proba"):
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            from sklearn.preprocessing import MinMaxScaler
            z_val = model.decision_function(X_val).reshape(-1, 1)
            z_test = model.decision_function(X_test).reshape(-1, 1)
            mm = MinMaxScaler()
            p_val  = mm.fit_transform(z_val).ravel()
            p_test = mm.transform(z_test).ravel()
        else:
            # fallback: hard predictions at 0.5
            pred = model.predict(X_test)
            P, R, F1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
            return dict(P=P, R=R, F1=F1, thr=0.5)

    prec, rec, thr = precision_recall_curve(y_val, p_val)
    # Avoid degenerate thresholds: thr has length len(prec)-1
    candidates = []
    for t in thr:
        pred_v = (p_val >= t).astype(int)
        P, R, F1, _ = precision_recall_fscore_support(y_val, pred_v, average="binary", zero_division=0)
        candidates.append((t, P, R, F1))
    if not candidates:
        t_star = 0.5
    else:
        # choose by best F1 on validation
        t_star = max(candidates, key=lambda x: x[3])[0]

    pred = (p_test >= t_star).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    return dict(P=P, R=R, F1=F1, thr=float(t_star))

# -----------------------------
# Main
# -----------------------------
def main():
    if not HOURLY.exists():
        raise FileNotFoundError("results/hourly.csv not found. Run: make hourly PAST_DAYS=90")

    df = pd.read_csv(HOURLY, parse_dates=["time"])
    y_all = make_labels(df, H, EVENT_MM)
    dfX = df.iloc[:-H].copy()

    # Use features from meta if present
    features_from_meta = None
    if META.exists():
        meta = json.loads(META.read_text())
        features_from_meta = meta.get("features", None)

    Xdf = rebuild_features_like_training(dfX, features_from_meta)
    n = len(Xdf)
    if len(y_all) < n:
        raise ValueError("Labels shorter than feature matrix; check preprocessing alignment.")
    y = y_all[-n:]
    X = Xdf.values[-n:]
    assert len(X) == len(y), "Feature matrix and labels misaligned."

    tscv = TimeSeriesSplit(n_splits=5)

    models = build_models()
    results = {name: [] for name in models}

    for name, model in models.items():
        print(f"\n=== {name} ===")
        fold_id = 1
        per_fold = []
        for tr_idx, te_idx in tscv.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            metrics = evaluate_fold(model, X_tr, y_tr, X_te, y_te)
            per_fold.append(metrics)
            print(f"Fold {fold_id} → P={metrics['P']:.3f}  R={metrics['R']:.3f}  F1={metrics['F1']:.3f}  thr={metrics['thr']:.3f}")
            fold_id += 1

        # Aggregate
        Pm = np.mean([m["P"] for m in per_fold])
        Rm = np.mean([m["R"] for m in per_fold])
        Fm = np.mean([m["F1"] for m in per_fold])
        print(f"Mean  → P={Pm:.3f}  R={Rm:.3f}  F1={Fm:.3f}")
        results[name] = dict(P=Pm, R=Rm, F1=Fm)

    print("\n=== SUMMARY (higher F1 is better) ===")
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["F1"], reverse=True):
        print(f"{name:18s}  F1={m['F1']:.3f}  P={m['P']:.3f}  R={m['R']:.3f}")

if __name__ == "__main__":
    main()
