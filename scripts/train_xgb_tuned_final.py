
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, roc_auc_score
import joblib

warnings.filterwarnings("ignore")

H = 12     
EVENT_MM = 1.0 
HOURLY = Path("results/hourly.csv")
TUNED = Path("models/xgb_tuned.json")
OUT_MODEL = Path("models/rain_xgb_tuned.joblib")
OUT_META  = Path("models/rain_xgb_tuned_meta.json")

# -----------------------------
# Label: >= EVENT_MM in next H hours
# -----------------------------
def make_labels(df: pd.DataFrame, horizon=H, event_mm=EVENT_MM):
    prec = df["precip_mm"].values
    y = np.zeros(len(df), dtype=int)
    for i in range(len(prec) - horizon):
        y[i] = 1 if np.nansum(prec[i+1:i+1+horizon]) >= event_mm else 0
    if horizon > 0:
        y = y[:-horizon]
        index = df.index[:-horizon]
    else:
        index = df.index
    return pd.Series(y, index=index)

# Feature builder (MATCH CLI/TRAINER)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time","temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Hourly data missing columns: {sorted(miss)}")

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

    is_rain = (df["precip_mm"] > 0).astype(int)
    dry = (~(is_rain.astype(bool))).astype(int)
    df["dry_streak_h"] = (dry.groupby((dry != dry.shift()).cumsum()).cumcount() + 1) * dry
    df["dry_streak_h"] = df["dry_streak_h"].where(dry == 1, 0)
    wet = is_rain
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

    feats = (
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

    Xdf = df[feats].copy()
    Xdf = Xdf.dropna()
    return Xdf

# -----------------------------
# Threshold pickers
# -----------------------------
def _eval_at_threshold(y_true, p, t):
    from sklearn.metrics import precision_recall_fscore_support
    pred = (p >= t).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    pos_rate = float(pred.mean())
    return P, R, F1, pos_rate

def pick_by_best_f1(y_true, p):
    """
    DEFAULT: balanced. 
    Requirements:
      - Precision ≥ 0.70
      - Recall ≥ 0.55
      - 0.15 ≤ positive rate ≤ 0.60
      - Final threshold floor at 0.15
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, p)
    best = None
    for t in thr:
        P, R, F1, pr = _eval_at_threshold(y_true, p, t)
        if (P >= 0.70) and (R >= 0.55) and (0.15 <= pr <= 0.60):
            if (best is None) or (F1 > best[1]):
                best = (t, F1)
    if best is not None:
        return float(max(best[0], 0.15))
    # fallback: conservative quantile
    return float(max(np.quantile(p, 0.70), 0.15))

def pick_high_recall(y_true, p):
    """
    RECALL: warn more, but not silly.
    Targets / limits:
      - Recall ≥ 0.88
      - Precision ≥ 0.60
      - positive rate ≤ 0.70
      - Final threshold floor at 0.10
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, p)
    best = None
    for t in thr:
        P, R, F1, pr = _eval_at_threshold(y_true, p, t)
        if (R >= 0.88) and (P >= 0.60) and (pr <= 0.70):
            if (best is None) or (R, F1) > (best[1], best[2]):
                best = (t, R, F1)
    if best is not None:
        return float(max(best[0], 0.10))
    # fallback: lower-but-not-silly quantile
    return float(max(np.quantile(p, 0.60), 0.10))

def pick_high_precision(y_true, p):
    """
    PRECISION: be picky.
    Requirements:
      - Precision ≥ 0.85
      - Recall ≥ 0.40
      - positive rate ≤ 0.50
      - Final threshold floor at 0.60
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, p)
    best = None
    for t in thr:
        P, R, F1, pr = _eval_at_threshold(y_true, p, t)
        if (P >= 0.85) and (R >= 0.40) and (pr <= 0.50):
            if (best is None) or (P, F1) > (best[1], best[2]):
                best = (t, P, F1)
    if best is not None:
        return float(max(best[0], 0.60))
    return float(max(np.quantile(p, 0.90), 0.60))

# -----------------------------
# Per-fold evaluation with time-aware val slice
# -----------------------------
def eval_timeseries_cv(model, X, y, n_splits=5, val_frac=0.15):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        v = max(int(len(X_tr) * val_frac), 1)
        X_fit, y_fit = X_tr[:-v], y_tr[:-v]
        X_val, y_val = X_tr[-v:],  y_tr[-v:]

        model.fit(X_fit, y_fit)
        p_val = model.predict_proba(X_val)[:,1]
        p_te  = model.predict_proba(X_te)[:,1]

        thr = pick_by_best_f1(y_val, p_val)
        pred = (p_te >= thr).astype(int)
        P,R,F1,_ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
        try: auc = roc_auc_score(y_te, p_te)
        except: auc = float("nan")
        scores.append(dict(P=P,R=R,F1=F1,AUC=auc,thr=float(thr)))
        print(f"Fold {fold+1} → P={P:.3f} R={R:.3f} F1={F1:.3f} thr={thr:.3f}")
    mean = {k: float(np.mean([s[k] for s in scores])) for k in ["P","R","F1","AUC"]}
    print(f"Mean → P={mean['P']:.3f} R={mean['R']:.3f} F1={mean['F1']:.3f} AUC={mean['AUC']:.3f}")
    return scores, mean

# -----------------------------
# Main
# -----------------------------
def main():
    if not HOURLY.exists():
        raise FileNotFoundError("results/hourly.csv not found. Run: make hourly PAST_DAYS=90")
    if not TUNED.exists():
        raise FileNotFoundError("models/xgb_tuned.json not found. Run: python scripts/xgb_tune_timeseries.py")

    df = pd.read_csv(HOURLY, parse_dates=["time"])

    # Build labels first
    y_series = make_labels(df, H, EVENT_MM)

    # Build features on all rows except final H (no future window)
    dfX = df.iloc[:-H].copy()
    Xdf = build_features(dfX)              # keeps index
    y_aligned = y_series.loc[Xdf.index]    # align by index
    X = Xdf.values
    y = y_aligned.values

    # Load tuned params
    tuned = json.loads(TUNED.read_text())["params"]
    clf = XGBClassifier(
        **tuned,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
    )

    print("⚙️  Cross-validating tuned XGB…")
    cv_scores, cv_mean = eval_timeseries_cv(clf, X, y, n_splits=5, val_frac=0.15)

    # Fit on all data (with small tail held as validation for threshold picking)
    v = max(int(len(X) * 0.20), 1)
    X_fit, y_fit = X[:-v], y[:-v]
    X_val, y_val = X[-v:],  y[-v:]

    clf.fit(X_fit, y_fit)
    p_val = clf.predict_proba(X_val)[:,1]

    thr_default = pick_by_best_f1(y_val, p_val)
    thr_recall  = pick_high_recall(y_val, p_val)
    thr_prec    = pick_high_precision(y_val, p_val)

    # Save model + meta
    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, OUT_MODEL)

    meta = {
        "features": list(Xdf.columns),
        "horizon_hours": H,
        "event_mm": EVENT_MM,
        "model": {"type": "xgboost", "params": tuned},
        "thresholds": {
            "default": float(thr_default),
            "high_recall": float(thr_recall),
            "high_precision": float(thr_prec),
        },
        "cv_mean": cv_mean,
        "cv_folds": cv_scores,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print("\n💾 Saved", OUT_MODEL, "and", OUT_META)
    print(f"Chosen thresholds → default={thr_default:.3f}  recall={thr_recall:.3f}  precision={thr_prec:.3f}")

if __name__ == "__main__":
    main()
