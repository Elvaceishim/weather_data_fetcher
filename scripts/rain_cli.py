
import argparse, json, joblib, pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Rain warning in next 6h")
    ap.add_argument("--mode", choices=["recall","precision","default"], default="recall")
    args = ap.parse_args()

    meta = json.load(open("models/rain_model_meta.json"))
    clf = joblib.load("models/rain_classifier_hourly.joblib")

    df = pd.read_csv("results/hourly.csv", parse_dates=["time"])
    row = df.iloc[-1:].copy()

    # rebuild features like training
    base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
    for col in base:
        row[f"d_{col}"] = df[col].diff().iloc[-1]
        row[f"ma3_{col}"] = df[col].rolling(3).mean().iloc[-1]

    X = row[meta["features"]].values
    p = float(clf.predict_proba(X)[0,1])

    thr = {
        "default": meta["thresholds"]["default"],
        "recall": meta["thresholds"]["high_recall"],
        "precision": meta["thresholds"]["high_precision"],
    }[args.mode]

    decision = "RAIN" if p >= thr else "No rain"
    print(f"{row['time'].iloc[0]}  |  P(rain ≤{meta['horizon_hours']}h)={p:.3f}  |  mode={args.mode} thr={thr:.2f}  →  {decision}")

if __name__ == "__main__":
    main()
