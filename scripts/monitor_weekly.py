#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, brier_score_loss

LOG = Path("logs/predictions.csv")
OUT = Path("results"); OUT.mkdir(exist_ok=True)

def week_key(ts):  # ISO year-week
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"

def calibration_plot(p, y, bins=10, out_png="results/calibration.png"):
    df = pd.DataFrame({"p":p, "y":y}).dropna()
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    g = df.groupby("bin").agg(avg_p=("p","mean"), frac_pos=("y","mean"), n=("y","size")).reset_index(drop=True)
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(g["avg_p"], g["frac_pos"], marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration")
    for i, n in enumerate(g["n"]):
        plt.annotate(str(int(n)), (g["avg_p"].iloc[i], g["frac_pos"].iloc[i]))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def main():
    if not LOG.exists():
        print("No logs yet.")
        return
    df = pd.read_csv(LOG, parse_dates=["ts_pred","logged_at"])
    df = df[df["y_true"].astype(str).isin(["0","1"])].copy()
    if df.empty:
        print("No rows with y_true yet.")
        return
    df["y_true"] = df["y_true"].astype(int)
    df["week"] = df["ts_pred"].apply(week_key)

    # Weekly metrics per mode
    rows = []
    for (wk, mode), grp in df.groupby(["week","mode"]):
        y = grp["y_true"].values
        # decision at time of logging
        yhat = (grp["p"].values >= grp["threshold"].values).astype(int)
        P,R,F1,_ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        alerts = float(yhat.mean())
        brier = brier_score_loss(y, grp["p"].values)
        rows.append({"week":wk,"mode":mode,"n":len(grp),"precision":P,"recall":R,"f1":F1,"alert_rate":alerts,"brier":brier})
    rep = pd.DataFrame(rows).sort_values(["week","mode"])
    rep.to_csv(OUT/"weekly_report.csv", index=False)
    print(rep)

    # Overall calibration (all modes combined)
    calibration_plot(df["p"].values, df["y_true"].values, bins=12, out_png=str(OUT/"calibration.png"))
    print("Saved:", OUT/"weekly_report.csv", "and", OUT/"calibration.png")

if __name__ == "__main__":
    main()
