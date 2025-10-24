import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

RESULTS_DIR = "results"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RESULTS_DIR, ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RESULTS_DIR, ".cache"))

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_dataset(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv("results/hourly.csv", parse_dates=["time"])
    horizon = meta["horizon_hours"]

    precip = df["precip_mm"].values
    rain_future = np.zeros(len(df), dtype=int)
    for i in range(len(precip) - horizon):
        rain_future[i] = 1 if np.any(precip[i + 1 : i + 1 + horizon] > 0) else 0

    df = df.iloc[: len(precip) - horizon].copy()
    labels = rain_future[: len(df)]
    features = df[meta["features"]].values
    return features, labels


def plot_importance(feature_names: list[str], importances: np.ndarray, std: np.ndarray) -> None:
    order = np.argsort(importances)[::-1]
    feature_names = np.array(feature_names)[order]
    importances = importances[order]

    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importances, align="center")
    plt.yticks(y_pos, feature_names)
    plt.gca().invert_yaxis()
    plt.xlabel("Permutation importance (F1 drop)")
    plt.title("Rain classifier — feature importances")
    plt.tight_layout()

    Path(RESULTS_DIR).mkdir(exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))
    plt.close()


def main() -> None:
    meta = json.load(open("models/rain_model_meta.json"))
    model = joblib.load("models/rain_classifier_hourly.joblib")

    X, y = build_dataset(meta)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=25,
        random_state=42,
        scoring="f1",
    )

    plot_importance(meta["features"], result.importances_mean, result.importances_std)
    print("✅ Wrote results/feature_importance.png")


if __name__ == "__main__":
    main()
