
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split

meta = json.load(open("models/rain_model_meta.json"))
clf = joblib.load("models/rain_classifier_hourly.joblib")

df = pd.read_csv("results/hourly.csv", parse_dates=["time"])
base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
for c in base:
    df[f"d_{c}"] = df[c].diff()
    df[f"ma3_{c}"] = df[c].rolling(3).mean()
df = df.dropna().reset_index(drop=True)

X = df[meta["features"]].values
y = None

logreg = clf.named_steps["logreg"]
coefs = logreg.coef_[0] 
features = meta["features"]

rank = sorted(zip(features, coefs), key=lambda x: abs(x[1]), reverse=True)

out_lines = ["Feature coefficients (standardized space):"]
for name, w in rank[:15]:
    out_lines.append(f"{name:20s} {w:+.3f}")

print("\n".join(out_lines))
with open("results/coef_top15.txt", "w") as f:
    f.write("\n".join(out_lines))

print("✅ Wrote results/coef_top15.txt")
