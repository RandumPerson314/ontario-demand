"""Runs once per invocation (GitHub Actions calls it every 30 minutes)."""
import json
import os

import joblib
import pandas as pd

from features import FEATURES, fetch_demand, fetch_live, make_features, predict_demand

LOG = "docs/demand_log"
KEEP = 24 * 30

model = joblib.load("model/model.joblib")
now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
dem = fetch_demand([now.year - 1, now.year])
F = make_features(fetch_live(), dem)

log = {}
if os.path.exists(LOG + ".json"):
    with open(LOG + ".json") as fh:
        log = {r["t"]: r for r in json.load(fh)}

use = F.loc[now - pd.DateOffset(hours=72):now].dropna(subset=FEATURES)
for idx, p, temp in zip(use.index, predict_demand(model, use), use["temp"]):
    key = (idx - pd.DateOffset(hours=5)).strftime("%Y-%m-%dT%H:00")   # EST, like IESO
    if key not in log:                      # predictions are frozen once made
        log[key] = dict(t=key, pred=round(float(p), 1), actual=None,
                        temp=round(float(temp), 1), bf=int(idx < now))

for r in log.values():                      # fill in actuals as IESO publishes them
    ts = pd.Timestamp(r["t"]) + pd.DateOffset(hours=5)
    if r["actual"] is None and ts in dem.index:
        r["actual"] = float(dem[ts])

rows = sorted(log.values(), key=lambda r: r["t"])[-KEEP:]
os.makedirs("docs", exist_ok=True)
with open(LOG + ".json", "w") as fh:
    json.dump(rows, fh)
with open(LOG + ".js", "w") as fh:
    fh.write("window.LOG=" + json.dumps(rows) + ";")
print(f"{len(rows)} rows, latest prediction {rows[-1]['pred']:.0f} MW at {rows[-1]['t']}")
