"""One update run (GitHub Actions calls this twice an hour).

Writes docs/demand_log.js (past predictions vs actuals + live demand) and
docs/forecast.js (7-day forecast with the feature values the website explorer needs)."""
import json
import os

import joblib
import pandas as pd

from features import (NOW_FEATURES, TORONTO, WEEK_FEATURES, fetch_demand, fetch_live,
                      fetch_live_demand, make_features, predict_now, predict_week)

LOG = "docs/demand_log"
KEEP = 24 * 30

now_model = joblib.load("model/now.joblib")
week_model = joblib.load("model/week.joblib")

now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")   # current hour, naive UTC
dem = fetch_demand([now.year - 1, now.year])
F = make_features(fetch_live(), dem)


def local_label(u, fmt="%Y-%m-%dT%H:00"):
    return u.tz_localize("UTC").tz_convert(TORONTO).strftime(fmt)


def ukey(u):
    return u.strftime("%Y-%m-%dT%H:00")


# ---------------------------------------------------------------- 1. past + current hour log
log = {}
if os.path.exists(LOG + ".json"):
    with open(LOG + ".json") as fh:
        log = {r["u"]: r for r in json.load(fh) if "u" in r}     # old-format rows are dropped

use = F.loc[now - pd.DateOffset(hours=72):now].dropna(subset=NOW_FEATURES)
for idx, p, temp in zip(use.index, predict_now(now_model, use), use["temp"]):
    k = ukey(idx)
    if k not in log:                                   # predictions are frozen once made
        log[k] = dict(u=k, t=local_label(idx), pred=round(float(p), 1), actual=None,
                      temp=round(float(temp), 1), bf=int(idx < now))

for r in log.values():                                 # fill in actuals as IESO publishes them
    ts = pd.Timestamp(r["u"])
    if r["actual"] is None and ts in dem.index:
        r["actual"] = float(dem[ts])

rows = sorted(log.values(), key=lambda r: r["u"])[-KEEP:]

meta = {"updated": pd.Timestamp.now(tz="UTC").isoformat()}
try:                                                   # latest 5-minute reading (IESO times are EST)
    live = fetch_live_demand()
    ts = live["timestamp"] + pd.DateOffset(hours=5)
    meta.update(live_mw=live["demand"], live_t=local_label(ts, "%Y-%m-%dT%H:%M"))
    print(f"IESO live demand: {live['demand']:.0f} MW at {meta['live_t']} Toronto time")
except Exception as e:
    print("live demand unavailable:", e)

os.makedirs("docs", exist_ok=True)
with open(LOG + ".json", "w") as fh:
    json.dump(rows, fh)
with open(LOG + ".js", "w") as fh:
    fh.write("window.LOG=" + json.dumps(rows) + ";window.LOG_META=" + json.dumps(meta) + ";")
print(f"log: {len(rows)} rows; latest {rows[-1]['t']} predicted {rows[-1]['pred']:.0f} MW")

# ---------------------------------------------------------------- 2. 7-day forecast
fc = F.loc[now:now + pd.DateOffset(hours=167)].dropna(subset=WEEK_FEATURES)
fc_pred = predict_week(week_model, fc)
out = dict(updated=meta["updated"],
           u=[ukey(i) for i in fc.index], t=[local_label(i) for i in fc.index],
           pred=[round(float(p)) for p in fc_pred], temp=[round(float(v), 1) for v in fc["temp"]],
           X=[[round(float(v), 4) for v in r] for r in fc[WEEK_FEATURES].values])
with open("docs/forecast.js", "w") as fh:
    fh.write("window.FC=" + json.dumps(out, separators=(",", ":")) + ";")
print(f"forecast: {len(fc)} hours, from {out['t'][0]} to {out['t'][-1]} Toronto time")
