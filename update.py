"""One update run (GitHub Actions calls this twice an hour).

Writes docs/demand_log.js      past hours: predictions from each model + IESO actuals + live demand
       docs/forecast.js        the 7-day forecast (both models) + feature values for the explorer
       docs/forecast_archive.js  one frozen 7-day forecast per day, to measure real forecast accuracy"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from features import (NEXT_FEATURES, NOW_FEATURES, POLY_FEATURES, TORONTO, WEEK_FEATURES,
                      fetch_demand, fetch_live, fetch_live_demand, make_features,
                      predict_next, predict_now, predict_week)

LOG, ARCH = "docs/demand_log", "docs/forecast_archive.js"
KEEP = 24 * 30

now_m, next_m, week_m, poly_m = (joblib.load(f"model/{n}.joblib") for n in ("now", "next", "week", "poly"))

# DEMAND_NOW lets you re-run the updater "as if" it were another hour (for testing).
now = (pd.Timestamp(os.environ["DEMAND_NOW"]) if "DEMAND_NOW" in os.environ
       else pd.Timestamp.now(tz="UTC").tz_localize(None)).floor("h")      # current hour, naive UTC
nxt = now + pd.DateOffset(hours=1)
dem = fetch_demand([now.year - 1, now.year])
F = make_features(fetch_live(), dem)


def local_label(u, fmt="%Y-%m-%dT%H:00"):
    return u.tz_localize("UTC").tz_convert(TORONTO).strftime(fmt)


def ukey(u):
    return u.strftime("%Y-%m-%dT%H:00")


# ---------------------------------------------------------------- 1. hourly log
log = {}
if os.path.exists(LOG + ".json"):
    with open(LOG + ".json") as fh:
        log = {r["u"]: {"pn": None, "pp": None, **r}              # rows from older versions get the new fields
               for r in json.load(fh) if "u" in r}                # rows without a UTC key are dropped


def get_row(idx):
    k = ukey(idx)
    if k not in log:       # a row created ahead of time (as the "next hour") is the only live kind
        log[k] = dict(u=k, t=local_label(idx), pred=None, pn=None, pp=None, actual=None,
                      temp=round(float(F.at[idx, "temp"]), 1), bf=0 if idx == nxt else 1)
    return log[k]


def fill(row, key, value, live):
    """Store a prediction once (they are frozen); anything not made on time marks the row as retrospective."""
    if row[key] is None:
        row[key] = round(float(value), 1)
        if not live:
            row["bf"] = 1


win = F.loc[now - pd.DateOffset(hours=72):nxt]
P = win.dropna(subset=POLY_FEATURES)                                 # polynomial (weather + calendar only)
for idx, p in zip(P.index, poly_m.predict(P[POLY_FEATURES])):
    fill(get_row(idx), "pp", p, idx == nxt)
U = win.loc[:now].dropna(subset=NOW_FEATURES)                        # "this hour": knows demand through last hour
for idx, p in zip(U.index, predict_now(now_m, U)):
    fill(get_row(idx), "pred", p, idx == now)
N = win.dropna(subset=NEXT_FEATURES)                                 # "next hour": same data, one hour further ahead
for idx, p in zip(N.index, predict_next(next_m, N)):
    fill(get_row(idx), "pn", p, idx == nxt)

for r in log.values():                                               # fill in actuals as IESO publishes them
    ts = pd.Timestamp(r["u"])
    if r["actual"] is None and ts in dem.index:
        r["actual"] = float(dem[ts])

rows = sorted(log.values(), key=lambda r: r["u"])[-KEEP:]

# ---------------------------------------------------------------- 2. live demand (hourly)
meta = {"updated": pd.Timestamp.now(tz="UTC").isoformat()}
try:   # average of the 5-minute readings so far in the current IESO hour (IESO times are EST)
    live = fetch_live_demand()
    vals = [r["demand"] for r in live["rows"]]
    meta.update(live_mw=round(float(np.mean(vals)), 1), live_n=len(vals),
                live_t=local_label(live["hour_start"] + pd.DateOffset(hours=5)),
                live_last=local_label(live["timestamp"] + pd.DateOffset(hours=5), "%H:%M"))
    print(f"IESO demand so far in the {meta['live_t']} hour: {meta['live_mw']:.0f} MW ({len(vals)} readings)")
except Exception as e:
    print("live demand unavailable:", e)

os.makedirs("docs", exist_ok=True)
with open(LOG + ".json", "w") as fh:
    json.dump(rows, fh)
with open(LOG + ".js", "w") as fh:
    fh.write("window.LOG=" + json.dumps(rows) + ";window.LOG_META=" + json.dumps(meta) + ";")
cur = next((r for r in rows if r["u"] == ukey(now)), rows[-1])
print(f"log: {len(rows)} rows; this hour ({cur['t']}): {cur['pred']} MW; next hour: "
      f"{next((r['pn'] for r in rows if r['u'] == ukey(nxt)), None)} MW")

# ---------------------------------------------------------------- 3. 7-day forecast
fc = F.loc[now:now + pd.DateOffset(hours=167)].dropna(subset=WEEK_FEATURES + POLY_FEATURES)
out = dict(updated=meta["updated"], u=[ukey(i) for i in fc.index], t=[local_label(i) for i in fc.index],
           pred=[round(float(p)) for p in predict_week(week_m, fc)],
           pp=[round(float(p)) for p in poly_m.predict(fc[POLY_FEATURES])],
           temp=[round(float(v), 1) for v in fc["temp"]],
           X=[[round(float(v), 4) for v in r] for r in fc[WEEK_FEATURES].values])
with open("docs/forecast.js", "w") as fh:
    fh.write("window.FC=" + json.dumps(out, separators=(",", ":")) + ";")
print(f"forecast: {len(fc)} hours, from {out['t'][0]} to {out['t'][-1]} Toronto time")

# one frozen forecast per UTC day, so real 7-day accuracy can be measured as the days pass
arch = []
if os.path.exists(ARCH):
    with open(ARCH) as fh:
        arch = json.loads(fh.read().strip()[len("window.ARCH="):].rstrip(";"))
if len(fc) and not any(a["u0"][:10] == ukey(now)[:10] for a in arch):
    arch.append(dict(u0=out["u"][0], gb=out["pred"], pp=out["pp"]))
with open(ARCH, "w") as fh:
    fh.write("window.ARCH=" + json.dumps(arch[-21:], separators=(",", ":")) + ";")
