"""One update run (GitHub Actions calls this every 15 minutes).

Writes docs/demand_log.js       recent hours: every model's predictions, IESO's forecast, actuals, status
       docs/forecast.js         the 7-day forecast of every model (+ IESO) and the explorer's inputs
       docs/forecast_archive.js one frozen 7-day forecast per day, to measure real forecast accuracy"""
import json
import os

import joblib
import numpy as np
import pandas as pd

import features as ft
from features import (FAMILIES, TORONTO, fetch_adequacy, fetch_demand, fetch_live,
                      fetch_live_demand, fetch_multiday, make_features, merge_demand, variants)

LOG, FC_FILE, ARCH = "docs/demand_log.js", "docs/forecast.js", "docs/forecast_archive.js"
KEEP = 24 * 14
FIDS = list(FAMILIES)
KEYS = [f"{fid}{s}" for fid in FIDS for s in ("_h", "_n")] + ["ieso_h", "ieso_n"]   # _h = this hour, _n = next hour
status = []


def note(name, ok, detail=""):
    status.append(dict(name=name, ok=bool(ok), detail=str(detail)[:220]))
    print(("ok   " if ok else "FAIL ") + f"{name}: {detail}")


def read_js(path, prefix):
    if not os.path.exists(path):
        return None
    txt = open(path).read().strip()
    return json.JSONDecoder().raw_decode(txt[txt.index(prefix) + len(prefix):])[0]


def write_log(rows, meta):
    os.makedirs("docs", exist_ok=True)
    with open(LOG, "w") as fh:
        fh.write("window.LOG=" + json.dumps(rows, separators=(",", ":")) + ";window.LOG_META=" + json.dumps(meta) + ";")


def local_label(u, fmt="%Y-%m-%dT%H:00"):
    return u.tz_localize("UTC").tz_convert(TORONTO).strftime(fmt)


def ukey(u):
    return u.strftime("%Y-%m-%dT%H:00")


# ---------------------------------------------------------------- load previous log (and migrate old formats)
log = {}
for r in read_js(LOG, "window.LOG=") or []:
    if "u" not in r:
        continue
    for old, new in (("pred", "gbm_h"), ("pn", "gbm_n"), ("pp", "poly0_n")):
        if old in r:
            r.setdefault(new, r.pop(old))
    for k in KEYS:
        r.setdefault(k, None)
    log[r["u"]] = r
prev_meta = read_js(LOG, "window.LOG_META=") or {}

try:
    MODELS = {(fid, v): joblib.load(f"model/{fid}_{v}.joblib") for fid in FIDS for v in variants(fid)}
except Exception as e:
    note("Models", False, f"could not load saved models ({e}); run train_model.py")
    meta = {**prev_meta, "status": status}
    write_log(sorted(log.values(), key=lambda r: r["u"]), meta)
    raise


def model_for(fid, v):
    return MODELS[(fid, v if FAMILIES[fid]["demand"] else "all")]


now = (pd.Timestamp(os.environ["DEMAND_NOW"]) if "DEMAND_NOW" in os.environ
       else pd.Timestamp.now(tz="UTC").tz_localize(None)).floor("h")      # current hour, naive UTC
nxt = now + pd.DateOffset(hours=1)

# ---------------------------------------------------------------- data sources (each one may fail on its own)
csv = fetch_demand([now.year - 1, now.year])
note("IESO hourly file", len(csv) > 0, f"newest hour {local_label(csv.index.max())} Toronto time" if len(csv) else "no data")

multi, proj, adeq = None, pd.Series(dtype=float), pd.Series(dtype=float)
try:
    multi = fetch_multiday(csv)
    proj = multi["projected"]
    note("IESO live hourly feed", True, multi["info"])
except Exception as e:
    note("IESO live hourly feed", False, e)
try:
    adeq, info = fetch_adequacy()
    note("IESO forecast report", True, info)
except Exception as e:
    note("IESO forecast report", False, e)
ieso = proj.combine_first(adeq)                    # IESO's official forecast: short-range feed, then the adequacy report

dem = merge_demand(csv, multi["actual"] if multi else None, now)
try:
    weather = fetch_live()
    note("Weather (Open-Meteo)", ft.LIVE_STATUS["ok"], ft.LIVE_STATUS["detail"])
except Exception as e:
    note("Weather (Open-Meteo)", False, f"unavailable and no saved copy: {e}")
    write_log(sorted(log.values(), key=lambda r: r["u"]), {**prev_meta, "status": status})   # keeps the old 'updated' time
    raise
F = make_features(weather, dem)


# ---------------------------------------------------------------- hourly log
def get_row(idx):
    k = ukey(idx)
    if k not in log:       # a row created ahead of time (as the "next hour") is the only live kind
        log[k] = dict(u=k, t=local_label(idx), actual=None, temp=round(float(F.at[idx, "temp"]), 1),
                      bf=0 if idx == nxt else 1, **{key: None for key in KEYS})
    return log[k]


def fill(row, key, value, live):
    """Store a prediction once (they are frozen); one not made on time marks the row as retrospective."""
    if row[key] is None:
        row[key] = round(float(value), 1)
        if not live:
            row["bf"] = 1


win = F.loc[now - pd.DateOffset(hours=72):nxt]
for fid in FIDS:
    for suffix, v, end in (("_h", "h1", now), ("_n", "h2", nxt)):
        m = model_for(fid, v)
        X = win.loc[:end].dropna(subset=m.feats)
        if len(X):
            for idx, p in zip(X.index, m.predict(X)):
                fill(get_row(idx), fid + suffix, p, idx == end)
for idx, key in ((now, "ieso_h"), (nxt, "ieso_n")):    # IESO's forecast, captured when it was a genuine forecast
    if idx in ieso.index and idx in F.index and not np.isnan(ieso[idx]) and get_row(idx)[key] is None:
        log[ukey(idx)][key] = round(float(ieso[idx]), 1)
for idx in win.index:                                  # past hours: the IESO's own published projection (its chart feed keeps ~5 days)
    if idx < now and idx in proj.index and not np.isnan(proj[idx]):
        r = get_row(idx)
        for key in ("ieso_h", "ieso_n"):
            if r[key] is None:
                r[key] = round(float(proj[idx]), 1)

for r in log.values():                                 # actuals: official hourly file first, then the live feed
    ts = pd.Timestamp(r["u"])
    if ts in dem.index:
        r["actual"] = float(dem[ts])
rows = sorted(log.values(), key=lambda r: r["u"])[-KEEP:]

# ---------------------------------------------------------------- live demand so far this hour
meta = {"updated": pd.Timestamp.now(tz="UTC").isoformat()}
try:   # average of the 5-minute readings so far in the current IESO hour (IESO times are EST)
    live = fetch_live_demand()
    vals = [r["demand"] for r in live["rows"]]
    meta.update(live_mw=round(float(np.mean(vals)), 1), live_n=len(vals),
                live_t=local_label(live["hour_start"] + pd.DateOffset(hours=5)),
                live_last=local_label(live["timestamp"] + pd.DateOffset(hours=5), "%H:%M"))
    note("IESO 5-minute readings", True, f"{len(vals)} readings, hour starting {meta['live_t'][11:]}")
except Exception as e:
    note("IESO 5-minute readings", False, e)
meta["status"] = status
meta["ieso_hours"] = sum(1 for r in rows if r.get("ieso_n") is not None)
meta["ieso_source"] = ("live feed projection" if len(proj) else "") + (" + adequacy report" if len(adeq) else "") or "none"
write_log(rows, meta)
cur = next((r for r in rows if r["u"] == ukey(now)), None)
print("this hour:", {k: cur[k] for k in KEYS if cur and cur[k] is not None} if cur else None)

# ---------------------------------------------------------------- 7-day forecast (rewritten once an hour)
prev_fc = read_js(FC_FILE, "window.FC=")
if prev_fc and prev_fc.get("u") and prev_fc["u"][0] == ukey(now) and "m" in prev_fc:
    print("7-day forecast already written for this hour")
else:
    names = sorted({c for fid in FIDS for c in model_for(fid, "wk").feats})     # every input any model's explorer needs
    fc = F.loc[now:now + pd.DateOffset(hours=167)].dropna(subset=names)
    if len(fc):
        M = {fid: [round(float(p)) for p in model_for(fid, "wk").predict(fc)] for fid in FIDS}
        off = [None if i not in ieso.index or np.isnan(ieso[i]) else round(float(ieso[i])) for i in fc.index]
        out = dict(updated=meta["updated"], u=[ukey(i) for i in fc.index], t=[local_label(i) for i in fc.index],
                   pred=M["gbm"], m=M, ieso=off, temp=[round(float(v), 1) for v in fc["temp"]],
                   names=names, X=[[round(float(v), 4) for v in r] for r in fc[names].values])
        with open(FC_FILE, "w") as fh:
            fh.write("window.FC=" + json.dumps(out, separators=(",", ":")) + ";")
        print(f"forecast: {len(fc)} hours, {out['t'][0]} to {out['t'][-1]} Toronto time")

        arch = read_js(ARCH, "window.ARCH=") or []        # one frozen forecast per UTC day
        for a in arch:
            if "gb" in a:
                a["m"] = {"gbm": a.pop("gb"), "poly0": a.pop("pp", None)}
        if not any(a["u0"][:10] == ukey(now)[:10] for a in arch):
            arch.append(dict(u0=out["u"][0], m=M, ieso=off))
        with open(ARCH, "w") as fh:
            fh.write("window.ARCH=" + json.dumps(arch[-21:], separators=(",", ":")) + ";")
    else:
        print("no complete forecast rows (missing inputs)")
