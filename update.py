"""One update run (GitHub Actions calls this every 15 minutes).

Writes docs/demand_log.js       recent hours: every model's predictions, IESO's forecast, actuals, status
       docs/forecast.js         the 7-day forecast of every model (+ IESO) and the explorer's inputs
       docs/forecast_archive.js one frozen 7-day forecast per day, to measure real forecast accuracy"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

import features as ft
from features import (FAMILIES, TORONTO, fetch_adequacy, fetch_demand, fetch_live,
                      fetch_live_demand, fetch_recent_hours, make_features, merge_demand, variants)

LOG, FC_FILE, ARCH = "docs/demand_log.js", "docs/forecast.js", "docs/forecast_archive.js"
KEEP = 24 * 14
FIDS = list(FAMILIES)
KEYS = [f"{fid}{s}" for fid in FIDS for s in ("_h", "_n")] + ["ieso_h", "ieso_n"]   # _h = this hour, _n = next hour
status = []
T0 = time.monotonic()


def note(name, ok, detail=""):
    status.append(dict(name=name, ok=bool(ok), detail=str(detail)[:500]))
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
BACKFILL = "data/backfill_ieso.json"            # one-off reconstruction of past hours (tools/backfill_ieso_times.py)
if os.path.exists(BACKFILL):
    for k, rec in json.load(open(BACKFILL)).items():
        if k in log and log[k].get("ieso_h") is not None and "ip" not in log[k]:
            log[k].update(rec)
    os.remove(BACKFILL)

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

recent, adeq = None, pd.Series(dtype=float)
try:
    recent, info = fetch_recent_hours(csv, now)
    note("IESO recent hours (5-minute reports)", True, info)
except Exception as e:
    note("IESO recent hours (5-minute reports)", False, e)
dem = merge_demand(csv, recent, now)
try:
    adeq, info = fetch_adequacy(actual=dem, days=3, today=now)   # the undated report is for a day weeks ahead; the dated files cover today and the next two days
    note("IESO forecast report", True, info)
except Exception as e:
    note("IESO forecast report", False, e)
ieso = adeq                                        # IESO's official forecast (the adequacy report)
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


WALL = pd.Timestamp.now(tz="UTC").tz_localize(None)    # when this run happened (for the logs page)


def stamp(row, key):
    """Remember when a value was saved: minutes after the start of the row's hour (negative = before the hour)."""
    row.setdefault("at", {})[key] = int((WALL - pd.Timestamp(row["u"])).total_seconds() // 60)


def fill(row, key, value, live):
    """Store a prediction once (they are frozen); one not made on time marks the row as retrospective."""
    if row[key] is None:
        row[key] = round(float(value), 1)
        stamp(row, key)
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
# ---------------------------------------------------------------- the IESO's forecast: when it was published, and
# what each model was predicting at that moment. The IESO republishes its forecast every 30 minutes; for every
# upcoming hour we track its current value, when that value first appeared, and the history of each model's
# latest prediction for the hour (7-day run, then next-hour, then this-hour). When the IESO's number for an hour
# is saved, the models' predictions made at or before its publication are saved beside it.
TRACK = "data/ieso_track.json"
track = json.load(open(TRACK)) if os.path.exists(TRACK) else {}
fc_prev = read_js(FC_FILE, "window.FC=") or {}
fc_ix = {k: i for i, k in enumerate(fc_prev.get("u", []))}
iso = lambda t: pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


def latest_model_values(k):
    """Each model's newest prediction for hour k as of this run: (made, value, source)."""
    r, out = log.get(k), {}
    for fid in FIDS:
        got = None
        for sfx, src in (("_h", "h"), ("_n", "n")):
            if r and r.get(fid + sfx) is not None:
                at = (r.get("at") or {}).get(fid + sfx)
                made = pd.Timestamp(k) + pd.DateOffset(minutes=at) if at is not None else WALL
                got = (iso(made), r[fid + sfx], src)
                break
        if got is None and k in fc_ix and fc_prev.get("m", {}).get(fid):
            v = fc_prev["m"][fid][fc_ix[k]]
            if v is not None:
                got = (iso(pd.Timestamp(fc_prev["updated"]).tz_convert("UTC").tz_localize(None)), float(v), "7d")
        if got:
            out[fid] = got
    return out


older = {}
for idx in sorted(ft.ADEQ_SRC):
    k = ukey(idx)
    if idx < now or idx not in ieso.index or np.isnan(ieso[idx]) or (k in log and log[k].get("ieso_h") is not None):
        continue
    v, pub = round(float(ieso[idx]), 1), ft.ADEQ_PUB.get(idx)
    e = track.get(k)
    if e is None or e["v"] != v:                     # new hour, or the IESO changed its number: a new "first published"
        first, file = int(e is None), ft.ADEQ_SRC[idx]
        if e is None:                                # first sight: the number may be older; look back through versions
            if file not in older:
                older[file] = ft.adequacy_versions_before(file)
            he = (idx - pd.DateOffset(hours=5)).hour + 1
            for f, p, vals in older[file]:
                if vals.get(he) != v:
                    first = 0                        # found the version before it changed: the time is exact
                    break
                file, pub = f, p
        e = dict(v=v, pub=iso(pub) if pub is not None else None, first=first, file=file, m=(e or {}).get("m", {}))
        track[k] = e
    for fid, item in latest_model_values(k).items():
        hist = e["m"].setdefault(fid, [])
        if not hist or hist[-1][1:] != list(item[1:]):
            hist.append(list(item))
            del hist[:-8]
track = {k: e for k, e in track.items() if k >= ukey(now - pd.DateOffset(hours=2))}


def models_at_ieso(r):
    """Copy, into the row, when the IESO's saved number was first published and each model's prediction then."""
    e = track.get(r["u"])
    if not e or e["v"] != r["ieso_h"] or not e.get("pub"):
        return
    start, T = pd.Timestamp(r["u"]), pd.Timestamp(e["pub"])
    rel = lambda t: int((pd.Timestamp(t) - start).total_seconds() // 60)
    r.update(ip=rel(T), ipf=e["first"], iv=e["file"], mi={}, mim={}, mis={})
    late = 0
    for fid, hist in e["m"].items():
        before = [h for h in hist if pd.Timestamp(h[0]) <= T]
        made, v, src = before[-1] if before else hist[0]
        late |= not before
        r["mi"][fid], r["mim"][fid], r["mis"][fid] = round(v, 1), rel(made), src
    if late:
        r["mil"] = 1        # some model had nothing saved before the IESO published (tracking began later)


for idx, key in ((now, "ieso_h"), (nxt, "ieso_n")):    # IESO's forecast, captured when it was a genuine forecast
    if idx in ieso.index and idx in F.index and not np.isnan(ieso[idx]) and get_row(idx)[key] is None:
        r = log[ukey(idx)]
        r[key] = round(float(ieso[idx]), 1)
        stamp(r, key)
        if idx in ft.ADEQ_SRC:                         # the IESO file the number was read from
            r.setdefault("f", {})[key] = ft.ADEQ_SRC[idx]
        if key == "ieso_h":
            models_at_ieso(r)
os.makedirs("data", exist_ok=True)
json.dump(track, open(TRACK, "w"), separators=(",", ":"))
for r in log.values():                                 # actuals: official hourly file first, then the 5-minute reports
    ts = pd.Timestamp(r["u"])
    if ts in dem.index:
        r["actual"] = float(dem[ts])
        r["src"] = "csv" if ts in csv.index else "5min"      # yearly file, or the hour's 5-minute report
rows = sorted(log.values(), key=lambda r: r["u"])[-KEEP:]


# ---------------------------------------------------------------- whole-project archive of every scored hour
# The log above keeps 14 days; this compact copy of every completed hour is never trimmed, so accuracy can be
# compared across the whole project. Each run refreshes the hours still in the log (actuals can be refined).
ARC_FILE = "docs/score_archive.js"
SEED = "data/archive_seed.json"                 # every hour logged before the archive existed (from git history)
arc = read_js(ARC_FILE, "window.ARC=") or (json.load(open(SEED)) if os.path.exists(SEED) else {"ids": FIDS, "rows": []})
if os.path.exists(SEED):
    os.remove(SEED)
by_u = {a["u"]: a for a in arc["rows"]}
for r in rows:
    if r.get("actual") is None or r["u"] >= ukey(now):
        continue
    a = dict(u=r["u"], a=r["actual"], bf=r.get("bf", 1), h=[r.get(f + "_h") for f in FIDS],
             n=[r.get(f + "_n") for f in FIDS], i=r.get("ieso_h"))
    if r.get("mi"):
        a.update(ip=r.get("ip"), ipf=r.get("ipf", 0), m=[r["mi"].get(f) for f in FIDS], mm=[r["mim"].get(f) for f in FIDS])
        if r.get("mil"):
            a["ml"] = 1
    by_u[r["u"]] = a
with open(ARC_FILE, "w") as fh:
    fh.write("window.ARC=" + json.dumps({"ids": FIDS, "rows": [by_u[k] for k in sorted(by_u)]}, separators=(",", ":")) + ";")

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
meta["ieso_source"] = "adequacy report" if len(adeq) else "none"
write_log(rows, meta)
cur = next((r for r in rows if r["u"] == ukey(now)), None)
print("this hour:", {k: cur[k] for k in KEYS if cur and cur[k] is not None} if cur else None)

# ---------------------------------------------------------------- 7-day forecast (rewritten once an hour)
prev_fc = read_js(FC_FILE, "window.FC=")
if prev_fc and prev_fc.get("u") and prev_fc["u"][0] == ukey(now) and "m" in prev_fc:
    print("7-day forecast already written for this hour")
    status.extend(s for s in prev_meta.get("status", []) if s["name"] == "IESO 7-day report files")   # keep the last result
else:
    names = sorted({c for fid in FIDS for c in model_for(fid, "wk").feats})     # every input any model's explorer needs
    fc = F.loc[now:now + pd.DateOffset(hours=167)].dropna(subset=names)
    if len(fc):
        M = {fid: [round(float(p)) for p in model_for(fid, "wk").predict(fc)] for fid in FIDS}
        ieso7 = ieso
        try:                                           # the dated daily report files cover the days ahead
            if time.monotonic() - T0 > 240:
                raise RuntimeError("skipped: this update was already running long")
            adeq7, info7 = fetch_adequacy(actual=dem, days=8, today=now)
            ieso7 = adeq7
            note("IESO 7-day report files", True, info7)
        except Exception as e:
            note("IESO 7-day report files", False, e)
        off = [None if i not in ieso7.index or np.isnan(ieso7[i]) else round(float(ieso7[i])) for i in fc.index]
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
write_log(rows, meta)                                  # again, so statuses recorded while building the forecast are saved
