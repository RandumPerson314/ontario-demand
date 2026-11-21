"""One-off reconstruction, from this repository's git history and the IESO's archived report versions, of
(1) when each saved IESO forecast was first published and what every model was predicting at that moment, and
(2) a seed for the whole-project score archive (every hour the site ever logged).

Run from the repository root with full git history (git fetch --unshallow):   python tools/backfill_ieso_times.py
Writes data/backfill_ieso.json and data/archive_seed.json; the next update run merges both in.

How it decides, for an hour whose IESO forecast the site saved:
* capture time  = the first commit of docs/demand_log.js in which that hour's ieso_h appears;
* IESO version  = the newest PUB_Adequacy3_<day>_vN.xml published before the capture whose value matches;
* published at  = the publication time of the earliest version in the unbroken run of versions with that same
                  value, read from the IESO directory listing (EST, to the minute);
* each model's prediction then = its newest prediction made at or before that time: the hourly 7-day forecast
  runs (docs/forecast.js history, timed by their 'updated' field), and the next-hour / this-hour predictions
  (timed by the commit that first contained them, which is slightly after they were made, so never early)."""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import ADEQUACY_BASE, FAMILIES, _adequacy_groups  # noqa: E402

FIDS = list(FAMILIES)
PATH = "DocBody/ForecastDemand/OntarioDemand/ForecastOntDemand/Demand"
REF = sys.argv[1] if len(sys.argv) > 1 else "origin/main"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def commits(path):
    out = git("log", REF, "--reverse", "--format=%H %ct", "--", path).split()
    return [(out[i], pd.Timestamp(int(out[i + 1]), unit="s")) for i in range(0, len(out), 2)]


def read_js(sha, path, prefix):
    try:
        txt = git("show", f"{sha}:{path}")
    except subprocess.CalledProcessError:
        return None
    return json.JSONDecoder().raw_decode(txt[txt.index(prefix) + len(prefix):])[0]


# ---------------------------------------------------------------- 1. log history: first time each value appeared
print("reading docs/demand_log.js history ...")
first_seen, latest = {}, {}                 # (u, key) -> commit time ; u -> newest version of the row
for sha, t in commits("docs/demand_log.js"):
    rows = read_js(sha, "docs/demand_log.js", "window.LOG=") or []
    for r in rows:
        if "u" not in r:
            continue
        for old, new in (("pred", "gbm_h"), ("pn", "gbm_n"), ("pp", "poly0_n")):
            if old in r:
                r.setdefault(new, r.pop(old))
        latest[r["u"]] = r
        for k, v in r.items():
            if v is not None and (k.endswith("_h") or k.endswith("_n")):
                first_seen.setdefault((r["u"], k), t)
print(f"  {len(latest)} hours ever logged")

# ---------------------------------------------------------------- 2. forecast.js history: the hourly 7-day runs
print("reading docs/forecast.js history ...")
runs = []                                   # (made, {u: {fid: value}})
for sha, t in commits("docs/forecast.js"):
    fc = read_js(sha, "docs/forecast.js", "window.FC=")
    if not fc or "m" not in fc or not fc.get("updated"):
        continue
    made = pd.Timestamp(fc["updated"]).tz_convert("UTC").tz_localize(None)
    runs.append((made, {u: {f: fc["m"][f][i] for f in FIDS if fc["m"].get(f)} for i, u in enumerate(fc["u"])}))
runs.sort(key=lambda x: x[0])
print(f"  {len(runs)} forecast runs")

# ---------------------------------------------------------------- 3. IESO versions
listing = requests.get(ADEQUACY_BASE, timeout=60).text
versions = {}                               # day -> [(n, published UTC)]
for d, n, ts in re.findall(r"PUB_Adequacy3_(\d{8})_v(\d+)\.xml</a>\s+(\S+ \S+)", listing):
    versions.setdefault(d, []).append((int(n), pd.Timestamp(pd.to_datetime(ts, format="%d-%b-%Y %H:%M")) + pd.DateOffset(hours=5)))
for d in versions:
    versions[d].sort()
_cache = {}


def values(day, n):
    for attempt in range(5):
        if (day, n) in _cache:
            break
        try:
            r = requests.get(f"{ADEQUACY_BASE}PUB_Adequacy3_{day}_v{n}.xml", timeout=30)
            r.raise_for_status()
            _cache[(day, n)] = {h: float(mw) for d, h, mw in _adequacy_groups(r.content).get(PATH, [])}
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    return _cache[(day, n)]


todo = [u for u, r in latest.items() if r.get("ieso_h") is not None]
print(f"{len(todo)} hours with a saved IESO forecast; prefetching report versions ...")
need = set()
for u in todo:
    ts = pd.Timestamp(u) - pd.DateOffset(hours=5)
    day, cap = ts.strftime("%Y%m%d"), first_seen.get((u, "ieso_h"))
    for n, pub in versions.get(day, []):
        if cap is not None and cap - pd.DateOffset(days=3) <= pub <= cap:
            need.add((day, n))
with ThreadPoolExecutor(4) as pool:
    list(pool.map(lambda x: values(*x), sorted(need)))
print(f"  {len(_cache)} versions read")

# ---------------------------------------------------------------- 4. reconstruct each hour
out, problems = {}, []
for u in sorted(todo):
    r, start = latest[u], pd.Timestamp(u)
    est = start - pd.DateOffset(hours=5)
    day, he, cap, v = est.strftime("%Y%m%d"), est.hour + 1, first_seen.get((u, "ieso_h")), r["ieso_h"]
    vs = [(n, p) for n, p in versions.get(day, []) if cap is not None and p <= cap]
    i = len(vs) - 1
    while i >= 0 and len(vs) - 1 - i < 4 and values(day, vs[i][0]).get(he) != v:
        i -= 1                               # the matching version (a very new one may not have been read yet)
    if i < 0 or values(day, vs[i][0]).get(he) != v:
        problems.append(u)
        continue
    exact = False
    while i > 0:
        try:
            prev = values(day, vs[i - 1][0]).get(he)
        except Exception:
            break
        if prev != v:
            exact = True
            break
        i -= 1
    n, T = vs[i]
    rel = lambda t: int((t - start).total_seconds() // 60)
    rec = dict(ip=rel(T), ipf=0 if exact else 1, iv=f"PUB_Adequacy3_{day}_v{n}.xml", mi={}, mim={}, mis={})
    late = False
    for fid in FIDS:
        cands = [(made, run[u][fid], "7d") for made, run in runs if u in run and run[u].get(fid) is not None]
        for sfx, src in (("_n", "n"), ("_h", "h")):
            t = first_seen.get((u, fid + sfx))
            if t is not None and r.get(fid + sfx) is not None:
                cands.append((t, r[fid + sfx], src))
        cands.sort(key=lambda c: c[0])
        before = [c for c in cands if c[0] <= T]
        if not cands:
            continue
        made, val, src = before[-1] if before else cands[0]
        late |= not before
        rec["mi"][fid], rec["mim"][fid], rec["mis"][fid] = round(float(val), 1), rel(made), src
    if late:
        rec["mil"] = 1
    out[u] = rec
os.makedirs("data", exist_ok=True)
json.dump(out, open("data/backfill_ieso.json", "w"), separators=(",", ":"))
print(f"data/backfill_ieso.json: {len(out)} hours" + (f"; no matching IESO version for {len(problems)}: {problems[:5]}" if problems else ""))

# ---------------------------------------------------------------- 5. archive seed: every hour ever logged
seed = []
for u in sorted(latest):
    r = latest[u]
    if r.get("actual") is None:
        continue
    a = dict(u=u, a=r["actual"], bf=r.get("bf", 1), h=[r.get(f + "_h") for f in FIDS],
             n=[r.get(f + "_n") for f in FIDS], i=r.get("ieso_h"))
    b = out.get(u)
    if b:
        a.update(ip=b["ip"], ipf=b["ipf"], m=[b["mi"].get(f) for f in FIDS], mm=[b["mim"].get(f) for f in FIDS])
        if b.get("mil"):
            a["ml"] = 1
    seed.append(a)
json.dump({"ids": FIDS, "rows": seed}, open("data/archive_seed.json", "w"), separators=(",", ":"))
print(f"data/archive_seed.json: {len(seed)} hours, {seed[0]['u']} to {seed[-1]['u']}")
