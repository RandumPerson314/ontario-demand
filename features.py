"""Shared data loading + feature engineering (training and live updates use the same code)."""
import json
import os
import time
from xml.etree import ElementTree

import holidays
import numpy as np
import pandas as pd
import requests

BASE_TEMP = 18.0
TORONTO = "America/Toronto"

CITIES = {  # name: (lat, lon, population weight)
    "toronto": (43.65, -79.38, 0.50), "ottawa": (45.42, -75.70, 0.15),
    "london": (42.98, -81.25, 0.10), "windsor": (42.31, -83.04, 0.08),
    "kitchener": (43.45, -80.49, 0.08), "sudbury": (46.49, -81.00, 0.05),
    "thunder_bay": (48.38, -89.25, 0.04)}
VARS = {"temperature_2m": "temp", "apparent_temperature": "feels",
        "relative_humidity_2m": "humidity", "dew_point_2m": "dew",
        "wind_speed_10m": "wind", "cloud_cover": "cloud",
        "shortwave_radiation": "solar", "precipitation": "precip", "snowfall": "snow"}

WEATHER = list(VARS.values())
CITY_COLS = [f"ct_{c}" for c in CITIES]                         # each city's own temperature (not population-weighted)
HOUR_ONEHOT = [f"hr_{h}" for h in range(24)]                    # one-hot calendar columns, used by the neural networks
MONTH_ONEHOT = [f"mo_{m}" for m in range(1, 13)]
TEMP_DERIVED = ["hdd", "cdd", "temp_lag1", "temp_lag3", "temp_lag24", "temp_roll24",
                "temp_roll72", "dtemp3"]
CALENDAR = ["hour", "dow", "month", "doy", "hol", "hol_before", "hol_after", "xmas", "nonwork"]

# ------------------------------------------------------------------ model registry
# Demand-aware models come in three variants, by how much recent demand they can know:
#   h1 = "this hour"  (knows demand through the previous hour)
#   h2 = "next hour"  (knows demand through the hour before that)
#   wk = "7-day"      (knows demand only from a week or more earlier)
# Weather-only models have a single variant, "all".
DEMAND_COLS = {
    "h1": [f"dem_lag{k}" for k in (1, 2, 3, 24, 48, 168)] + ["dem_roll24_h1", "dem_trend_h1"],
    "h2": [f"dem_lag{k}" for k in (2, 3, 4, 24, 48, 168)] + ["dem_roll24_h2", "dem_trend_h2"],
    "wk": ["dtemp168", "dem_lag168", "dem_lag336", "dem_wk_mean"],
}
BASE_LAG = {"h1": "dem_lag1", "h2": "dem_lag2", "wk": "dem_lag168"}   # demand each variant adds its change to

FAMILIES = {
    "gbm":   dict(kind="gbm",  demand=True,  label="Gradient boosting + recent demand"),
    "gbm0":  dict(kind="gbm",  demand=False, label="Gradient boosting, weather only"),
    "poly":  dict(kind="poly", demand=True,  label="Polynomial + recent demand"),
    "poly0": dict(kind="poly", demand=False, label="Polynomial, weather only"),
    "nn":    dict(kind="nn",   demand=True,  label="Neural network + recent demand"),
    "nn0":   dict(kind="nn",   demand=False, label="Neural network, weather only"),
}

POLY_CORE = ["temp", "h_s1", "h_c1", "h_s2", "h_c2", "nonwork", "solar"]   # full polynomial terms
POLY_LINEAR = (["feels", "humidity", "dew", "wind", "cloud", "precip", "snow", "hdd", "cdd",
                "temp_lag1", "temp_lag3", "temp_lag24", "temp_roll24", "temp_roll72",
                "doy_s", "doy_c", "year_frac", "hol", "hol_before", "hol_after", "xmas"]
               + [f"dow{k}" for k in range(1, 7)])                           # linear terms
POLY_FEATURES = POLY_CORE + POLY_LINEAR
NN_BASE = (WEATHER + TEMP_DERIVED + ["h_s1", "h_c1", "h_s2", "h_c2", "doy_s", "doy_c", "year_frac",
                                     "hol", "hol_before", "hol_after", "xmas", "nonwork"]
           + [f"dow{k}" for k in range(1, 7)])
GBM0_FEATURES = WEATHER + TEMP_DERIVED + ["year_frac"] + CALENDAR


def variants(fid):
    return ["h1", "h2", "wk"] if FAMILIES[fid]["demand"] else ["all"]


def feature_list(fid, v, city=False):
    """Inputs for a model. city=True adds every city's own temperature as separate inputs."""
    fam = FAMILIES[fid]
    dem = DEMAND_COLS[v] if fam["demand"] else []
    if fam["kind"] == "gbm":
        base = WEATHER + TEMP_DERIVED + DEMAND_COLS[v] + CALENDAR if fam["demand"] else GBM0_FEATURES
    else:
        base = (POLY_FEATURES if fam["kind"] == "poly" else NN_BASE + HOUR_ONEHOT + MONTH_ONEHOT) + dem
    return base + (CITY_COLS if city else [])


def base_col(fid, v):
    return BASE_LAG[v] if FAMILIES[fid]["demand"] else None


_HOL = pd.to_datetime(list(holidays.Canada(subdiv="ON", years=range(2018, 2034)).keys()))


# ------------------------------------------------------------------ weather
def _get(url, params, tries=8, base_wait=60):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                raise requests.exceptions.RetryError("429 rate limited")
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.RetryError) as e:
            wait = base_wait * (i + 1)
            print(f"{type(e).__name__}: retrying in {wait}s ({i + 1}/{tries})")
            time.sleep(wait)
    raise RuntimeError("Open-Meteo kept failing; wait an hour and rerun (cached data is kept)")


def _hourly(d, wt):
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h.set_index("time")[list(VARS)].interpolate(limit=3) * wt


def _finish(acc):
    return (acc / sum(c[2] for c in CITIES.values())).rename(columns=VARS)


def fetch_archive(years, cache="cache"):
    """Historical weather (UTC index), cached per city-year. Complete years only are cached."""
    os.makedirs(cache, exist_ok=True)
    now, acc, ct = pd.Timestamp.now(), None, {}
    for name, (la, lo, wt) in CITIES.items():
        parts = []
        for y in years:
            p = f"{cache}/{name}_{y}.csv"
            if os.path.exists(p):
                parts.append(pd.read_csv(p, parse_dates=["time"]))
                continue
            end = min(pd.Timestamp(f"{y}-12-31"), now - pd.DateOffset(days=6))
            d = _get("https://archive-api.open-meteo.com/v1/archive", {
                "latitude": la, "longitude": lo, "start_date": f"{y}-01-01",
                "end_date": end.strftime("%Y-%m-%d"), "hourly": ",".join(VARS),
                "timezone": "GMT"})
            df = pd.DataFrame(d["hourly"])
            if y < now.year:
                df.to_csv(p, index=False)
            parts.append(df.assign(time=pd.to_datetime(df["time"])))
            print("fetched", name, y)
            time.sleep(10)
        h = pd.concat(parts).drop_duplicates("time").set_index("time")
        h = h[list(VARS)].interpolate(limit=3) * wt
        ct[f"ct_{name}"] = h["temperature_2m"] / wt
        acc = h if acc is None else acc.add(h)
    return pd.concat([_finish(acc), pd.DataFrame(ct)], axis=1)


LIVE_CACHE = "data/weather_live.csv"
LIVE_STATUS = {"ok": True, "detail": "fresh"}


def fetch_live(forecast_days=9):
    """Last 14 days of weather + forecast (UTC index). Falls back to the last good copy if
    Open-Meteo is unavailable, so one bad request doesn't stall the whole website."""
    C = list(CITIES.values())
    try:
        d = _get("https://api.open-meteo.com/v1/forecast", {
            "latitude": ",".join(str(c[0]) for c in C), "longitude": ",".join(str(c[1]) for c in C),
            "hourly": ",".join(VARS), "past_days": 14, "forecast_days": forecast_days,
            "timezone": "GMT"}, tries=3, base_wait=15)
        d = d if isinstance(d, list) else [d]
        acc, ct = None, {}
        for x, c, name in zip(d, C, CITIES):
            h = _hourly(x, c[2])
            ct[f"ct_{name}"] = h["temperature_2m"] / c[2]
            acc = h if acc is None else acc.add(h)
        w = pd.concat([_finish(acc), pd.DataFrame(ct)], axis=1)
        os.makedirs("data", exist_ok=True)
        w.to_csv(LIVE_CACHE)
        LIVE_STATUS.update(ok=True, detail="fresh")
        return w
    except Exception as e:
        if not os.path.exists(LIVE_CACHE):
            raise
        LIVE_STATUS.update(ok=False, detail=f"using an older saved copy ({str(e)[:80]})")
        return pd.read_csv(LIVE_CACHE, index_col=0, parse_dates=True)


# ------------------------------------------------------------------ IESO demand
def fetch_demand(years):
    """IESO hourly Ontario demand, indexed by naive UTC hour start.

    IESO reports are in Eastern *Standard* Time all year (no daylight saving),
    so EST = UTC - 5 always. NOTE: IESO updates these yearly files only about once a day,
    so they lag by up to ~30 hours; recent hours come from fetch_multiday()."""
    out = []
    for y in years:
        try:
            df = pd.read_csv(f"https://reports-public.ieso.ca/public/Demand/PUB_Demand_{y}.csv",
                             skiprows=3).dropna(subset=["Ontario Demand"])
        except Exception as e:
            print(f"IESO {y}: {e}")
            continue
        est = pd.to_datetime(df["Date"].astype(str)) + pd.to_timedelta(df["Hour"].astype(int) - 1, unit="h")
        out.append(pd.Series(df["Ontario Demand"].astype(float).values,
                             index=pd.DatetimeIndex(est + pd.DateOffset(hours=5))))
    if not out:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return pd.concat(out).groupby(level=0).last().sort_index()


def fetch_live_demand():
    """Ontario 5-minute demand for the current IESO hour (timestamps are naive EST).
    Returns the newest reading plus all readings of the hour in 'rows'."""
    r = requests.get("https://reports-public.ieso.ca/public/RealtimeTotals/PUB_RealtimeTotals.xml",
                     timeout=60)
    r.raise_for_status()
    root = ElementTree.fromstring(r.content)
    tag = lambda e: e.tag.split("}")[-1]

    def first(name):
        return next(e.text for e in root.iter() if tag(e) == name)

    base = pd.Timestamp(first("DeliveryDate")) + pd.DateOffset(hours=int(first("DeliveryHour")) - 1)
    rows = []
    for ie in root.iter():
        if tag(ie) != "IntervalEnergy":
            continue
        interval = next((int(c.text) for c in ie.iter() if tag(c) == "Interval"), None)
        mw = None
        for mq in ie.iter():
            if tag(mq) == "MQ":
                kids = {tag(k): k.text for k in mq}
                if kids.get("MarketQuantity") == "ONTARIO DEMAND":
                    mw = float(kids["EnergyMW"])
        if interval is not None and mw is not None:
            rows.append({"timestamp": base + pd.DateOffset(minutes=(interval - 1) * 5), "demand": mw})
    if not rows:
        raise RuntimeError("No Ontario demand intervals found in IESO report")
    return {**rows[-1], "rows": rows, "hour_start": base}


MULTIDAY_URL = "https://www.ieso.ca/-/media/Files/IESO/Power-Data/Ontario-Demand-multiday.ashx"
CAL_FILE = "data/multiday_cal.json"


def fetch_multiday(csv):
    """IESO's own hourly chart feed: recent hourly *actual* demand (fresh, unlike the yearly
    file) and IESO's *projected* (official forecast) demand. Its clock convention is not
    documented, so we compare it with the official hourly file to work out the alignment."""
    r = requests.get(MULTIDAY_URL, timeout=60)
    r.raise_for_status()
    root = ElementTree.fromstring(r.content)
    tag = lambda e: e.tag.split("}")[-1]
    start = pd.Timestamp(next(e.text for e in root.iter() if tag(e) == "StartDate"))
    series = {}
    for ds in root.iter():
        if tag(ds) == "DataSet":
            series[ds.get("Series")] = np.array([float(v.text) if v.text and v.text.strip() else np.nan
                                                 for v in ds.iter() if tag(v) == "Value"])
    pick = lambda *keys: next((v for k, v in series.items() if k and any(x in k.lower() for x in keys)), None)
    act, proj = pick("actual"), pick("project", "forecast")
    if act is None:
        raise RuntimeError(f"feed has no 'Actual' series (found: {list(series)})")

    def build(vals, name, shift):                      # every series gets its own length and clock
        t = pd.DatetimeIndex(start + pd.to_timedelta(np.arange(len(vals)), unit="h"))
        if name == "est":
            base = t + pd.DateOffset(hours=5)
        else:
            base = (t.tz_localize(TORONTO, ambiguous="NaT", nonexistent="shift_forward")
                     .tz_convert("UTC").tz_localize(None))
        s = pd.Series(vals, index=base + pd.DateOffset(hours=shift))
        return s[s.index.notna()]

    best = None
    for name in ("est", "local"):
        for shift in (-1, 0, 1):
            a = build(act, name, shift).dropna()
            common = a.index.intersection(csv.index)
            if len(common) >= 12:
                err = float((a.loc[common] - csv.loc[common]).abs().mean())
                if best is None or err < best[0]:
                    best = (err, name, shift, len(common))
    if best is None:                                   # no overlap with the daily file: reuse last calibration
        if not os.path.exists(CAL_FILE):
            raise RuntimeError("cannot calibrate the hourly feed (no overlap with the official hourly file)")
        c = json.load(open(CAL_FILE))
        best = (np.nan, c["name"], c["shift"], 0)
    elif best[0] > 700:
        raise RuntimeError(f"hourly feed disagrees with the official file (avg {best[0]:.0f} MW)")
    else:
        os.makedirs("data", exist_ok=True)
        json.dump({"name": best[1], "shift": best[2]}, open(CAL_FILE, "w"))
    err, name, shift, n = best
    return dict(actual=build(act, name, shift).dropna(),
                projected=build(proj, name, shift).dropna() if proj is not None else pd.Series(dtype=float),
                info=f"clock={name}, shift={shift}h, matched {n} h to the official file (avg diff {err:.0f} MW)")


ADEQUACY_URLS = ["https://reports-public.ieso.ca/public/Adequacy3/PUB_Adequacy3.xml",
                 "https://reports-public.ieso.ca/public/Adequacy2/PUB_Adequacy2.xml"]


def _parse_adequacy(content):
    root = ElementTree.fromstring(content)
    tag = lambda e: e.tag.split("}")[-1]
    date, system, groups = None, "", {}
    for el in root.iter():
        n = tag(el)
        if el.get("DeliveryDate"):
            date = el.get("DeliveryDate")[:10]
        if n == "DeliveryDate" and el.text and el.text.strip():
            date = el.text.strip()[:10]
        elif n in ("SystemName", "ZoneName", "AreaName", "Zone", "Area") and el.text and el.text.strip():
            system = el.text.strip()
        else:
            kids = {tag(k): (k.text or "").strip() for k in el}
            h = el.get("DeliveryHour") or el.get("Hour") or kids.get("DeliveryHour") or kids.get("Hour")
            mw = next((kids[k] for k in ("EnergyMW", "MW", "Value", "Energy") if kids.get(k)), None)
            if h and mw and date:
                try:
                    groups.setdefault((system, n), []).append((date, int(h), float(mw)))
                except ValueError:
                    pass

    def series(key):
        vals = {}
        for d, h, mw in groups[key]:
            vals.setdefault(pd.Timestamp(d) + pd.DateOffset(hours=h - 1 + 5), mw)   # hour-ending EST -> hour start UTC
        return pd.Series(vals).sort_index()

    def tag_rank(n):
        n = n.lower()
        if "demand" not in n or any(x in n for x in ("peak", "market", "uncertain", "reserve", "capacity")):
            return None
        return 0 if "average" in n else 1 if "forecast" in n else 2

    # 1) an Ontario-wide series
    whole = [k for k in groups if tag_rank(k[1]) is not None and (not k[0] or "ontario" in k[0].lower() or "total" in k[0].lower())]
    if whole:
        key = min(whole, key=lambda k: (tag_rank(k[1]), -len(groups[k])))
        s = series(key)
        return s, f"used '{key[1]}' for {key[0] or 'Ontario'} ({len(s)} hours, {s.index[0]:%m-%d} to {s.index[-1]:%m-%d})"
    # 2) the four forecast zones: add them up
    zone_tags = {}
    for k in groups:
        if tag_rank(k[1]) is not None and k[0]:
            zone_tags.setdefault(k[1], []).append(k)
    for n, keys in sorted(zone_tags.items(), key=lambda kv: (tag_rank(kv[0]), -len(kv[1]))):
        if len(keys) >= 3:
            parts = pd.concat([series(k) for k in keys], axis=1, join="inner")
            if len(parts):
                s = parts.sum(axis=1)
                return s, f"added {len(keys)} zones of '{n}' ({len(s)} hours, {s.index[0]:%m-%d} to {s.index[-1]:%m-%d})"
    seen = ", ".join(f"{k[0] or '-'}/{k[1]}({len(v)})" for k, v in list(groups.items())[:10])
    raise RuntimeError(f"no demand forecast recognised in the report; hourly groups seen: {seen or 'none'}")


def fetch_adequacy():
    """IESO's Adequacy Report: hourly demand forecast for today and up to ~34 days ahead
    (published per forecast zone; the zones are added up). Returns (Series by naive UTC hour start, note)."""
    errors = []
    for url in ADEQUACY_URLS:
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            s, note = _parse_adequacy(r.content)
            return s, f"{url.split('/')[-1]}: {note}"
        except Exception as e:
            errors.append(f"{url.split('/')[-1]}: {str(e)[:150]}")
    raise RuntimeError(" | ".join(errors))


def merge_demand(csv, extra, now):
    """Official hourly file where it exists, plus fresher completed hours from the live feed."""
    if extra is None or extra.empty:
        return csv
    e = extra[(extra.index < now) & ~extra.index.isin(csv.index)]
    return pd.concat([csv, e]).sort_index()


# ------------------------------------------------------------------ features
def make_features(w, dem):
    f = w.copy()
    f["hdd"] = (BASE_TEMP - f["temp"]).clip(lower=0)
    f["cdd"] = (f["temp"] - BASE_TEMP).clip(lower=0)
    for k in (1, 3, 24):
        f[f"temp_lag{k}"] = f["temp"].shift(k)
    f["temp_roll24"] = f["temp"].rolling(24).mean()
    f["temp_roll72"] = f["temp"].rolling(72).mean()
    f["dtemp3"] = f["temp"] - f["temp_lag3"]
    f["dtemp168"] = f["temp"] - f["temp"].shift(168)

    # Demand lags are computed on the full demand history (not just the weather window),
    # so 1- and 2-week lags exist for every forecast hour.
    d = dem.reindex(pd.date_range(min(dem.index.min(), f.index.min()), f.index.max(), freq="h"))
    for k in (1, 2, 3, 4, 24, 48, 168, 336):
        f[f"dem_lag{k}"] = d.shift(k).reindex(f.index)
    for h in (1, 2):
        f[f"dem_roll24_h{h}"] = d.shift(h).rolling(24).mean().reindex(f.index)
        f[f"dem_trend_h{h}"] = f[f"dem_lag{h}"] - f[f"dem_lag{h + 1}"]
    f["dem_wk_mean"] = d.shift(168).rolling(168).mean().reindex(f.index)
    f["demand"] = d.reindex(f.index)

    loc = f.index.tz_localize("UTC").tz_convert(TORONTO)
    f["hour"], f["dow"] = loc.hour, loc.dayofweek
    f["month"], f["doy"] = loc.month, loc.dayofyear
    day = pd.DatetimeIndex(loc.tz_localize(None).normalize())
    f["hol"] = day.isin(_HOL).astype(int)
    f["hol_before"] = (day + pd.DateOffset(days=1)).isin(_HOL).astype(int)
    f["hol_after"] = (day - pd.DateOffset(days=1)).isin(_HOL).astype(int)
    f["xmas"] = (((loc.month == 12) & (loc.day >= 24)) | ((loc.month == 1) & (loc.day <= 2))).astype(int)
    f["nonwork"] = ((loc.dayofweek >= 5) | (f["hol"].values == 1)).astype(int)

    a, b = 2 * np.pi * np.asarray(loc.hour) / 24, 2 * np.pi * np.asarray(loc.dayofyear) / 365.25
    f["h_s1"], f["h_c1"], f["h_s2"], f["h_c2"] = np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)
    f["doy_s"], f["doy_c"] = np.sin(b), np.cos(b)
    f["year_frac"] = np.asarray(loc.year) + (np.asarray(loc.dayofyear) - 1) / 365.25
    for k in range(1, 7):
        f[f"dow{k}"] = (np.asarray(loc.dayofweek) == k).astype(int)
    for h in range(24):
        f[f"hr_{h}"] = (np.asarray(loc.hour) == h).astype(int)
    for m in range(1, 13):
        f[f"mo_{m}"] = (np.asarray(loc.month) == m).astype(int)
    return f
