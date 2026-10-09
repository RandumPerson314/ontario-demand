"""Shared data loading + feature engineering (training and live updates use the same code)."""
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
TEMP_DERIVED = ["hdd", "cdd", "temp_lag1", "temp_lag3", "temp_lag24", "temp_roll24",
                "temp_roll72", "dtemp3"]
CALENDAR = ["hour", "dow", "month", "doy", "hol", "hol_before", "hol_after", "xmas", "nonwork"]



def hourly_features(h):
    """Features for a model that knows demand up to h hours before the hour being predicted."""
    lags = [h, h + 1, h + 2, 24, 48, 168]
    return (WEATHER + TEMP_DERIVED + [f"dem_lag{k}" for k in lags]
            + [f"dem_roll24_h{h}", f"dem_trend_h{h}"] + CALENDAR)


NOW_FEATURES = hourly_features(1)    # "this hour": knows demand through the previous hour
NEXT_FEATURES = hourly_features(2)   # "next hour": same latest known hour, one hour further ahead
# "Week" model: only uses demand from >= 7 days earlier, so it can forecast 7 days ahead.
WEEK_FEATURES = (WEATHER + TEMP_DERIVED + ["dtemp168", "dem_lag168", "dem_lag336", "dem_wk_mean"]
                 + CALENDAR)

# Polynomial model: a fixed formula in weather + calendar only (no recent demand).
POLY_CORE = ["temp", "h_s1", "h_c1", "h_s2", "h_c2", "nonwork", "solar"]   # full polynomial terms
POLY_LINEAR = (["feels", "humidity", "dew", "wind", "cloud", "precip", "snow", "hdd", "cdd",
                "temp_lag1", "temp_lag3", "temp_lag24", "temp_roll24", "temp_roll72",
                "doy_s", "doy_c", "year_frac", "hol", "hol_before", "hol_after", "xmas"]
               + [f"dow{k}" for k in range(1, 7)])                           # linear terms
POLY_FEATURES = POLY_CORE + POLY_LINEAR

_HOL = pd.to_datetime(list(holidays.Canada(subdiv="ON", years=range(2018, 2034)).keys()))


def _get(url, params):
    for i in range(8):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429:
                raise requests.exceptions.RetryError("429 rate limited")
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.RetryError) as e:
            wait = 60 * (i + 1)
            print(f"{type(e).__name__}: retrying in {wait}s ({i + 1}/8)")
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
    now, acc = pd.Timestamp.now(), None
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
        acc = h if acc is None else acc.add(h)
    return _finish(acc)


def fetch_live(forecast_days=9):
    """Last 14 days of weather + forecast (UTC index) from the forecast API."""
    C = list(CITIES.values())
    d = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": ",".join(str(c[0]) for c in C), "longitude": ",".join(str(c[1]) for c in C),
        "hourly": ",".join(VARS), "past_days": 14, "forecast_days": forecast_days,
        "timezone": "GMT"})
    d = d if isinstance(d, list) else [d]
    acc = None
    for x, c in zip(d, C):
        h = _hourly(x, c[2])
        acc = h if acc is None else acc.add(h)
    return _finish(acc)


def fetch_demand(years):
    """IESO hourly Ontario demand, indexed by naive UTC hour start.

    IESO reports are in Eastern *Standard* Time all year (no daylight saving),
    so EST = UTC - 5 always. (Treating them as Toronto local time shifts every
    summer hour by one.)"""
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
    # all intervals of the current delivery hour (timestamps are naive EST), plus the newest one
    return {**rows[-1], "rows": rows, "hour_start": base}


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

    # extra columns used only by the polynomial model
    a, b = 2 * np.pi * np.asarray(loc.hour) / 24, 2 * np.pi * np.asarray(loc.dayofyear) / 365.25
    f["h_s1"], f["h_c1"], f["h_s2"], f["h_c2"] = np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)
    f["doy_s"], f["doy_c"] = np.sin(b), np.cos(b)
    f["year_frac"] = np.asarray(loc.year) + (np.asarray(loc.dayofyear) - 1) / 365.25
    for k in range(1, 7):
        f[f"dow{k}"] = (np.asarray(loc.dayofweek) == k).astype(int)
    return f


def predict_hourly(model, X, h):
    """Hourly models predict the change from the demand known h hours earlier."""
    return X[f"dem_lag{h}"] + model.predict(X[hourly_features(h)])


def predict_now(model, X):
    return predict_hourly(model, X, 1)


def predict_next(model, X):
    return predict_hourly(model, X, 2)


def predict_week(model, X):
    return X["dem_lag168"] + model.predict(X[WEEK_FEATURES])
