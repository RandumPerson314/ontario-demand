"""Shared data loading + feature engineering (training and live updates use the same code)."""
import os
import time

import holidays
import pandas as pd
import requests

BASE_TEMP = 18.0
# Hours between the newest demand value we know and the hour being predicted.
# 1 = use the previous hour's demand (most accurate). If IESO publishes too slowly
# for live use, set 2 and retrain.
HORIZON = 1
LAGS = sorted({HORIZON, HORIZON + 1, HORIZON + 2, 24, 48, 168})

CITIES = {  # name: (lat, lon, population weight)
    "toronto": (43.65, -79.38, 0.50), "ottawa": (45.42, -75.70, 0.15),
    "london": (42.98, -81.25, 0.10), "windsor": (42.31, -83.04, 0.08),
    "kitchener": (43.45, -80.49, 0.08), "sudbury": (46.49, -81.00, 0.05),
    "thunder_bay": (48.38, -89.25, 0.04)}
VARS = {"temperature_2m": "temp", "apparent_temperature": "feels",
        "relative_humidity_2m": "humidity", "dew_point_2m": "dew",
        "wind_speed_10m": "wind", "cloud_cover": "cloud",
        "shortwave_radiation": "solar", "precipitation": "precip", "snowfall": "snow"}

FEATURES = (list(VARS.values())
            + ["hdd", "cdd", "temp_lag1", "temp_lag3", "temp_lag24", "temp_roll24",
               "temp_roll72", "dtemp3"]
            + [f"dem_lag{k}" for k in LAGS] + ["dem_roll24", "dem_trend"]
            + ["hour", "dow", "month", "doy", "hol", "hol_before", "hol_after",
               "xmas", "nonwork"])

_HOL = pd.to_datetime(list(holidays.Canada(subdiv="ON", years=range(2018, 2034)).keys()))


def _get(url, params):
    for i in range(6):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        time.sleep(60 * (i + 1))
    raise RuntimeError("Open-Meteo rate limit; try again later")


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
                "latitude": la,
                "longitude": lo,
                "start_date": f"{y}-01-01",
                "end_date": end.strftime("%Y-%m-%d"),
                "hourly": ",".join(VARS),
                "timezone": "GMT",
            })

            df = pd.DataFrame(d["hourly"])

            # Cache immediately after a successful download.
            if y < now.year:
                df.to_csv(p, index=False)

            parts.append(df.assign(time=pd.to_datetime(df["time"])))
            print("fetched", name, y)
            time.sleep(10)
        h = pd.concat(parts).drop_duplicates("time").set_index("time")
        h = h[list(VARS)].interpolate(limit=3) * wt
        acc = h if acc is None else acc.add(h)
    return _finish(acc)


def fetch_live():
    """Last 10 days + today of weather (UTC index) from the forecast/analysis API."""
    C = list(CITIES.values())
    d = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": ",".join(str(c[0]) for c in C), "longitude": ",".join(str(c[1]) for c in C),
        "hourly": ",".join(VARS), "past_days": 10, "forecast_days": 1, "timezone": "GMT"})
    d = d if isinstance(d, list) else [d]
    acc = None
    for x, c in zip(d, C):
        h = _hourly(x, c[2])
        acc = h if acc is None else acc.add(h)
    return _finish(acc)


def fetch_demand(years):
    """IESO hourly Ontario demand, indexed by naive UTC hour."""
    from zoneinfo import ZoneInfo

    out = []
    tz = ZoneInfo("America/Toronto")
    utc = ZoneInfo("UTC")

    for y in years:
        try:
            df = pd.read_csv(
                f"https://reports-public.ieso.ca/public/Demand/PUB_Demand_{y}.csv",
                skiprows=3
            ).dropna(subset=["Ontario Demand"])
        except Exception as e:
            print(f"IESO {y}: {e}")
            continue

        local = pd.to_datetime(
            df["Date"].astype(str)
            + " "
            + (df["Hour"].astype(int) - 1).astype(str)
            + ":00"
        )

        utc_values = []
        seen = {}

        for ts in local:
            key = ts.to_pydatetime()

            # Handle the repeated 01:00 during the fall DST transition.
            fold = min(seen.get(key, 0), 1)
            seen[key] = seen.get(key, 0) + 1

            aware = key.replace(
                tzinfo=tz,
                fold=fold
            )

            utc_values.append(
                aware.astimezone(utc).replace(tzinfo=None)
            )

        out.append(
            pd.Series(
                df["Ontario Demand"].astype(float).values,
                index=pd.DatetimeIndex(utc_values)
            )
        )

    return pd.concat(out).groupby(level=0).last().sort_index()

def fetch_live_demand():
    """Fetch the latest Ontario 5-minute demand from IESO."""
    from xml.etree import ElementTree

    url = (
        "https://reports-public.ieso.ca/public/"
        "RealtimeTotals/PUB_RealtimeTotals.xml"
    )

    r = requests.get(url, timeout=120)
    r.raise_for_status()

    root = ElementTree.fromstring(r.content)

    def find_text(name):
        """Find an XML element by local name, ignoring namespaces."""
        for elem in root.iter():
            if elem.tag.split("}")[-1] == name:
                return elem.text
        return None

    delivery_date = find_text("DeliveryDate")
    delivery_hour = int(find_text("DeliveryHour"))

    # IESO hour 1 = 00:00-01:00, hour 2 = 01:00-02:00, etc.
    base = (
        pd.Timestamp(delivery_date)
        + pd.Timedelta(hours=delivery_hour - 1)
    )

    rows = []

    # Find every 5-minute interval.
    for interval_energy in root.iter():
        if interval_energy.tag.split("}")[-1] != "IntervalEnergy":
            continue

        interval = None
        ontario_demand = None

        for child in interval_energy.iter():
            name = child.tag.split("}")[-1]

            if name == "Interval":
                interval = int(child.text)

            elif name == "MarketQuantity":
                quantity_name = child.text

                if quantity_name == "ONTARIO DEMAND":
                    # EnergyMW is inside the same MQ element.
                    for sibling in interval_energy.iter():
                        if sibling.tag.split("}")[-1] == "MQ":
                            # handled below
                            pass

            elif name == "EnergyMW":
                # We'll associate this with the MQ by parsing below.
                pass

        # Parse MQ records directly.
        for mq in interval_energy.iter():
            if mq.tag.split("}")[-1] != "MQ":
                continue

            quantity_name = None
            energy_mw = None

            for child in mq:
                name = child.tag.split("}")[-1]

                if name == "MarketQuantity":
                    quantity_name = child.text
                elif name == "EnergyMW":
                    energy_mw = child.text

            if quantity_name == "ONTARIO DEMAND":
                ontario_demand = float(energy_mw)
                break

        if interval is not None and ontario_demand is not None:
            timestamp = base + pd.Timedelta(
                minutes=(interval - 1) * 5
            )

            rows.append({
                "timestamp": timestamp,
                "demand": ontario_demand,
            })

    if not rows:
        raise RuntimeError("No Ontario demand intervals found in IESO report")

    latest = rows[-1]

    return latest


def make_features(w, dem):
    f = w.copy()
    f["hdd"] = (BASE_TEMP - f["temp"]).clip(lower=0)
    f["cdd"] = (f["temp"] - BASE_TEMP).clip(lower=0)
    for k in (1, 3, 24):
        f[f"temp_lag{k}"] = f["temp"].shift(k)
    f["temp_roll24"] = f["temp"].rolling(24).mean()
    f["temp_roll72"] = f["temp"].rolling(72).mean()
    f["dtemp3"] = f["temp"] - f["temp_lag3"]

    d = dem.reindex(f.index)                      # past demand, aligned to the weather clock
    for k in LAGS:
        f[f"dem_lag{k}"] = d.shift(k)
    f["dem_roll24"] = d.shift(HORIZON).rolling(24).mean()
    f["dem_trend"] = f[f"dem_lag{HORIZON}"] - f[f"dem_lag{HORIZON + 1}"]
    f["demand"] = d

    loc = f.index.tz_localize("UTC").tz_convert("America/Toronto")
    f["hour"], f["dow"] = loc.hour, loc.dayofweek
    f["month"], f["doy"] = loc.month, loc.dayofyear
    day = pd.DatetimeIndex(loc.tz_localize(None).normalize())
    f["hol"] = day.isin(_HOL).astype(int)
    f["hol_before"] = (day + pd.DateOffset(days=1)).isin(_HOL).astype(int)
    f["hol_after"] = (day - pd.DateOffset(days=1)).isin(_HOL).astype(int)
    f["xmas"] = (((loc.month == 12) & (loc.day >= 24)) | ((loc.month == 1) & (loc.day <= 2))).astype(int)
    f["nonwork"] = ((loc.dayofweek >= 5) | (f["hol"].values == 1)).astype(int)
    return f


def predict_demand(model, X):
    """The model predicts the change from the newest known demand value."""
    return X[f"dem_lag{HORIZON}"] + model.predict(X[FEATURES])
