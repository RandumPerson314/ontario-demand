"""Update the Ontario demand prediction log."""

import json
import os

import joblib
import pandas as pd

from features import (
    FEATURES,
    fetch_demand,
    fetch_live,
    fetch_live_demand,
    make_features,
    predict_demand,
)

LOG = "docs/demand_log"
KEEP = 24 * 30
TORONTO = "America/Toronto"

model = joblib.load("model/model.joblib")

# Current hour in UTC, kept naive because the feature pipeline uses
# a naive UTC index internally.
now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")

# Historical hourly demand for model features and retrospective accuracy.
dem = fetch_demand([now.year - 1, now.year])

# Weather data.
weather = fetch_live()

# Build model features.
F = make_features(weather, dem)

# Current IESO 5-minute demand.
live = fetch_live_demand()

print(
    f"IESO live demand: {live['demand']:.0f} MW "
    f"at {live['timestamp'].strftime('%Y-%m-%d %H:%M %Z')}"
)

# -------------------------------------------------------------------
# Load existing log.
# -------------------------------------------------------------------

log = {}

if os.path.exists(LOG + ".json"):
    with open(LOG + ".json") as fh:
        log = {r["t"]: r for r in json.load(fh)}

# -------------------------------------------------------------------
# Retrospective predictions.
#
# These preserve the existing accuracy calculation. We only create
# predictions for hours that have enough historical data to evaluate.
# -------------------------------------------------------------------

use = (
    F.loc[now - pd.DateOffset(hours=72):now]
    .dropna(subset=FEATURES)
)

for idx, p, temp in zip(
    use.index,
    predict_demand(model, use),
    use["temp"]
):
    # Convert UTC -> Toronto correctly, including DST.
    local = (
        idx.tz_localize("UTC")
        .tz_convert(TORONTO)
    )

    key = local.strftime("%Y-%m-%dT%H:00")

    if key not in log:
        log[key] = dict(
            t=key,
            pred=round(float(p), 1),
            actual=None,
            temp=round(float(temp), 1),
            bf=int(idx < now),
        )

# -------------------------------------------------------------------
# Fill historical actuals from IESO hourly data.
# -------------------------------------------------------------------

for r in log.values():
    local = pd.Timestamp(r["t"], tz=TORONTO)

    # Convert dashboard timestamp back to naive UTC.
    utc = (
        local
        .tz_convert("UTC")
        .tz_localize(None)
    )

    if r["actual"] is None and utc in dem.index:
        r["actual"] = float(dem[utc])

# -------------------------------------------------------------------
# Create the current next-hour prediction.
#
# The model's HORIZON=1 means the prediction at hour X uses the
# previous known hour's demand and weather features for X.
# -------------------------------------------------------------------

live_local = live["timestamp"]

# The target is the next local clock hour.
target_local = live_local.ceil("h")
local = (
    idx.tz_localize("UTC")
    .tz_convert("America/Toronto")
)

target_utc = (
    target_local
    .tz_localize("America/Toronto")
    .tz_convert("UTC")
    .tz_localize(None)
)

if target_utc in F.index:
    target_row = F.loc[[target_utc]].dropna(subset=FEATURES)

    if not target_row.empty:
        prediction = float(
            predict_demand(model, target_row).iloc[0]
        )

        key = local.strftime("%Y-%m-%dT%H:00")

        log[key] = dict(
            t=key,
            pred=round(prediction, 1),
            actual=None,
            temp=round(float(target_row["temp"].iloc[0]), 1),
            bf=0,
        )

        print(
            f"Next-hour prediction: {prediction:.0f} MW "
            f"for {key}"
        )
    else:
        print(
            f"Target hour {target_local} does not yet have "
            f"complete model features."
        )
else:
    print(
        f"Target hour {target_utc} is not available in weather data yet."
    )

# -------------------------------------------------------------------
# Keep the most recent 30 days.
# -------------------------------------------------------------------

rows = sorted(
    log.values(),
    key=lambda r: r["t"]
)[-KEEP:]

os.makedirs("docs", exist_ok=True)

with open(LOG + ".json", "w") as fh:
    json.dump(rows, fh)

with open(LOG + ".js", "w") as fh:
    fh.write("window.LOG=" + json.dumps(rows) + ";")

print(
    f"{len(rows)} rows, "
    f"latest prediction {rows[-1]['pred']:.0f} MW "
    f"at {rows[-1]['t']}"
)