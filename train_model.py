"""Train both models locally:  python train_model.py
Creates model/now.joblib, model/week.joblib and docs/week_model.js (for the website's explorer)."""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae

from features import (NOW_FEATURES, WEEK_FEATURES, fetch_archive, fetch_demand,
                      make_features)

YEARS = range(2019, 2027)          # last year may be partial; that's fine
NOW_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
                  l2_regularization=1.0, random_state=0, early_stopping=False)
WEEK_PARAMS = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
                   l2_regularization=2.0, random_state=0, early_stopping=False)


def fit_eval(F, name, feats, lag, params, cap, baselines):
    sub = F.dropna(subset=feats + ["demand"])
    y = sub["demand"] - sub[lag]                       # predict change from a known past value
    train, val, test = sub[sub.index.year < 2024], sub[sub.index.year == 2024], sub[sub.index.year >= 2025]
    print(f"\n=== {name}: rows train {len(train)}, val {len(val)}, test {len(test)}")

    m = HistGradientBoostingRegressor(max_iter=cap, **params).fit(train[feats], y[train.index])
    curve = [mae(val["demand"], val[lag] + p) for p in m.staged_predict(val[feats])]
    best = int(np.argmin(curve)) + 1                   # number of trees chosen on the 2024 validation year
    print("best number of trees:", best)

    trv = sub[sub.index.year < 2025]                   # honest score on the untouched final period
    m2 = HistGradientBoostingRegressor(max_iter=best, **params).fit(trv[feats], y[trv.index])
    pred = test[lag] + m2.predict(test[feats])
    print(f"TEST model MAE {mae(test['demand'], pred):.0f} MW "
          f"(MAPE {np.mean(np.abs(pred - test['demand']) / test['demand']) * 100:.2f}%)")
    for label, p in baselines(test).items():
        print(f"     baseline '{label}': MAE {mae(test['demand'], p):.0f} MW")

    final = HistGradientBoostingRegressor(max_iter=best, **params).fit(sub[feats], y)
    return final, sub


def export_trees(model, feats, X_check, path):
    """Write the trees as JavaScript so the page can evaluate the model in the browser."""
    base = float(np.ravel(model._baseline_prediction)[0])
    g = float                                         # full precision: rounding flips splits
    trees = []
    for it in model._predictors:
        nd = it[0].nodes
        trees.append(dict(
            f=np.where(nd["is_leaf"] == 1, -1, nd["feature_idx"]).astype(int).tolist(),
            t=[g(v) for v in nd["num_threshold"]],
            l=nd["left"].astype(int).tolist(), r=nd["right"].astype(int).tolist(),
            v=[round(float(v), 3) for v in nd["value"]]))

    def py_predict(x):                                 # same algorithm as the JavaScript
        s = base
        for T in trees:
            n = 0
            while T["f"][n] >= 0:
                n = T["l"][n] if x[T["f"][n]] <= T["t"][n] else T["r"][n]
            s += T["v"][n]
        return s

    Xc = X_check[feats].iloc[::max(1, len(X_check) // 100)].head(100)
    diff = np.abs(np.array([py_predict(r) for r in Xc.values]) - model.predict(Xc)).max()
    print(f"tree export self-check: max difference {diff:.3f} MW")
    if diff > 2:
        print("WARNING: exported trees do not match the model; the explorer would be wrong.")
    with open(path, "w") as fh:
        fh.write("window.WM=" + json.dumps(dict(features=feats, base=base, trees=trees),
                                            separators=(",", ":")) + ";")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    w, dem = fetch_archive(YEARS), fetch_demand(YEARS)
    F = make_features(w, dem)
    os.makedirs("model", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    now_m, _ = fit_eval(F, "NOW model (uses last hour's demand)", NOW_FEATURES, "dem_lag1",
                        NOW_PARAMS, 1500, lambda t: {"last hour": t["dem_lag1"],
                                                     "same hour yesterday": t["dem_lag24"],
                                                     "same hour last week": t["dem_lag168"]})
    joblib.dump(now_m, "model/now.joblib", compress=3)

    week_m, sub = fit_eval(F, "WEEK model (7-day forecast)", WEEK_FEATURES, "dem_lag168",
                           WEEK_PARAMS, 700, lambda t: {"same hour last week": t["dem_lag168"],
                                                        "avg of last 2 weeks": (t["dem_lag168"] + t["dem_lag336"]) / 2})
    joblib.dump(week_m, "model/week.joblib", compress=3)
    export_trees(week_m, WEEK_FEATURES, sub, "docs/week_model.js")
    print("\nsaved model/now.joblib, model/week.joblib, docs/week_model.js")
