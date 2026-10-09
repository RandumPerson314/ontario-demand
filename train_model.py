"""Train every model locally:  python train_model.py

Creates model/{now,next,week,poly}.joblib and docs/{week_model,poly_model,backtest}.js."""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae

from features import (NEXT_FEATURES, NOW_FEATURES, POLY_FEATURES, WEEK_FEATURES,
                      fetch_archive, fetch_demand, make_features)
from poly_model import export_poly, fit_poly

YEARS = range(2019, 2027)          # last year may be partial; that's fine
TEST_FROM = 2025                   # years >= this are never used to pick settings
HOURLY_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
                     l2_regularization=1.0, random_state=0, early_stopping=False)
WEEK_PARAMS = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
                   l2_regularization=2.0, random_state=0, early_stopping=False)


def fit_eval(F, name, feats, lag, params, cap, baselines):
    """Returns (final model on all data, model trained before TEST_FROM for the backtest)."""
    sub = F.dropna(subset=feats + ["demand"])
    y = sub["demand"] - sub[lag]                       # predict change from a known past value
    train, val = sub[sub.index.year < TEST_FROM - 1], sub[sub.index.year == TEST_FROM - 1]
    test = sub[sub.index.year >= TEST_FROM]
    print(f"\n=== {name}: rows train {len(train)}, val {len(val)}, test {len(test)}")

    m = HistGradientBoostingRegressor(max_iter=cap, **params).fit(train[feats], y[train.index])
    curve = [mae(val["demand"], val[lag] + p) for p in m.staged_predict(val[feats])]
    best = int(np.argmin(curve)) + 1                   # tree count chosen on the validation year
    print("best number of trees:", best)

    trv = sub[sub.index.year < TEST_FROM]              # honest score on the untouched final period
    m2 = HistGradientBoostingRegressor(max_iter=best, **params).fit(trv[feats], y[trv.index])
    pred = test[lag] + m2.predict(test[feats])
    print(f"TEST model MAE {mae(test['demand'], pred):.0f} MW "
          f"(MAPE {np.mean(np.abs(pred - test['demand']) / test['demand']) * 100:.2f}%)")
    for label, p in baselines(test).items():
        print(f"     baseline '{label}': MAE {mae(test['demand'], p):.0f} MW")

    final = HistGradientBoostingRegressor(max_iter=best, **params).fit(sub[feats], y)
    return final, m2


def export_trees(model, feats, X_check, path):
    """Write the trees as JavaScript so the page can evaluate the model in the browser."""
    base = float(np.ravel(model._baseline_prediction)[0])
    trees = []
    for it in model._predictors:
        nd = it[0].nodes
        trees.append(dict(
            f=np.where(nd["is_leaf"] == 1, -1, nd["feature_idx"]).astype(int).tolist(),
            t=[float(v) for v in nd["num_threshold"]],   # full precision: rounding flips splits
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


def score(y, p):
    y, e = np.asarray(y), np.asarray(p) - np.asarray(y)
    return dict(mae=round(float(np.abs(e).mean()), 1), rmse=round(float(np.sqrt((e ** 2).mean())), 1),
                mape=round(float((np.abs(e) / y).mean() * 100), 2), bias=round(float(e.mean()), 1))


def backtest(F, next_m2, week_m2, poly_m2):
    """Next-hour and 7-day accuracy of every model on the same held-out hours."""
    out = {}
    for task, feats, lag, gb, label in (
            ("next", NEXT_FEATURES, "dem_lag2", next_m2, "Last known hour (naive)"),
            ("week", WEEK_FEATURES, "dem_lag168", week_m2, "Same hour last week (naive)")):
        T = F.dropna(subset=feats + POLY_FEATURES + ["demand"])
        T = T[T.index.year >= TEST_FROM]
        y, hrs = T["demand"], T["hour"].values
        preds = {"gb": T[lag] + gb.predict(T[feats]), "poly": poly_m2.predict(T[POLY_FEATURES]),
                 "naive": T[lag]}
        out[task] = dict(
            n=int(len(T)), naive_label=label, start=str(T.index.min())[:10], end=str(T.index.max())[:10],
            models={k: score(y, v) for k, v in preds.items()},
            by_hour={k: [round(float(np.abs(np.asarray(v) - np.asarray(y))[hrs == h].mean()), 1)
                         for h in range(24)] for k, v in preds.items()})
        print(f"backtest {task}: " + ", ".join(f"{k} MAE {v['mae']:.0f}" for k, v in out[task]["models"].items()))
    return out


if __name__ == "__main__":
    w, dem = fetch_archive(YEARS), fetch_demand(YEARS)
    F = make_features(w, dem)
    os.makedirs("model", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    now_m, _ = fit_eval(F, "THIS HOUR model (knows last hour)", NOW_FEATURES, "dem_lag1",
                        HOURLY_PARAMS, 1500, lambda t: {"last hour": t["dem_lag1"],
                                                        "same hour yesterday": t["dem_lag24"],
                                                        "same hour last week": t["dem_lag168"]})
    next_m, next_m2 = fit_eval(F, "NEXT HOUR model (knows demand 2 hours back)", NEXT_FEATURES,
                               "dem_lag2", HOURLY_PARAMS, 1500,
                               lambda t: {"last known hour": t["dem_lag2"],
                                          "same hour last week": t["dem_lag168"]})
    week_m, week_m2 = fit_eval(F, "WEEK model (7-day forecast)", WEEK_FEATURES, "dem_lag168",
                               WEEK_PARAMS, 700,
                               lambda t: {"same hour last week": t["dem_lag168"],
                                          "avg of last 2 weeks": (t["dem_lag168"] + t["dem_lag336"]) / 2})

    P = F.dropna(subset=POLY_FEATURES + ["demand"])
    Ptr = P[P.index.year < TEST_FROM]
    poly_m2 = fit_poly(Ptr, Ptr["demand"])
    Pte = P[P.index.year >= TEST_FROM]
    print(f"\n=== POLYNOMIAL: TEST MAE {mae(Pte['demand'], poly_m2.predict(Pte[POLY_FEATURES])):.0f} MW")
    poly = fit_poly(P, P["demand"])

    for name, m in (("now", now_m), ("next", next_m), ("week", week_m), ("poly", poly)):
        joblib.dump(m, f"model/{name}.joblib", compress=3)
    export_trees(week_m, WEEK_FEATURES, F.dropna(subset=WEEK_FEATURES + ["demand"]), "docs/week_model.js")
    export_poly(poly, F, "docs/poly_model.js")

    bt = backtest(F, next_m2, week_m2, poly_m2)
    with open("docs/backtest.js", "w") as fh:
        fh.write("window.BT=" + json.dumps(bt) + ";")
    print("\nsaved model/*.joblib and docs/week_model.js, poly_model.js, backtest.js")
