"""Train every model locally:  python train_model.py     (needs: pip install -r requirements-train.txt)

Creates model/<family>_<variant>.joblib and docs/{week_model,poly_model,backtest}.js.
The neural networks use the GPU automatically if PyTorch can see one."""
import json
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

from features import (FAMILIES, WEEK_FEATURES, base_col, feature_list, fetch_archive, fetch_demand,
                      make_features, variants)
from models import HOURLY_PARAMS, WEEK_PARAMS, Predictor, fit_gbm, fit_nn, fit_poly
from poly_model import export_poly

YEARS = range(2019, 2027)          # last year may be partial; that's fine
TEST_FROM = 2025                   # years >= this are never used to choose any settings


def fit_variant(F, fid, v):
    """Returns (final model trained on everything, backtest model trained before TEST_FROM)."""
    kind, feats, base = FAMILIES[fid]["kind"], feature_list(fid, v), base_col(fid, v)
    sub = F.dropna(subset=feats + ["demand"])
    y = sub["demand"] - (sub[base] if base else 0.0)        # predict the change from a known demand value
    train, val = sub[sub.index.year < TEST_FROM - 1], sub[sub.index.year == TEST_FROM - 1]
    trv, test = sub[sub.index.year < TEST_FROM], sub[sub.index.year >= TEST_FROM]
    print(f"\n=== {fid} / {v}: {len(feats)} inputs, rows train {len(train)}, val {len(val)}, test {len(test)}")

    if kind == "gbm":
        params, cap = (HOURLY_PARAMS, 1500) if v in ("h1", "h2") else (WEEK_PARAMS, 700)
        m2, final = fit_gbm(train, y[train.index], val, y[val.index], params, cap, feats,
                            [(trv, y[trv.index]), (sub, y)])
    elif kind == "poly":
        m2, final = fit_poly(trv, y[trv.index], feats), fit_poly(sub, y, feats)
    else:
        _, epochs = fit_nn(train[feats], y[train.index], val[feats], y[val.index])
        print(f"   best number of epochs: {epochs}")
        m2, _ = fit_nn(trv[feats], y[trv.index], epochs=epochs)
        final, _ = fit_nn(sub[feats], y, epochs=epochs)

    bt, fin = Predictor(m2, feats, base), Predictor(final, feats, base)
    print(f"   TEST average error {mae(test['demand'], bt.predict(test)):.0f} MW")
    return fin, bt


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


def backtest(F, bt_models):
    """Next-hour and 7-day accuracy of every model on the same held-out hours."""
    out = {}
    for task, v, naive, label in (("next", "h2", "dem_lag2", "Last known hour (naive)"),
                                  ("week", "wk", "dem_lag168", "Same hour last week (naive)")):
        pick = {fid: bt_models[(fid, v if FAMILIES[fid]["demand"] else "all")] for fid in FAMILIES}
        need = sorted({c for p in pick.values() for c in p.feats})
        T = F.dropna(subset=need + ["demand"])
        T = T[T.index.year >= TEST_FROM]
        y, hrs = T["demand"], T["hour"].values
        preds = {fid: p.predict(T) for fid, p in pick.items()}
        preds["naive"] = T[naive].values
        out[task] = dict(
            n=int(len(T)), naive_label=label, start=str(T.index.min())[:10], end=str(T.index.max())[:10],
            models={k: score(y, v_) for k, v_ in preds.items()},
            by_hour={k: [round(float(np.abs(np.asarray(v_) - np.asarray(y))[hrs == h].mean()), 1)
                         for h in range(24)] for k, v_ in preds.items()})
        print(f"backtest {task}: " + ", ".join(f"{k} {m['mae']:.0f}" for k, m in out[task]["models"].items()))
    return out


if __name__ == "__main__":
    w, dem = fetch_archive(YEARS), fetch_demand(YEARS)
    F = make_features(w, dem)
    os.makedirs("model", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    final, bt = {}, {}
    for fid in FAMILIES:
        for v in variants(fid):
            final[(fid, v)], bt[(fid, v)] = fit_variant(F, fid, v)
            joblib.dump(final[(fid, v)], f"model/{fid}_{v}.joblib", compress=3)

    export_trees(final[("gbm", "wk")].model, WEEK_FEATURES, F.dropna(subset=WEEK_FEATURES + ["demand"]),
                 "docs/week_model.js")
    export_poly(final[("poly0", "all")].model, F, "docs/poly_model.js")
    with open("docs/backtest.js", "w") as fh:
        fh.write("window.BT=" + json.dumps(backtest(F, bt)) + ";")
    print("\nsaved model/*.joblib and docs/week_model.js, poly_model.js, backtest.js")
