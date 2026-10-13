"""Train every model locally:  python train_model.py     (needs: pip install -r requirements-train.txt)

For each of the 12 model variants it:
  * tries the population-weighted temperature alone AND with every city's own temperature added,
    and keeps the per-city version only if it lowers the validation error by at least 0.5%;
  * (neural networks) tries five network shapes and keeps the best;
then refits on all the data and exports the model for the website.
Creates model/<family>_<variant>.joblib and docs/{model_<family>,explain_common,model_info,backtest}.js.
The neural networks use the GPU automatically if PyTorch can see one."""
import json
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

from export_models import export_common, export_model
from features import (FAMILIES, base_col, feature_list, fetch_archive, fetch_demand, make_features,
                      variants)
from models import (HOURLY_PARAMS, WEEK_PARAMS, Predictor, fit_nn, fit_poly, gbm_fit, gbm_select,
                    nn_select, poly_select)

YEARS = range(2019, 2027)          # last year may be partial; that's fine
TEST_FROM = 2025                   # years >= this are never used to choose any settings
FINAL_SEEDS = 5                    # neural networks: size of the final ensemble


def _data(F, feats, base):
    sub = F.dropna(subset=feats + ["demand"])
    y = sub["demand"] - (sub[base] if base else 0.0)        # predict the change from a known demand value
    ix = sub.index.year
    return sub, y, sub[ix < TEST_FROM - 1], sub[ix == TEST_FROM - 1], sub[ix < TEST_FROM], sub[ix >= TEST_FROM]


def fit_variant(F, fid, v):
    """Returns (final model trained on everything, backtest model trained before TEST_FROM)."""
    kind, base = FAMILIES[fid]["kind"], base_col(fid, v)
    params, cap = (HOURLY_PARAMS, 1500) if v in ("h1", "h2") else (WEEK_PARAMS, 700)
    print(f"\n=== {fid} / {v}")
    chosen, cfg = {}, None
    for city in (False, True):
        feats = feature_list(fid, v, city)
        sub, y, train, val, trv, test = _data(F, feats, base)
        a, b = train[feats], val[feats]
        if kind == "gbm":
            sel = gbm_select(train, y[train.index], val, y[val.index], params, cap, feats)
        elif kind == "poly":
            sel = poly_select(train, y[train.index], val, y[val.index], feats)
        else:
            sel = nn_select(a.values, y[train.index].values, b.values, y[val.index].values, cfg=cfg)
            cfg = sel["cfg"]
        chosen[city] = dict(sel=sel, feats=feats)
        label = "with each city's own temperature" if city else "population-weighted temperature"
        print(f"   {label:34s}: {len(feats)} inputs, validation error {sel['val']:.0f} MW")
    use_city = chosen[True]["sel"]["val"] < chosen[False]["sel"]["val"] * 0.995
    sel, feats = chosen[use_city]["sel"], chosen[use_city]["feats"]
    print("   -> using " + ("each city's own temperature" if use_city else "the population-weighted temperature"))

    sub, y, train, val, trv, test = _data(F, feats, base)
    if kind == "gbm":
        m2, final = (gbm_fit(d, y[d.index], feats, params, sel["ntrees"]) for d in (trv, sub))
    elif kind == "poly":
        m2, final = fit_poly(trv, y[trv.index], feats), fit_poly(sub, y, feats)
    else:
        print(f"   final networks: shape {sel['cfg']['hidden']}, {sel['epochs']} epochs, {FINAL_SEEDS} seeds")
        m2 = fit_nn(trv[feats].values, y[trv.index].values, epochs=sel["epochs"], seeds=FINAL_SEEDS, cfg=sel["cfg"])[0]
        final = fit_nn(sub[feats].values, y.values, epochs=sel["epochs"], seeds=FINAL_SEEDS, cfg=sel["cfg"])[0]

    info = dict(city=bool(use_city), val_base=round(chosen[False]["sel"]["val"], 1),
                val_city=round(chosen[True]["sel"]["val"], 1))
    bt, fin = Predictor(m2, feats, base, info), Predictor(final, feats, base, info)
    print(f"   TEST average error {mae(test['demand'], bt.predict(test)):.0f} MW")
    return fin, bt


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

    print()
    export_common(F)
    for fid in FAMILIES:                               # each model's 7-day variant powers its page's explorer
        p = final[(fid, "wk" if FAMILIES[fid]["demand"] else "all")]
        export_model(fid, p, F.dropna(subset=p.feats), f"docs/model_{fid}.js")
    info = {fid: {v: final[(fid, v)].info for v in variants(fid)} for fid in FAMILIES}
    with open("docs/model_info.js", "w") as fh:
        fh.write("window.MI=" + json.dumps(info) + ";")
    with open("docs/backtest.js", "w") as fh:
        fh.write("window.BT=" + json.dumps(backtest(F, bt)) + ";")
    print("\nsaved model/*.joblib and docs/model_*.js, explain_common.js, model_info.js, backtest.js")
