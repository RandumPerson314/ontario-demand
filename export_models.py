"""Write each model's parameters as JavaScript so the website's explorer can evaluate it in the browser
(trees for gradient boosting, coefficients for polynomials, weights for neural networks)."""
import json
import os

import numpy as np

from features import BASE_TEMP, FAMILIES, POLY_CORE


def _tree_spec(model, feats, X_check):
    base = float(np.ravel(model._baseline_prediction)[0])
    trees = []
    for it in model._predictors:
        nd = it[0].nodes
        trees.append(dict(
            f=np.where(nd["is_leaf"] == 1, -1, nd["feature_idx"]).astype(int).tolist(),
            t=[float(v) for v in nd["num_threshold"]],   # full precision: rounding flips splits
            l=nd["left"].astype(int).tolist(), r=nd["right"].astype(int).tolist(),
            v=[round(float(v), 3) for v in nd["value"]]))

    def walk(x):                                         # same algorithm as the JavaScript
        s = base
        for T in trees:
            n = 0
            while T["f"][n] >= 0:
                n = T["l"][n] if x[T["f"][n]] <= T["t"][n] else T["r"][n]
            s += T["v"][n]
        return s

    Xc = X_check[feats].iloc[::max(1, len(X_check) // 100)].head(100)
    diff = float(np.abs(np.array([walk(r) for r in Xc.values]) - model.predict(Xc)).max())
    return dict(base=base, trees=trees), diff


def _poly_spec(pipe, feats):
    pre = pipe.named_steps["columntransformer"]
    sc2, ridge = pipe.named_steps["standardscaler"], pipe.named_steps["ridgecv"]
    pp = pre.named_transformers_["poly"]
    sc1, poly = pp.named_steps["standardscaler"], pp.named_steps["polynomialfeatures"]
    coef = ridge.coef_ / sc2.scale_                      # fold the second scaler into the coefficients
    icpt = float(ridge.intercept_ - np.sum(ridge.coef_ * sc2.mean_ / sc2.scale_))
    core = [c for c in POLY_CORE if c in feats]
    return dict(core=core, linear=[c for c in feats if c not in core], mean=sc1.mean_.tolist(),
                scale=sc1.scale_.tolist(), powers=poly.powers_.tolist(), coef=coef.tolist(), icpt=icpt)


def _nn_spec(net, keep=2):
    """The explorer gets the first `keep` networks of the ensemble (the full ensemble would make the file
    several times larger); its numbers can differ from the site's own forecast by a few MW."""
    r = lambda a: np.round(a, 4).tolist()
    return dict(mu=r(net.mu), sd=r(net.sd), ym=float(net.ym), ys=float(net.ys), ensemble=len(net.nets),
                nets=[[[r(W), r(b)] for W, b in layers] for layers in net.nets[:keep]])


def export_model(fid, pred, X_check, path):
    kind, feats = FAMILIES[fid]["kind"], pred.feats
    note = ""
    if kind == "gbm":
        spec, diff = _tree_spec(pred.model, feats, X_check)
        note = f"self-check max difference {diff:.3f} MW"
        if diff > 2:
            note += "  <-- WARNING: exported trees do not match the model"
    elif kind == "poly":
        spec = _poly_spec(pred.model, feats)
    else:
        spec = _nn_spec(pred.model)
    spec.update(kind=kind, lag=pred.base, features=feats)
    with open(path, "w") as fh:
        fh.write(f"window.MJ=window.MJ||{{}};MJ.{fid}=" + json.dumps(spec, separators=(",", ":")) + ";")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB) {note}")


def export_common(F, path="docs/explain_common.js"):
    """Slider ranges and the small relationships the explorer needs (feels-like, dew point)."""
    D = F.dropna(subset=["feels", "dew", "humidity", "wind", "temp"])

    def fit(y, cols):
        A = np.column_stack([np.ones(len(D))] + [D[c] for c in cols])
        return np.linalg.lstsq(A, D[y], rcond=None)[0].tolist()

    def q(c, hi=0.98):
        return [float(F[c].quantile(0.02)), float(F[c].median()), float(F[c].quantile(hi))]

    ex = dict(base=BASE_TEMP, feels=fit("feels", ["temp", "humidity", "wind"]), dew=fit("dew", ["temp", "humidity"]),
              stats={**{c: q(c) for c in ["temp", "humidity", "wind", "cloud"]},
                     **{c: q(c, 0.999) for c in ["precip", "snow"]}})
    with open(path, "w") as fh:
        fh.write("window.EX=" + json.dumps(ex, separators=(",", ":")) + ";")
    print(f"wrote {path}")
