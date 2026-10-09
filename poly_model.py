"""Polynomial (ridge) model: fit it, and export its coefficients for docs/polynomial.html."""
import json

import holidays
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from features import BASE_TEMP, CITIES, POLY_CORE, POLY_FEATURES, POLY_LINEAR

DEGREE = 3


def fit_poly(X, y):
    model = make_pipeline(
        ColumnTransformer([
            ("poly", make_pipeline(StandardScaler(),
                                   PolynomialFeatures(DEGREE, include_bias=False)), POLY_CORE),
            ("lin", "passthrough", POLY_LINEAR)]),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 3, 12)))
    return model.fit(X[POLY_FEATURES], y)


def export_poly(model, F, path="docs/poly_model.js"):
    """Write the fitted polynomial as plain numbers the page can evaluate in the browser."""
    pre = model.named_steps["columntransformer"]
    sc2 = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridgecv"]
    pp = pre.named_transformers_["poly"]
    sc1, poly = pp.named_steps["standardscaler"], pp.named_steps["polynomialfeatures"]

    coef = ridge.coef_ / sc2.scale_                  # fold the second scaler into the coefficients
    icpt = float(ridge.intercept_ - np.sum(ridge.coef_ * sc2.mean_ / sc2.scale_))
    D = F.dropna(subset=["feels", "dew", "humidity", "wind", "temp", "year_frac", "demand"])

    def fit(y, cols):                                # feels-like / dew point follow the sliders
        A = np.column_stack([np.ones(len(D))] + [D[c] for c in cols])
        return np.linalg.lstsq(A, D[y], rcond=None)[0].tolist()

    def q(c, hi=0.98):
        return [float(F[c].quantile(0.02)), float(F[c].median()), float(F[c].quantile(hi))]

    hol = holidays.Canada(subdiv="ON", years=range(2019, 2033))
    model_json = dict(
        core=list(POLY_CORE), linear=list(POLY_LINEAR), base=BASE_TEMP, icpt=icpt,
        mean=sc1.mean_.tolist(), scale=sc1.scale_.tolist(),
        powers=poly.powers_.tolist(), coef=coef.tolist(),
        feels=fit("feels", ["temp", "humidity", "wind"]), dew=fit("dew", ["temp", "humidity"]),
        stats={**{c: q(c) for c in ["temp", "humidity", "wind", "cloud"]},
               **{c: q(c, 0.999) for c in ["precip", "snow"]}},
        year=[float(D["year_frac"].min()), float(D["year_frac"].max())],
        solar=[[float(F[(F["month"] == m) & (F["hour"] == h)]["solar"].median())
                for h in range(24)] for m in range(1, 13)],
        mean_demand=float(F["demand"].mean()),
        cities=[list(c) for c in CITIES.values()],
        holidays=sorted(d.isoformat() for d in hol))
    with open(path, "w") as fh:
        fh.write("window.PM=" + json.dumps(model_json, separators=(",", ":")) + ";")
    print(f"wrote {path}")
