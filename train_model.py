"""Train the demand model locally:  python train_model.py   ->  model/model.joblib"""
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae

from features import FEATURES, HORIZON, fetch_archive, fetch_demand, make_features, predict_demand

YEARS = range(2019, 2027)          # last year may be partial; that's fine
LAG = f"dem_lag{HORIZON}"
PARAMS = dict(learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
              l2_regularization=1.0, random_state=0)

w, dem = fetch_archive(YEARS), fetch_demand(YEARS)
F = make_features(w, dem).dropna(subset=FEATURES + ["demand"])
y = F["demand"] - F[LAG]           # predict the change from the last known hour

train, val, test = F[F.index.year < 2024], F[F.index.year == 2024], F[F.index.year >= 2025]
print(f"rows: train {len(train)}, val {len(val)}, test {len(test)}")

# 1) pick the number of trees on the validation year
m = HistGradientBoostingRegressor(max_iter=1500, **PARAMS).fit(train[FEATURES], y[train.index])
curve = [mae(val["demand"], val[LAG] + p) for p in m.staged_predict(val[FEATURES])]
best = int(np.argmin(curve)) + 1
print("best number of trees:", best)

# 2) honest test score: refit on train+val, score on the untouched final period
trv = F[F.index.year < 2025]
m2 = HistGradientBoostingRegressor(max_iter=best, **PARAMS).fit(trv[FEATURES], y[trv.index])
pred = predict_demand(m2, test)
print(f"\nTEST  model MAE {mae(test['demand'], pred):.0f} MW   "
      f"(MAPE {np.mean(np.abs(pred - test['demand']) / test['demand']) * 100:.2f}%)")
for name, col in [("last hour", LAG), ("same hour yesterday", "dem_lag24"),
                  ("same hour last week", "dem_lag168")]:
    print(f"      baseline '{name}': MAE {mae(test['demand'], test[col]):.0f} MW")

# 3) final model on everything
final = HistGradientBoostingRegressor(max_iter=best, **PARAMS).fit(F[FEATURES], y)
os.makedirs("model", exist_ok=True)
joblib.dump(final, "model/model.joblib", compress=3)
print("\nsaved model/model.joblib")
