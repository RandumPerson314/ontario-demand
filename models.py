"""Fitting and prediction for the three kinds of model: gradient boosting, polynomial, neural network.

Every fitted model is wrapped in a Predictor, which remembers which columns it needs and
whether it predicts demand directly or the *change* from a known recent demand value.
Neural networks are trained with PyTorch but saved as plain NumPy weights, so the hourly
website update does not need PyTorch installed."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from features import POLY_CORE

HOURLY_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=48, min_samples_leaf=40,
                     l2_regularization=1.0, random_state=0, early_stopping=False)
WEEK_PARAMS = dict(learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
                   l2_regularization=2.0, random_state=0, early_stopping=False)


class Predictor:
    """A fitted model plus the columns it needs. predict(X) returns megawatts."""

    def __init__(self, model, feats, base=None):
        self.model, self.feats, self.base = model, list(feats), base

    def predict(self, X):
        out = np.asarray(self.model.predict(X[self.feats]), dtype=float)
        return out + (X[self.base].values if self.base else 0.0)


# ------------------------------------------------------------------ gradient boosting
def fit_gbm(train, ytr, val, yva, params, cap, feats, final_sets):
    """Pick the number of trees on the validation set, then refit on each set in final_sets."""
    m = HistGradientBoostingRegressor(max_iter=cap, **params).fit(train[feats], ytr)
    curve = [mae(yva, p) for p in m.staged_predict(val[feats])]
    best = int(np.argmin(curve)) + 1
    print(f"   best number of trees: {best}")
    return [HistGradientBoostingRegressor(max_iter=best, **params).fit(X[feats], y) for X, y in final_sets]


# ------------------------------------------------------------------ polynomial
def fit_poly(X, y, feats):
    core = [c for c in POLY_CORE if c in feats]
    linear = [c for c in feats if c not in core]
    model = make_pipeline(
        ColumnTransformer([
            ("poly", make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False)), core),
            ("lin", "passthrough", linear)]),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 3, 12)))
    return model.fit(X[feats], y)


# ------------------------------------------------------------------ neural network
class NumpyMLP:
    """Inference-only copy of a trained network (an average of several seeds)."""

    def __init__(self, nets, mu, sd, ym, ys):
        self.nets, self.mu, self.sd, self.ym, self.ys = nets, mu, sd, ym, ys

    def predict(self, X):
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        outs = []
        for layers in self.nets:
            a = Z
            for i, (W, b) in enumerate(layers):
                a = a @ W.T + b
                if i < len(layers) - 1:
                    a = a / (1.0 + np.exp(-np.clip(a, -60, 60)))        # SiLU
            outs.append(a[:, 0])
        return np.mean(outs, axis=0) * self.ys + self.ym


def fit_nn(X, y, Xva=None, yva=None, epochs=40, seeds=3, batch=512):
    """Train with PyTorch (on the GPU if there is one). If validation data is given, also
    returns the best epoch count. Returns (NumpyMLP, best_epochs)."""
    import torch
    from torch import nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ym, ys = float(y.mean()), float(y.std() + 1e-6)
    tx = torch.tensor((X - mu) / sd, device=dev)
    ty = torch.tensor((y - ym) / ys, device=dev).unsqueeze(1)
    if Xva is not None:
        vx = torch.tensor((np.asarray(Xva, dtype=np.float32) - mu) / sd, device=dev)
        vy = np.asarray(yva, dtype=np.float32)

    nets, best_eps = [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Dropout(0.1),
                            nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.1),
                            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = nn.SmoothL1Loss(beta=0.5)
        best = (np.inf, epochs, None)
        for ep in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(len(tx), device=dev)
            for i in range(0, len(tx), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss_fn(net(tx[idx]), ty[idx]).backward()
                opt.step()
            sched.step()
            if Xva is not None:
                net.eval()
                with torch.no_grad():
                    err = float(np.abs(net(vx).cpu().numpy()[:, 0] * ys + ym - vy).mean())
                if err < best[0]:
                    best = (err, ep, [p.detach().clone() for p in net.parameters()])
        if Xva is not None and best[2] is not None:       # keep the weights from the best epoch
            for p, q in zip(net.parameters(), best[2]):
                p.data.copy_(q)
        best_eps.append(best[1])
        nets.append([(m.weight.detach().cpu().numpy().astype(np.float64),
                      m.bias.detach().cpu().numpy().astype(np.float64))
                     for m in net if isinstance(m, nn.Linear)])
    return NumpyMLP(nets, mu.astype(np.float64), sd.astype(np.float64), ym, ys), int(np.mean(best_eps))
