"""Small forecasting models with one interface.

All learned models predict the *change* from the last observed value (y - y0) and add it
back, so an untrained/zero model reduces to persistence. Inputs are standardized with
training statistics only.

Small-data regularization (cold start means a few hundred windows): every choice uses the training windows only,
in time order. Ridge picks alpha and Huber epsilon by time-ordered CV on MAE, the forest uses large leaves,
LightGBM picks the number of trees by early stopping on the last 10% of training windows, then refits on all of
them. The change y - y0 is heavy-tailed, so selection is on MAE, the reported metric.
Every predicted change is clipped to the [p1, p99] range of the training changes, so one wild extrapolation
cannot dominate a test day.

Model(seed).fit(X, y, y0) -> self ; .predict(X, y0) -> y_hat
X has shape (n, n_lags * n_channels), channels interleaved per lag (see evaluate.make_windows).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

ALPHAS = np.logspace(-2, 4, 13)
EPSILONS = [1.1, 1.35, 1.75, 2.5, 4.0]  # Huber threshold in robust-scale units; 1.35 is sklearn's default


GAP = 12  # default gap in windows; the runner passes n_lags + horizon_steps


def _val_split(n: int, frac: float = 0.1, gap: int = GAP) -> tuple[slice, slice]:
    """Fit on the head, validate on the time-ordered tail; `gap` windows are skipped so overlapping windows
    (a window spans n_lags + horizon_steps grid steps) do not leak from fit into validation."""
    n_val = max(1, int(frac * n))
    return slice(0, max(1, n - n_val - gap)), slice(n - n_val, n)


def _time_cv_search(est, grid: dict, X, d, cv_splits: int, gap: int):
    """Pick hyperparameters by time-ordered CV on MAE. Very short histories cannot hold cv_splits folds plus gaps:
    fewer folds are used, and None is returned when not even two fit."""
    for k in range(cv_splits, 1, -1):
        if len(X) - gap - k * (len(X) // (k + 1)) > 0 and len(X) // (k + 1) >= 5:
            cv = TimeSeriesSplit(n_splits=k, gap=gap)
            return GridSearchCV(est, grid, cv=cv, scoring="neg_mean_absolute_error").fit(X, d)
    return None


CLIP_Q = (1.0, 99.0)  # percentiles of the training changes y - y0 that bound every predicted change


class _ChangeModel:
    """Shared output step: y_hat = y0 + clip(predicted change, training [p1, p99]).

    `cv_rel_mae_` is the model's own time-ordered validation score, MAE(model) / MAE(persistence) on the same
    validation windows, left NaN when a history is too short to validate; the cold-start policy reads it."""

    clip = True
    cv_rel_mae_ = float("nan")

    def _set_clip(self, d):
        self.d_lo, self.d_hi = np.percentile(d, CLIP_Q) if self.clip else (-np.inf, np.inf)

    def _out(self, y0, d_hat):
        return y0 + np.clip(d_hat, self.d_lo, self.d_hi)


def _cv_rel_mae(search, X, d) -> float:
    """Best CV MAE of a GridSearchCV over the mean persistence (zero change) MAE on the same validation folds."""
    base = np.mean([np.mean(np.abs(d[val])) for _, val in search.cv.split(X)])
    return float(-search.best_score_ / max(base, 1e-12))


class _Scaled(_ChangeModel):
    def _fit_scale(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8

    def _scale(self, X):
        return (X - self.mu) / self.sd


class RidgeDelta(_Scaled):
    name = "ridge"

    def __init__(self, seed: int = 0, alpha: float | None = None, cv_splits: int = 5, gap: int = GAP):
        self.alpha, self.cv_splits, self.gap = alpha, cv_splits, gap  # alpha=None: chosen by time-ordered CV

    def fit(self, X, y, y0):
        self._fit_scale(X)
        Xs, d = self._scale(X), y - y0
        self._set_clip(d)
        if self.alpha is not None:
            self.m = Ridge(alpha=self.alpha).fit(Xs, d)
            return self
        search = _time_cv_search(Ridge(), {"alpha": ALPHAS}, Xs, d, self.cv_splits, self.gap)
        if search is not None:
            self.m, self.alpha_ = search.best_estimator_, search.best_params_["alpha"]
            self.cv_rel_mae_ = _cv_rel_mae(search, Xs, d)
            return self
        self.alpha_ = 1.0  # too short for CV
        self.m = Ridge(alpha=self.alpha_).fit(Xs, d)
        return self

    def predict(self, X, y0):
        return self._out(y0, self.m.predict(self._scale(X)))


class HuberDelta(_Scaled):
    """Linear model with Huber loss: the change y - y0 is heavy-tailed, and an L2 fit chases the spikes of older days."""

    name = "huber"

    def __init__(self, seed: int = 0, epsilon: float | None = None, alpha: float = 1e-4, cv_splits: int = 5,
                 gap: int = GAP, max_iter: int = 500, tol: float = 1e-5):
        self.epsilon, self.alpha, self.cv_splits, self.gap = epsilon, alpha, cv_splits, gap
        self.max_iter, self.tol = max_iter, tol

    def _est(self, epsilon: float = 1.35):
        return HuberRegressor(epsilon=epsilon, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)

    def fit(self, X, y, y0):
        self._fit_scale(X)
        Xs, d = self._scale(X), y - y0
        self._set_clip(d)
        if self.epsilon is not None:
            self.epsilon_, self.m = self.epsilon, self._est(self.epsilon).fit(Xs, d)
            return self
        search = _time_cv_search(self._est(), {"epsilon": EPSILONS}, Xs, d, self.cv_splits, self.gap)
        if search is not None:
            self.m, self.epsilon_ = search.best_estimator_, search.best_params_["epsilon"]
            self.cv_rel_mae_ = _cv_rel_mae(search, Xs, d)
            return self
        self.epsilon_ = 1.35  # too short for CV
        self.m = self._est(self.epsilon_).fit(Xs, d)
        return self

    def predict(self, X, y0):
        return self._out(y0, self.m.predict(self._scale(X)))


class ForestDelta(_ChangeModel):
    name = "random_forest"

    def __init__(self, seed: int = 0, n_estimators: int = 100, max_depth: int | None = 8, min_samples_leaf: int = 20,
                 n_jobs: int = 1, gap: int = GAP):
        # gap is accepted for a uniform interface; the forest does no validation
        self.m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                       random_state=seed, n_jobs=n_jobs)

    def fit(self, X, y, y0):
        self._set_clip(y - y0)
        self.m.fit(X, y - y0)
        return self

    def predict(self, X, y0):
        return self._out(y0, self.m.predict(X))


class LGBMDelta(_ChangeModel):
    name = "lightgbm"

    def __init__(self, seed: int = 0, n_estimators: int = 300, num_leaves: int = 7, patience: int = 20,
                 n_jobs: int = 1, gap: int = GAP):
        import lightgbm as lgb

        self.patience, self.gap = patience, gap
        self.m = lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=0.05,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=seed, n_jobs=n_jobs, verbose=-1)

    def fit(self, X, y, y0):
        import lightgbm as lgb

        d = y - y0
        self._set_clip(d)
        fit, val = _val_split(len(X), gap=self.gap)
        probe = lgb.LGBMRegressor(**self.m.get_params())
        probe.fit(X[fit], d[fit], eval_X=X[val], eval_y=d[val], eval_metric="l1",
                  callbacks=[lgb.early_stopping(self.patience, first_metric_only=True, verbose=False)])
        # the tail is the most recent (most relevant) history: refit on all windows with the chosen number of trees
        self.best_iteration_ = probe.best_iteration_ or probe.n_estimators
        # the early-stopping tail is LightGBM's time-ordered validation: its score there vs persistence
        self.cv_rel_mae_ = float(probe.best_score_["valid_0"]["l1"] / max(np.mean(np.abs(d[val])), 1e-12))
        self.m.set_params(n_estimators=self.best_iteration_)
        self.m.fit(X, d)
        self.n_fit_ = len(X)
        return self

    def predict(self, X, y0):
        return self._out(y0, self.m.predict(X))


class GRUDelta(_Scaled):
    """Tiny GRU (hidden=16, ~1k params). Early stopping on the last 10% of training windows (time order, after a gap).

    The last `n_static` columns of X (calendar features) are repeated at every time step as extra input channels.
    """

    name = "gru"

    def __init__(self, seed: int = 0, n_channels: int = 4, hidden: int = 16, epochs: int = 60,
                 lr: float = 1e-2, patience: int = 8, batch: int = 128, threads: int = 1, gap: int = GAP,
                 n_static: int = 0):
        self.seed, self.C, self.hidden, self.epochs, self.gap = seed, n_channels, hidden, epochs, gap
        self.n_static = n_static
        self.lr, self.patience, self.batch, self.threads = lr, patience, batch, threads

    def _net(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, c, h):
                super().__init__()
                s.gru = nn.GRU(c, h, batch_first=True)
                s.out = nn.Linear(h, 1)

            def forward(s, x):
                _, hN = s.gru(x)
                return s.out(hN[-1]).squeeze(-1)

        return Net(self.C + self.n_static, self.hidden)

    def _tensor(self, X):
        import torch

        Xs = self._scale(X)
        n_dyn = Xs.shape[1] - self.n_static
        seq = Xs[:, :n_dyn].reshape(len(X), -1, self.C)
        if self.n_static:
            static = np.repeat(Xs[:, None, n_dyn:], seq.shape[1], axis=1)
            seq = np.concatenate([seq, static], axis=2)
        return torch.tensor(seq, dtype=torch.float32)

    def fit(self, X, y, y0):
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        self._fit_scale(X)
        d = y - y0
        self._set_clip(d)
        self.dmu, self.dsd = d.mean(), d.std() + 1e-8
        Xt = self._tensor(X)
        dt = torch.tensor((d - self.dmu) / self.dsd, dtype=torch.float32)
        fit, val = _val_split(len(Xt), gap=self.gap)
        Xtr, dtr, Xva, dva = Xt[fit], dt[fit], Xt[val], dt[val]
        self.net = self._net()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(Xtr), self.batch):
                b = perm[i : i + self.batch]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(Xtr[b]), dtr[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(self.net(Xva), dva).item()
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X, y0):
        import torch

        self.net.eval()
        with torch.no_grad():
            d = self.net(self._tensor(X)).numpy()
        return self._out(y0, d * self.dsd + self.dmu)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def __getstate__(self):  # make pickle size measurable without torch module objects
        st = self.__dict__.copy()
        if "net" in st:
            st["net"] = {k: v.numpy() for k, v in st["net"].state_dict().items()}
        return st


class PolicyDelta:
    """Cold-start policy: what a brand-new kit should run on day N, chosen from its own training data only.

    No extra training: each candidate is fit as usual and brings its own time-ordered validation score, cv_rel_mae_ =
    MAE(model) / MAE(persistence) on its validation windows (ridge and huber: their hyperparameter CV folds; LightGBM:
    its early-stopping tail). The candidate with the lowest score is used if it beats persistence there (< 1);
    otherwise the policy predicts persistence. A candidate without a score (history too short to validate) never wins.
    The runner passes the candidates it already fit (from_fitted), so the policy costs nothing beyond them."""

    name = "policy"
    CANDIDATES = ("ridge", "huber", "lightgbm")

    def __init__(self, seed: int = 0, gap: int = GAP, candidates=CANDIDATES):
        self.seed, self.gap, self.candidates = seed, gap, candidates

    def fit(self, X, y, y0):
        return self._choose({n: REGISTRY[n](seed=self.seed, gap=self.gap).fit(X, y, y0) for n in self.candidates})

    @classmethod
    def from_fitted(cls, fitted: dict) -> "PolicyDelta":
        return cls(candidates=tuple(fitted))._choose(fitted)

    def _choose(self, fitted: dict):
        self.cv_rel_mae_ = {n: m.cv_rel_mae_ for n, m in fitted.items()}
        scored = {n: v for n, v in self.cv_rel_mae_.items() if np.isfinite(v)}
        best = min(scored, key=scored.get) if scored else None
        self.choice_ = best if best is not None and scored[best] < 1 else "persistence"
        self.m = fitted.get(self.choice_)
        return self

    def predict(self, X, y0):
        return y0.astype(float) if self.m is None else self.m.predict(X, y0)


REGISTRY = {"ridge": RidgeDelta, "huber": HuberDelta,
            "random_forest": ForestDelta, "lightgbm": LGBMDelta, "gru": GRUDelta,
            "policy": PolicyDelta}
