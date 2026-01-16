#!/usr/bin/env python3
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Try your GPManager first; fallback to sklearn if unavailable.
try:
    from gp_dynamics import GPManager
    HAVE_GPMANAGER = True
except Exception:
    HAVE_GPMANAGER = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, ConstantKernel as C
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


@dataclass
class TrainConfig:
    npz_path: str = "utils/invpend_run_dt0p1.npz"
    N_target: int = 2000
    iters: int = 300
    kernel: str = "RQ"     # for GPManager
    out_dir: str = "models_invpend"
    seed: int = 0


def build_invpend_dataset(
    obs: np.ndarray,
    u: np.ndarray,
    dt: float,
    episode_id: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    obs[t] = [x, theta, x_dot, theta_dot]
    X_t = [x_t, theta_t, x_dot_t, theta_dot_t, u_t]
    Y_t = (obs[t+1] - obs[t]) / dt  -> 4 dims
    """
    obs = np.asarray(obs, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32).reshape(-1)

    N = min(len(obs), len(u))
    obs = obs[:N]
    u = u[:N]

    if N < 2:
        raise ValueError(f"Need at least 2 samples, got N={N}")

    if episode_id is None:
        valid = np.ones(N - 1, dtype=bool)
    else:
        ep = np.asarray(episode_id, dtype=np.int64).reshape(-1)[:N]
        valid = (ep[1:] == ep[:-1])

    s_t = obs[:-1][valid]          # (M,4)
    s_tp1 = obs[1:][valid]         # (M,4)
    u_t = u[:-1][valid]            # (M,)

    X = np.concatenate([s_t, u_t[:, None]], axis=1)          # (M,5)
    Y = (s_tp1 - s_t) / float(dt)                            # (M,4)

    if X.shape[0] == 0:
        raise ValueError("No valid transitions after episode_id filtering.")

    return X.astype(np.float32), Y.astype(np.float32)


def select_subset(X: np.ndarray, Y: np.ndarray, N_target: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple, robust selection for a first test:
      - stratify by theta and |theta_dot| bins
      - plus random fill
    Keeps the code small and stable.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N <= N_target:
        return X, Y

    theta = X[:, 1]
    theta_dot = X[:, 3]

    # bins
    n_bins_theta = 25
    n_bins_rate = 15

    def strat_idx(vals, n_bins, per_bin):
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            idx = np.arange(len(vals))
            rng.shuffle(idx)
            return idx[:per_bin]
        bins = np.linspace(vmin, vmax, n_bins + 1)
        chosen = []
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (vals >= bins[i]) & (vals <= bins[i + 1])
            else:
                mask = (vals >= bins[i]) & (vals < bins[i + 1])
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            chosen.extend(idx[:per_bin])
        return np.asarray(chosen, dtype=int)

    per_bin_theta = max(1, int(0.35 * N_target / n_bins_theta))
    per_bin_rate  = max(1, int(0.35 * N_target / n_bins_rate))

    idx_theta = strat_idx(theta, n_bins_theta, per_bin_theta)
    idx_rate  = strat_idx(np.abs(theta_dot), n_bins_rate, per_bin_rate)

    idx = np.unique(np.concatenate([idx_theta, idx_rate]))
    if len(idx) > N_target:
        rng.shuffle(idx)
        idx = idx[:N_target]
    else:
        # fill remaining with random
        remaining = N_target - len(idx)
        pool = np.setdiff1d(np.arange(N), idx, assume_unique=False)
        rng.shuffle(pool)
        idx = np.concatenate([idx, pool[:remaining]])

    return X[idx], Y[idx]


def train_with_gpmanager(X: np.ndarray, Y: np.ndarray, cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    gps = []
    for d in range(Y.shape[1]):
        gp = GPManager(kernel=cfg.kernel, iters=cfg.iters)
        gp.fit(X, Y[:, d])
        gp.save(os.path.join(cfg.out_dir, f"gp_invpend_dyn_{d}.pt"))
        gps.append(gp)
        print(f"[train] Saved GP[{d}] -> {cfg.out_dir}/gp_invpend_dyn_{d}.pt")
    return gps


def train_with_sklearn(X: np.ndarray, Y: np.ndarray, cfg: TrainConfig):
    if not HAVE_SKLEARN:
        raise RuntimeError("Neither GPManager nor sklearn is available in your environment.")

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Simple kernels; you can tune later.
    kernel = C(1.0, (1e-2, 1e2)) * RationalQuadratic(length_scale=1.0, alpha=1.0)

    models = []
    for d in range(Y.shape[1]):
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True, random_state=cfg.seed)
        gpr.fit(X, Y[:, d])
        models.append(gpr)
        # Save via numpy (pickle is also fine)
        import joblib
        joblib.dump(gpr, os.path.join(cfg.out_dir, f"sk_gp_invpend_dyn_{d}.joblib"))
        print(f"[train] Saved sklearn GP[{d}] -> {cfg.out_dir}/sk_gp_invpend_dyn_{d}.joblib")
    return models


def main():
    cfg = TrainConfig()

    D = np.load(cfg.npz_path)
    for k in ("dt", "obs", "u"):
        if k not in D.files:
            raise KeyError(f"NPZ missing '{k}'. Found: {list(D.files)}")

    dt = float(np.asarray(D["dt"]).reshape(()))
    obs = np.asarray(D["obs"], dtype=np.float32)
    u = np.asarray(D["u"], dtype=np.float32).reshape(-1)
    episode_id = np.asarray(D["episode_id"], dtype=np.int64).reshape(-1) if "episode_id" in D.files else None

    X, Y = build_invpend_dataset(obs=obs, u=u, dt=dt, episode_id=episode_id)
    print("[train] Full dataset:", X.shape, Y.shape, "dt=", dt)

    Xs, Ys = select_subset(X, Y, N_target=cfg.N_target, seed=cfg.seed)
    print("[train] Selected:", Xs.shape, Ys.shape)

    if HAVE_GPMANAGER:
        print("[train] Using GPManager.")
        train_with_gpmanager(Xs, Ys, cfg)
    else:
        print("[train] GPManager not found; using sklearn.")
        train_with_sklearn(Xs, Ys, cfg)


if __name__ == "__main__":
    main()
