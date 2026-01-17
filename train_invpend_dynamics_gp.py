#!/usr/bin/env python3
import os
import numpy as np
import torch
from pathlib import Path


from gp_dynamics import GPManager

# ============================================================
# USER SETTINGS (edit here)
# ============================================================
HERE = Path(__file__).resolve().parent
NPZ_PATH = HERE / "dataset" / "invpend_v5_dt0p1.npz"
OUT_DIR  = HERE / "models"

KERNEL    = "RQ"                       # "RBF" | "Matern" | "RQ"
ITERS     = 300
LR        = 0.03

# Exact GP is O(N^3). Keep this modest (e.g., 800–2500). Set <=0 to disable.
N_TARGET  = 1500
SEED      = 0

# GP target:
#  "dstate_dt" -> (s_next - s)/dt   (recommended for rollout integration)
#  "delta"     -> (s_next - s)
TARGET    = "dstate_dt"

# ============================================================
# Helpers
# ============================================================

def load_npz(npz_path: str):
    D = np.load(npz_path)
    for k in ("dt", "s", "u", "s_next"):
        if k not in D.files:
            raise KeyError(f"NPZ missing '{k}'. Found keys: {list(D.files)}")

    dt = float(np.asarray(D["dt"]).reshape(()))
    s = np.asarray(D["s"], dtype=np.float32)              # (N,4)
    u = np.asarray(D["u"], dtype=np.float32).reshape(-1)  # (N,)
    s_next = np.asarray(D["s_next"], dtype=np.float32)    # (N,4)

    if s.ndim != 2 or s.shape[1] != 4:
        raise ValueError(f"Expected s shape (N,4). Got {s.shape}")
    if s_next.shape != s.shape:
        raise ValueError(f"s_next must match s shape. Got {s_next.shape} vs {s.shape}")
    if len(u) != len(s):
        raise ValueError(f"u length must match N. Got len(u)={len(u)} vs N={len(s)}")
    if not (dt > 0.0):
        raise ValueError(f"dt must be > 0. Got {dt}")

    return dt, s, u, s_next


def build_dataset(dt: float, s: np.ndarray, u: np.ndarray, s_next: np.ndarray, target: str):
    # X = [x, theta, x_dot, theta_dot, u]
    X = np.concatenate([s, u[:, None]], axis=1).astype(np.float32)

    if target == "dstate_dt":
        Y = ((s_next - s) / dt).astype(np.float32)   # (N,4)
    elif target == "delta":
        Y = (s_next - s).astype(np.float32)          # (N,4)
    else:
        raise ValueError(f"Unknown TARGET='{target}'. Use 'dstate_dt' or 'delta'.")

    return X, Y


def subsample(X: np.ndarray, Y: np.ndarray, N_target: int, seed: int = 0):
    if N_target <= 0 or X.shape[0] <= N_target:
        return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=N_target, replace=False)
    return X[idx], Y[idx]


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda")
    print(f"[info] device = {device}")

    dt, s, u, s_next = load_npz(NPZ_PATH)
    print(f"[info] loaded {NPZ_PATH}: N={len(s)}, dt={dt:.6f}")
    print("[info] state order: [x, theta, x_dot, theta_dot]")

    X, Y = build_dataset(dt, s, u, s_next, TARGET)
    Xs, Ys = subsample(X, Y, N_TARGET, SEED)
    print(f"[info] training shapes: X={Xs.shape}, Y={Ys.shape} (TARGET={TARGET})")

    os.makedirs(OUT_DIR, exist_ok=True)

    gps = []
    for d in range(Ys.shape[1]):  # 4 outputs
        print(f"\n[train] output dim {d} ...")
        gp = GPManager(kernel=KERNEL, lr=LR, iters=ITERS, device=device)
        gp.fit(Xs, Ys[:, d])

        out_path = os.path.join(OUT_DIR, f"gp_invpend_out{d}_{TARGET}.pt")
        gp.save(out_path)
        print(f"[save] {out_path}")
        gps.append(gp)

    print("\nDone. Trained and saved 4 GPs.")

if __name__ == "__main__":
    main()
