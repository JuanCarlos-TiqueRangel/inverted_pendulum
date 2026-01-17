#!/usr/bin/env python3
import time
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from gp_dynamics import GPManager  # your GPManager with .load() + .predict_torch()

# -----------------------------
# Optional: silence pygame warning
# -----------------------------
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# =============================
# User settings (edit here)
# =============================

DATASET_PATH = Path("dataset") / "invpend_v5_dt0p1.npz"

MODEL_DIR = Path("models")
MODEL_NAMES = [
    "gp_invpend_out0_dstate_dt",
    "gp_invpend_out1_dstate_dt",
    "gp_invpend_out2_dstate_dt",
    "gp_invpend_out3_dstate_dt",
]

ENV_NAME = "InvertedPendulum-v5"

RENDER = True
RESET_NOISE_SCALE = 0.001   # smaller => cleaner starts near upright

# Reset behavior knobs
IGNORE_ENV_DONE = True    # True => keep stepping even if env says terminated (lets it “fall more”)
X_RESET_LIMIT = 0.9        # you pick
THETA_RESET_LIMIT = 1.4    # rad, you pick (only used if IGNORE_ENV_DONE=True)
EPISODE_TIME_LIMIT = 20.0  # seconds (soft reset)

# MPPI knobs
HORIZON = 40
NUM_ROLLOUTS = 2048
LAMBDA = 1.0
SIGMA = 1.6

# Cost weights (balancing)
W_THETA = 100.0
W_THETAD = 10.0
W_X = 1.0
W_XD = 0.5
W_U = 0.01

# =============================
# Helpers
# =============================

def resolve_model_path(stem: str) -> Path:
    p0 = MODEL_DIR / stem
    p1 = (MODEL_DIR / stem).with_suffix(".pt")
    if p0.exists():
        return p0
    if p1.exists():
        return p1
    raise FileNotFoundError(f"Could not find model '{stem}' as {p0} or {p1}")

def make_env_with_dt(dt_target: float):
    """
    Gymnasium v5 supports frame_skip to configure env.dt. :contentReference[oaicite:4]{index=4}
    We pick a frame_skip so env.dt ≈ dt_target (so the real env matches the GP dt).
    """
    # temporary env to read base timestep
    tmp = gym.make(ENV_NAME)
    try:
        base_dt = float(tmp.unwrapped.model.opt.timestep)
    finally:
        tmp.close()

    frame_skip = max(1, int(round(dt_target / base_dt)))

    env = gym.make(
        ENV_NAME,
        render_mode="human" if RENDER else None,
        reset_noise_scale=RESET_NOISE_SCALE,
        frame_skip=frame_skip,
    )
    return env

@torch.no_grad()
def gp_step_batch(gps, x, u, dt):
    """
    x: (K,4)  [x, theta, x_dot, theta_dot]
    u: (K,)   force
    gps[i] predicts d(state_i)/dt
    """
    Xin = torch.cat([x, u.unsqueeze(1)], dim=1)  # (K,5)
    d_list = []
    for i in range(4):
        m, _ = gps[i].predict_torch(Xin)
        d_list.append(m)
    dx = torch.stack(d_list, dim=1)             # (K,4)
    x_next = x + dx * dt
    return x_next

def stage_cost(x, u):
    """
    Quadratic cost for balancing around theta=0, x=0.
    """
    xpos = x[:, 0]
    th   = x[:, 1]
    xd   = x[:, 2]
    thd  = x[:, 3]

    return (
        W_THETA  * th**2 +
        W_THETAD * thd**2 +
        W_X      * xpos**2 +
        W_XD     * xd**2 +
        W_U      * u**2
    )

class MPPI:
    def __init__(self, gps, dt, u_min, u_max, device):
        self.gps = gps
        self.dt = float(dt)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.device = device
        self.u_nom = None  # (H,)

    @torch.no_grad()
    def act(self, x0_np):
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device).view(1, 4)
        H = HORIZON
        K = NUM_ROLLOUTS

        if self.u_nom is None:
            self.u_nom = torch.zeros(H, dtype=torch.float32, device=self.device)

        eps = torch.randn(K, H, device=self.device) * SIGMA
        U = torch.clamp(self.u_nom.unsqueeze(0) + eps, self.u_min, self.u_max)  # (K,H)

        x = x0.repeat(K, 1)   # (K,4)
        cost = torch.zeros(K, dtype=torch.float32, device=self.device)

        for t in range(H):
            u_t = U[:, t]
            cost = cost + stage_cost(x, u_t)
            x = gp_step_batch(self.gps, x, u_t, self.dt)

        cmin = cost.min()
        w = torch.exp(-(cost - cmin) / LAMBDA)
        wsum = w.sum() + 1e-8

        du = (w.unsqueeze(1) * eps).sum(dim=0) / wsum
        self.u_nom = torch.clamp(self.u_nom + du, self.u_min, self.u_max)

        u0 = float(self.u_nom[0].item())
        # shift warm start
        self.u_nom = torch.cat([self.u_nom[1:], torch.zeros(1, device=self.device)])
        return u0

def main():
    # ---- Load dt from dataset ----
    D = np.load(DATASET_PATH)
    dt_model = float(np.asarray(D["dt"]).reshape(()))
    print("Dataset keys:", D.files)
    print("dt_model =", dt_model, "(this is what your GP+MPPI should use)")

    # ---- Build env with env.dt ≈ dt_model (frame_skip) ----
    env = make_env_with_dt(dt_model)
    obs, _ = env.reset(seed=0)
    env_dt = float(getattr(env, "dt", dt_model))
    print("env.dt =", env_dt, " (after choosing frame_skip)")

    # ---- Action bounds from env ----
    u_min = float(env.action_space.low[0])   # [-3] :contentReference[oaicite:5]{index=5}
    u_max = float(env.action_space.high[0])  # [ 3] :contentReference[oaicite:6]{index=6}
    print("Action bounds:", (u_min, u_max))

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch device:", device)

    # ---- Load GP models ----
    gps = []
    for i, stem in enumerate(MODEL_NAMES):
        p = resolve_model_path(stem)
        gp = GPManager.load(str(p), device=device)
        gps.append(gp)
        print(f"Loaded GP[{i}] from {p}")

    ctrl = MPPI(gps=gps, dt=dt_model, u_min=u_min, u_max=u_max, device=device)

    # ---- Live plot ----
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    (l_th,)  = axs[0].plot([], [], lw=2); axs[0].set_ylabel("theta [rad]"); axs[0].grid(True)
    (l_thd,) = axs[1].plot([], [], lw=2); axs[1].set_ylabel("theta_dot [rad/s]"); axs[1].grid(True)
    (l_x,)   = axs[2].plot([], [], lw=2); axs[2].set_ylabel("x [m]"); axs[2].grid(True)
    (l_u,)   = axs[3].plot([], [], lw=2); axs[3].set_ylabel("u [N]"); axs[3].set_xlabel("t [s]"); axs[3].grid(True)

    t_log, th_log, thd_log, x_log, u_log = [], [], [], [], []
    t = 0.0
    ep_start_wall = time.perf_counter()
    last_plot = time.perf_counter()

    try:
        while True:
            # obs order: [x, theta, x_dot, theta_dot] :contentReference[oaicite:7]{index=7}
            x, th, xd, thd = map(float, obs)

            # ---- custom reset rules ----
            if abs(x) > X_RESET_LIMIT:
                obs, _ = env.reset()
                ctrl.u_nom = None
                ep_start_wall = time.perf_counter()
                continue

            if IGNORE_ENV_DONE:
                if abs(th) > THETA_RESET_LIMIT:
                    obs, _ = env.reset()
                    ctrl.u_nom = None
                    ep_start_wall = time.perf_counter()
                    continue

            if (time.perf_counter() - ep_start_wall) > EPISODE_TIME_LIMIT:
                obs, _ = env.reset()
                ctrl.u_nom = None
                ep_start_wall = time.perf_counter()
                continue

            # ---- MPPI action from GP model ----
            u = ctrl.act(np.array([x, th, xd, thd], dtype=np.float32))
            u = float(np.clip(u, u_min, u_max))

            # ---- step env once (env.dt already ~ dt_model) ----
            obs_next, _, terminated, truncated, _ = env.step(np.array([u], dtype=np.float32))

            if (not IGNORE_ENV_DONE) and (terminated or truncated):
                # env terminates when |theta| > 0.2 rad :contentReference[oaicite:8]{index=8}
                obs, _ = env.reset()
                ctrl.u_nom = None
                ep_start_wall = time.perf_counter()
            else:
                obs = obs_next

            # ---- log/plot ----
            t_log.append(t)
            x_log.append(x)
            th_log.append(th)
            thd_log.append(thd)
            u_log.append(u)
            t += env_dt

            now = time.perf_counter()
            if now - last_plot > 0.1 and len(t_log) > 2:
                l_th.set_data(t_log, th_log)
                l_thd.set_data(t_log, thd_log)
                l_x.set_data(t_log, x_log)
                l_u.set_data(t_log, u_log)

                axs[0].set_xlim(max(0.0, t - 10.0), max(10.0, t))
                axs[0].relim(); axs[0].autoscale_view()
                axs[1].relim(); axs[1].autoscale_view()
                axs[2].relim(); axs[2].autoscale_view()
                axs[3].set_ylim(u_min * 1.1, u_max * 1.1)

                fig.canvas.draw()
                fig.canvas.flush_events()
                last_plot = now

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
