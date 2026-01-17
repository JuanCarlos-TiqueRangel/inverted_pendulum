#!/usr/bin/env python3
import os
import time
import math
import csv
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque

import numpy as np
import torch
import gymnasium as gym

# Your GPManager (exact GP, GPU) with .load()/.save()/.predict_torch()
from gp_dynamics import GPManager


# ============================================================
# Config (EDIT HERE, no CLI args)
# ============================================================

@dataclass
class MPPIConfig:
    # -------- env / timing --------
    env_name: str = "InvertedPendulum-v5"
    seed: int = 0
    ctrl_dt_target: float = 0.08     # desired controller period
    realtime: bool = True           # sleep to watch in real time

    # action-hold: each env.step advances env_dt; we hold action N steps to get dt_eff≈ctrl_dt_target
    max_steps_per_action: int = 10  # safety cap

    # run length
    max_episodes: int = 1000
    max_wall_time_sec: float = 10_000.0

    # -------- MPPI --------
    horizon: int = 25
    num_rollouts: int = 4096
    lambda_: float = 1.0
    sigma: float = 1.0

    # Cost weights (balance upright)
    w_x: float = 0.5
    w_theta: float = 40.0
    w_xdot: float = 0.2
    w_thetadot: float = 1.0
    w_u: float = 0.01

    # Optional: encourage exploration via GP uncertainty
    entropy_beta: float = 0.0          # set 0.05–0.5 to encourage exploration; 0 disables
    entropy_use_log: bool = True
    entropy_var_floor: float = 1e-6
    entropy_var_cap: float = 1e2
    entropy_dt_scale: bool = True

    # -------- Reset behavior (YOU CAN CHANGE THESE) --------
    # If ignore_env_done=True, we don't reset just because env says terminated;
    # we reset only using your custom thresholds below.
    ignore_env_done: bool = True

    reset_x_abs: float = 0.9
    reset_theta_abs: float = 1.4
    reset_xdot_abs: float = 25.0
    reset_thetadot_abs: float = 35.0
    max_episode_steps: int = 10_000
    hold_on_reset_sec: float = 0.5

    # -------- Files --------
    models_dir: str = "models"
    # stems (script will resolve .pt automatically)
    gp_stems: Tuple[str, str, str, str] = (
        "gp_invpend_out0_dstate_dt",
        "gp_invpend_out1_dstate_dt",
        "gp_invpend_out2_dstate_dt",
        "gp_invpend_out3_dstate_dt",
    )

    # Seed dataset always included in retraining
    seed_npz_candidates: Tuple[str, str] = (
        "dataset/invpend_v5_dt0p1.npz",
        "dataset/invped_v5_dt0p1.npz",   # typo-safe
    )
    keep_seed: bool = True

    # -------- Online buffer / retraining --------
    log_dir: str = "logs_invpend"
    max_log_points: int = 200_000

    retrain_every_episodes: int = 10   # like monster car
    min_points_to_train: int = 2_000
    min_new_points_between_trains: int = 500

    # Exact GP is expensive -> subsample to this many training points
    N_target_train: int = 1500

    train_kernel: str = "RQ"
    train_lr: float = 0.03
    train_iters: int = 300

    # cap window before subsampling (keeps memory stable)
    max_points_for_train: int = 50_000

    # -------- Learning curve plot --------
    live_plot: bool = True
    live_plot_save_png: bool = True


# ============================================================
# Utils
# ============================================================

def resolve_model_path(models_dir: str, stem: str) -> str:
    """
    Accepts:
      - models/stem
      - models/stem.pt
    Returns the existing path if found, else default to .pt.
    """
    p1 = os.path.join(models_dir, stem)
    p2 = p1 + ".pt"
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return p2

def load_seed_npz(cfg: MPPIConfig):
    path = None
    for cand in cfg.seed_npz_candidates:
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        raise FileNotFoundError(f"Could not find seed npz. Tried: {cfg.seed_npz_candidates}")

    D = np.load(path)
    dt = float(np.asarray(D["dt"]).reshape(()))
    s = np.asarray(D["s"], dtype=np.float32)         # (N,4)
    u = np.asarray(D["u"], dtype=np.float32).reshape(-1, 1)  # (N,1)
    s_next = np.asarray(D["s_next"], dtype=np.float32)  # (N,4)

    N = min(len(s), len(u), len(s_next))
    s, u, s_next = s[:N], u[:N], s_next[:N]

    X = np.concatenate([s, u], axis=1)       # (N,5)
    Y = (s_next - s) / dt                    # (N,4)  = dstate/dt

    return path, dt, X, Y

def infer_env_step_dt(env_wrapped) -> float:
    # Gymnasium MuJoCo envs often expose dt
    if hasattr(env_wrapped, "dt"):
        try:
            return float(env_wrapped.dt)
        except Exception:
            pass
    u = env_wrapped.unwrapped
    if hasattr(u, "dt"):
        try:
            return float(u.dt)
        except Exception:
            pass
    # fallback
    return 0.02

def need_custom_reset(cfg: MPPIConfig, obs: np.ndarray, steps_in_ep: int) -> bool:
    if obs is None or (not np.isfinite(obs).all()):
        return True
    x, th, xd, thd = map(float, obs)
    if abs(x) > cfg.reset_x_abs:
        return True
    if abs(th) > cfg.reset_theta_abs:
        return True
    if abs(xd) > cfg.reset_xdot_abs:
        return True
    if abs(thd) > cfg.reset_thetadot_abs:
        return True
    if steps_in_ep >= cfg.max_episode_steps:
        return True
    return False


# ============================================================
# MPPI + Online Retraining Controller
# ============================================================

class InvPendMPPILearner:
    def __init__(self, cfg: MPPIConfig):
        self.cfg = cfg
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.models_dir, exist_ok=True)

        # ----- device -----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("torch device:", self.device)

        # ----- env -----
        self.env_wrapped = gym.make(cfg.env_name, render_mode="human", reset_noise_scale=0.05)
        self.env = self.env_wrapped.unwrapped  # similar spirit to monster car (avoid wrapper enforcement)
        self.obs, _ = self.env_wrapped.reset(seed=cfg.seed)

        self.env_step_dt = infer_env_step_dt(self.env_wrapped)
        self.steps_per_action = max(1, int(round(cfg.ctrl_dt_target / self.env_step_dt)))
        self.steps_per_action = min(self.steps_per_action, cfg.max_steps_per_action)
        self.dt_eff = self.steps_per_action * self.env_step_dt

        if abs(self.dt_eff - cfg.ctrl_dt_target) > 1e-6:
            print(f"[WARN] ctrl_dt_target={cfg.ctrl_dt_target:.3f}s not divisible by env_step_dt={self.env_step_dt:.3f}s "
                  f"-> steps_per_action={self.steps_per_action}, dt_eff={self.dt_eff:.3f}s")

        # action bounds
        self.u_min = float(self.env_wrapped.action_space.low[0])
        self.u_max = float(self.env_wrapped.action_space.high[0])
        print(f"env_step_dt={self.env_step_dt:.3f}s | steps_per_action={self.steps_per_action} | dt_eff={self.dt_eff:.3f}s")
        print(f"action bounds: [{self.u_min:.2f}, {self.u_max:.2f}]")

        # ----- load seed dataset -----
        self.seed_path, self.seed_dt, self.X_seed, self.Y_seed = load_seed_npz(cfg)
        print(f"Loaded seed NPZ: {self.seed_path} | seed_dt={self.seed_dt:.3f} | X_seed={self.X_seed.shape}")

        # ----- load GP models -----
        self.gp_paths = [resolve_model_path(cfg.models_dir, s) for s in cfg.gp_stems]
        self._load_models()

        # MPPI warm start
        self.plan: Optional[torch.Tensor] = None

        # ----- online logs (transition buffer) -----
        self.episode_id = 0
        self.last_train_size = 0

        self.log_s = deque(maxlen=cfg.max_log_points)       # (4,)
        self.log_u = deque(maxlen=cfg.max_log_points)       # scalar
        self.log_sn = deque(maxlen=cfg.max_log_points)      # (4,)
        self.log_ep = deque(maxlen=cfg.max_log_points)      # int

        self.log_lock = threading.Lock()
        self.model_lock = threading.Lock()

        # ----- retraining state -----
        self.training = False
        self.reload_pending = False
        self.train_thread: Optional[threading.Thread] = None

        # ----- episode timing -----
        self.ep_steps = 0
        self.ep_start_wall = None

        # metrics CSV
        self.metrics_path = os.path.join(cfg.log_dir, "episode_metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "time_alive_sec", "steps", "retrain_started"])

        # live plot (episode vs time-alive)
        self.live_plot_ok = False
        self.ep_hist: List[int] = []
        self.t_hist: List[float] = []
        if cfg.live_plot:
            self._init_live_plot()

    # -----------------------
    # plotting
    # -----------------------
    def _init_live_plot(self):
        try:
            import matplotlib.pyplot as plt
            self._plt = plt
            plt.ion()
            self.fig, self.ax = plt.subplots()
            (self.line,) = self.ax.plot([], [], marker="o")
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Time alive [s] (higher is better)")
            self.ax.grid(True)
            self.live_plot_ok = True
            print("[plot] live learning curve enabled.")
        except Exception as e:
            self.live_plot_ok = False
            print(f"[plot] disabled: {e}")

    def _update_live_plot(self, ep: int, t_alive: float):
        if not self.live_plot_ok:
            return
        self.ep_hist.append(ep)
        self.t_hist.append(t_alive)

        self.line.set_data(self.ep_hist, self.t_hist)
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

        if self.cfg.live_plot_save_png:
            out = os.path.join(self.cfg.log_dir, "learning_curve.png")
            self.fig.savefig(out, dpi=150)

    # -----------------------
    # model load / reload
    # -----------------------
    def _load_models(self):
        # load all 4
        self.gps = []
        for p in self.gp_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing GP model file: {p}")
            gp = GPManager.load(p, device=self.device)
            gp.device = self.device
            self.gps.append(gp)
        print("Loaded GP models:")
        for i, p in enumerate(self.gp_paths):
            try:
                print(f"  [{i}] {p} | size={os.path.getsize(p)} bytes | mtime={os.path.getmtime(p):.0f}")
            except Exception:
                print(f"  [{i}] {p}")

    def _reload_models_if_ready(self):
        if not self.reload_pending or self.training:
            return
        with self.model_lock:
            self._load_models()
            self.plan = None
        self.reload_pending = False
        print("[hot-swap] reloaded GP models.")

    # -----------------------
    # logging
    # -----------------------
    def _log_transition(self, s: np.ndarray, u: float, sn: np.ndarray):
        with self.log_lock:
            self.log_s.append(s.astype(np.float32).copy())
            self.log_u.append(float(u))
            self.log_sn.append(sn.astype(np.float32).copy())
            self.log_ep.append(int(self.episode_id))

    def _snapshot_dataset(self):
        with self.log_lock:
            s = np.asarray(list(self.log_s), dtype=np.float32)
            u = np.asarray(list(self.log_u), dtype=np.float32).reshape(-1, 1)
            sn = np.asarray(list(self.log_sn), dtype=np.float32)
            ep = np.asarray(list(self.log_ep), dtype=np.int64)
        return s, u, sn, ep

    def _save_snapshot_npz(self, s, u, sn, ep):
        out = os.path.join(self.cfg.log_dir, f"dataset_ep{self.episode_id:04d}.npz")
        np.savez_compressed(out, dt=np.float32(self.dt_eff), s=s, u=u.squeeze(-1), s_next=sn, episode_id=ep)
        print(f"[log] saved snapshot: {out}")

    # -----------------------
    # MPPI dynamics / cost
    # -----------------------
    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states: (K,4) = [x, theta, xdot, thetadot]
        x = states[:, 0]
        th = states[:, 1]
        xd = states[:, 2]
        thd = states[:, 3]
        u = actions

        c = (
            self.cfg.w_x * x * x +
            self.cfg.w_theta * th * th +
            self.cfg.w_xdot * xd * xd +
            self.cfg.w_thetadot * thd * thd +
            self.cfg.w_u * u * u
        )
        return c

    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        # X = [x, theta, xdot, thetadot, u]
        X = torch.cat([states, actions.unsqueeze(1)], dim=1)  # (K,5)

        with self.model_lock:
            if self.cfg.entropy_beta <= 0.0:
                means = []
                for gp in self.gps:
                    m, _ = gp.predict_torch(X)
                    means.append(m)
                dstate_mean = torch.stack(means, dim=1)  # (K,4)
                dstate_var = None
            else:
                means, vars_ = [], []
                for gp in self.gps:
                    m, v = gp.predict_torch(X)
                    means.append(m)
                    vars_.append(v)
                dstate_mean = torch.stack(means, dim=1)  # (K,4)
                dstate_var = torch.stack(vars_, dim=1)   # (K,4)  (variance)

        dt = float(self.dt_eff)
        next_states = states + dstate_mean * dt

        if self.cfg.entropy_beta <= 0.0:
            return next_states, None

        # entropy proxy from predictive variance
        v = torch.clamp(dstate_var, min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        if self.cfg.entropy_dt_scale:
            v = v * (dt * dt)

        if self.cfg.entropy_use_log:
            ent = 0.5 * torch.log(v).sum(dim=1)  # (K,)
        else:
            ent = v.sum(dim=1)

        ent = torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)
        return next_states, ent

    @torch.no_grad()
    def mppi_action(self, x0_np: np.ndarray) -> float:
        cfg = self.cfg
        H, K = cfg.horizon, cfg.num_rollouts

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)  # (4,)
        u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan

        eps = torch.randn(K, H, device=self.device) * cfg.sigma
        U = torch.clamp(u_init.unsqueeze(0) + eps, self.u_min, self.u_max)

        states = x0.unsqueeze(0).repeat(K, 1)
        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        beta = float(cfg.entropy_beta)

        for t in range(H):
            u_t = U[:, t]
            stage = self.stage_cost_torch(states, u_t)
            states, ent = self.gp_step_batch_torch(states, u_t)
            if ent is not None:
                stage = stage - beta * ent
            costs = costs + stage

        J_min = costs.min()
        w = torch.exp(-(costs - J_min) / cfg.lambda_)
        wsum = w.sum() + 1e-8

        du = (w.unsqueeze(1) * eps).sum(dim=0) / wsum
        u_new = torch.clamp(u_init + du, self.u_min, self.u_max)

        self.plan = u_new.detach()
        return float(u_new[0].detach().cpu())

    # -----------------------
    # retraining (monster-car style)
    # -----------------------
    def _start_retrain_async(self, force: bool = False) -> bool:
        if self.training:
            print("[train] already training; skip.")
            return False

        with self.log_lock:
            n = len(self.log_s)

        if not force:
            if n < self.cfg.min_points_to_train:
                print(f"[train] not enough data: {n} < {self.cfg.min_points_to_train}")
                return False
            if (n - self.last_train_size) < self.cfg.min_new_points_between_trains:
                print("[train] not enough new data since last train; skip.")
                return False

        s, u, sn, ep = self._snapshot_dataset()
        self._save_snapshot_npz(s, u, sn, ep)

        # cap window
        M = self.cfg.max_points_for_train
        if len(s) > M:
            s, u, sn, ep = s[-M:], u[-M:], sn[-M:], ep[-M:]

        n_at_start = n
        self.training = True
        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(s, u, sn, n_at_start),
            daemon=True,
        )
        self.train_thread.start()
        print("[train] started retrain thread.")
        return True

    def _train_worker(self, s: np.ndarray, u: np.ndarray, sn: np.ndarray, n_at_start: int):
        t0 = time.perf_counter()
        try:
            # build online X,Y
            dt = float(self.dt_eff)
            X_online = np.concatenate([s, u], axis=1)       # (N,5)
            Y_online = (sn - s) / dt                        # (N,4)

            # include seed (always)
            if self.cfg.keep_seed:
                X_full = np.concatenate([self.X_seed, X_online], axis=0)
                Y_full = np.concatenate([self.Y_seed, Y_online], axis=0)
            else:
                X_full, Y_full = X_online, Y_online

            N_full = X_full.shape[0]

            # subsample to N_target_train for exact GP
            rng = np.random.default_rng(0)
            N_tgt = int(self.cfg.N_target_train)
            if N_full > N_tgt:
                if self.cfg.keep_seed:
                    # keep ALL seed points if possible; sample the rest
                    n_seed = self.X_seed.shape[0]
                    if n_seed >= N_tgt:
                        idx = rng.choice(n_seed, size=N_tgt, replace=False)
                        X_train = self.X_seed[idx]
                        Y_train = self.Y_seed[idx]
                    else:
                        # keep seed, sample from online to fill
                        need = N_tgt - n_seed
                        n_online = X_online.shape[0]
                        idx_on = rng.choice(n_online, size=min(need, n_online), replace=False)
                        X_train = np.concatenate([self.X_seed, X_online[idx_on]], axis=0)
                        Y_train = np.concatenate([self.Y_seed, Y_online[idx_on]], axis=0)
                else:
                    idx = rng.choice(N_full, size=N_tgt, replace=False)
                    X_train = X_full[idx]
                    Y_train = Y_full[idx]
            else:
                X_train, Y_train = X_full, Y_full

            # train 4 GPs
            gps_new = []
            for d in range(4):
                gp = GPManager(
                    kernel=self.cfg.train_kernel,
                    lr=self.cfg.train_lr,
                    iters=self.cfg.train_iters,
                    device=self.device,
                )
                gp.fit(X_train, Y_train[:, d])
                gps_new.append(gp)

            # atomic save + replace
            for d, stem in enumerate(self.cfg.gp_stems):
                out_path = resolve_model_path(self.cfg.models_dir, stem)
                tmp_path = out_path + ".tmp"
                gps_new[d].save(tmp_path)
                os.replace(tmp_path, out_path)

            self.last_train_size = n_at_start
            self.reload_pending = True

            elapsed = time.perf_counter() - t0
            print(f"[train] done in {elapsed:.2f}s | N_full={N_full} | N_train={X_train.shape[0]} "
                  f"| kernel={self.cfg.train_kernel} | iters={self.cfg.train_iters}")

            # fingerprints
            for d, stem in enumerate(self.cfg.gp_stems):
                p = resolve_model_path(self.cfg.models_dir, stem)
                try:
                    print(f"  model[{d}] {p} | size={os.path.getsize(p)} | mtime={os.path.getmtime(p):.0f}")
                except Exception:
                    pass

        except Exception as e:
            print(f"[train] FAILED: {e}")
        finally:
            self.training = False

    # -----------------------
    # episode metric
    # -----------------------
    def _record_episode_metric(self, retrain_started: bool):
        t_alive = float(self.ep_steps) * float(self.dt_eff)
        ep = int(self.episode_id)

        print(f"[episode] {ep} time_alive={t_alive:.3f}s steps={self.ep_steps} retrain_started={int(retrain_started)}")

        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, t_alive, self.ep_steps, int(retrain_started)])

        self._update_live_plot(ep, t_alive)

    # -----------------------
    # main loop
    # -----------------------
    def run(self):
        start_wall = time.perf_counter()
        last_reset_wall = time.perf_counter()

        self.ep_steps = 0
        self.ep_start_wall = time.perf_counter()

        try:
            for ep in range(self.cfg.max_episodes):
                self.episode_id = ep
                self.ep_steps = 0
                self.ep_start_wall = time.perf_counter()

                # reset env
                self.obs, _ = self.env_wrapped.reset(seed=self.cfg.seed + ep)
                if self.cfg.realtime:
                    time.sleep(self.cfg.hold_on_reset_sec)

                done = False
                while not done:
                    if (time.perf_counter() - start_wall) > self.cfg.max_wall_time_sec:
                        print("[stop] max_wall_time_sec reached.")
                        return

                    # pause MPPI during training/reload (monster-car behavior)
                    if self.training or self.reload_pending:
                        self.plan = None
                        # still step with zero force so viewer stays responsive
                        u_cmd = 0.0
                    else:
                        # MPPI action
                        try:
                            u_cmd = self.mppi_action(self.obs.astype(np.float32))
                        except Exception as e:
                            print(f"[mppi] error: {e}")
                            u_cmd = 0.0

                    u_cmd = float(np.clip(u_cmd, self.u_min, self.u_max))
                    action = np.array([u_cmd], dtype=np.float32)

                    s_k = self.obs.astype(np.float32).copy()

                    # hold action for steps_per_action env steps
                    obs_next = self.obs
                    terminated_any = False
                    truncated_any = False

                    step_start = time.perf_counter()
                    for _ in range(self.steps_per_action):
                        obs_next, r, terminated, truncated, info = self.env.step(action)
                        terminated_any |= bool(terminated)
                        truncated_any |= bool(truncated)

                        # if we are honoring env done, break early
                        if (not self.cfg.ignore_env_done) and (terminated or truncated):
                            break

                    sn_k = obs_next.astype(np.float32).copy()

                    # log ONLY if not training/reloading (like monster-car)
                    if not (self.training or self.reload_pending):
                        self._log_transition(s_k, u_cmd, sn_k)

                    self.obs = obs_next
                    self.ep_steps += 1

                    # hot-swap when ready
                    self._reload_models_if_ready()

                    # termination logic
                    if self.cfg.ignore_env_done:
                        done = need_custom_reset(self.cfg, self.obs, self.ep_steps)
                    else:
                        done = bool(terminated_any or truncated_any) or need_custom_reset(self.cfg, self.obs, self.ep_steps)

                    # realtime pacing
                    if self.cfg.realtime:
                        elapsed = time.perf_counter() - step_start
                        sleep_t = self.dt_eff - elapsed
                        if sleep_t > 0:
                            time.sleep(sleep_t)

                # ---- episode ended ----
                ep_num_1based = ep + 1
                do_retrain = (self.cfg.retrain_every_episodes > 0) and (ep_num_1based % self.cfg.retrain_every_episodes == 0)
                retrain_started = False
                if do_retrain:
                    retrain_started = self._start_retrain_async(force=True)

                self._record_episode_metric(retrain_started=retrain_started)

        finally:
            self.env_wrapped.close()
            try:
                if self.live_plot_ok:
                    self._plt.ioff()
                    self._plt.show()
            except Exception:
                pass


# ============================================================
# main
# ============================================================

def main():
    cfg = MPPIConfig()
    learner = InvPendMPPILearner(cfg)
    learner.run()

if __name__ == "__main__":
    main()
