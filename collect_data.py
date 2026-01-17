#!/usr/bin/env python3
import time
import warnings
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path

# Optional: silence pygame/pkg_resources warning
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# ============================================================
# CONFIG (edit everything here)
# ============================================================

ENV_NAME = "InvertedPendulum-v5"
RENDER_MODE = "human"          # "human" or None
RESET_NOISE_SCALE = 0.05       # smaller => cleaner starts near upright

CTRL_DT = 0.08                  # desired controller logging period
DURATION = 180.0                # seconds wall time
SAVE_PATH = Path("dataset") / "invpend_v5_custom.npz"

# --- Reset policy knobs ---
IGNORE_ENV_DONE = True        # True => ignore terminated/truncated from env; use custom reset only
X_RESET_LIMIT = 0.9            # reset if |x| > this
THETA_RESET_LIMIT = 1.4        # reset if |theta| > this (only used if IGNORE_ENV_DONE=True)
EPISODE_TIME_LIMIT = 10.0      # seconds; reset if episode runs longer than this

# --- Try to make env.dt align with CTRL_DT (optional) ---
AUTO_FRAME_SKIP = True         # True => choose frame_skip to make env.dt ≈ CTRL_DT
FRAME_SKIP_OVERRIDE = None     # set an int to force it (e.g., 5), or None to not force

# --- PD + exploration for data richness ---
NOISE_STD = 0.25
KX, KXD, KTH, KTHD = 0.5, 1.0, 20.0, 3.0

# --- Plot settings ---
PLOT_UPDATE_SEC = 0.10
THETA_PLOT_LIM = (-THETA_RESET_LIMIT*1.1, THETA_RESET_LIMIT*1.1)  # just for display (does NOT affect physics)
SHOW_PLOT = True


# ============================================================
# Helpers
# ============================================================

def base_timestep_from_env() -> float:
    tmp = gym.make(ENV_NAME)
    try:
        u = tmp.unwrapped
        # MuJoCo base timestep
        return float(u.model.opt.timestep)
    finally:
        tmp.close()

def make_env():
    kwargs = dict(reset_noise_scale=RESET_NOISE_SCALE)
    if RENDER_MODE is not None:
        kwargs["render_mode"] = RENDER_MODE

    if FRAME_SKIP_OVERRIDE is not None:
        kwargs["frame_skip"] = int(FRAME_SKIP_OVERRIDE)
        return gym.make(ENV_NAME, **kwargs)

    if AUTO_FRAME_SKIP:
        base_dt = base_timestep_from_env()
        frame_skip = max(1, int(round(CTRL_DT / base_dt)))
        kwargs["frame_skip"] = frame_skip
        return gym.make(ENV_NAME, **kwargs)

    return gym.make(ENV_NAME, **kwargs)

def get_step_dt(env) -> float:
    # Gymnasium MuJoCo envs usually expose dt (duration of step()).
    if hasattr(env, "dt"):
        try:
            return float(env.dt)
        except Exception:
            pass
    u = env.unwrapped
    if hasattr(u, "dt"):
        try:
            return float(u.dt)
        except Exception:
            pass
    # Fallback: model.opt.timestep * frame_skip if available
    if hasattr(u, "model") and hasattr(u, "frame_skip"):
        try:
            return float(u.model.opt.timestep) * float(u.frame_skip)
        except Exception:
            pass
    return 0.02

def pd_with_small_noise(obs, noise_std=NOISE_STD):
    """
    InvertedPendulum obs order: [x, theta, x_dot, theta_dot]
    action: force in [-3, 3]
    """
    x, th, xd, thd = obs
    u = -(KX*x + KXD*xd + KTH*th + KTHD*thd)
    u += np.random.randn() * float(noise_std)
    return np.array([u], dtype=np.float32)


# ============================================================
# Main
# ============================================================

def main():
    env = make_env()
    obs, _ = env.reset(seed=0)

    step_dt = get_step_dt(env)

    # Controller dt -> how many env.step() calls per controller action
    steps_per_action = max(1, int(round(CTRL_DT / step_dt)))
    dt_eff = steps_per_action * step_dt

    if abs(dt_eff - CTRL_DT) > 1e-9:
        print(f"[WARN] CTRL_DT={CTRL_DT:.3f}s not divisible by step_dt={step_dt:.3f}s -> dt_eff={dt_eff:.3f}s")

    print(f"Env={ENV_NAME}")
    print(f"step_dt={step_dt:.6f}s | steps_per_action={steps_per_action} | dt_eff={dt_eff:.6f}s")
    if hasattr(env, "frame_skip"):
        print(f"frame_skip={getattr(env, 'frame_skip')}")

    # action bounds
    umin = env.action_space.low.astype(np.float32)
    umax = env.action_space.high.astype(np.float32)

    # live plot
    if SHOW_PLOT:
        plt.ion()
        fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        (l_th,)  = axs[0].plot([], [], lw=2); axs[0].set_ylabel("theta [rad]"); axs[0].grid(True)
        (l_thd,) = axs[1].plot([], [], lw=2); axs[1].set_ylabel("theta_dot [rad/s]"); axs[1].grid(True)
        (l_x,)   = axs[2].plot([], [], lw=2); axs[2].set_ylabel("x [m]"); axs[2].grid(True)
        (l_u,)   = axs[3].plot([], [], lw=2); axs[3].set_ylabel("u [N]"); axs[3].set_xlabel("t [s]"); axs[3].grid(True)

    # logs at dt_eff (~CTRL_DT)
    t_log = []
    s_log = []       # s_k = [x, theta, x_dot, theta_dot]
    u_log = []       # u_k
    sn_log = []      # s_{k+1}
    ep_log = []

    t = 0.0
    ep = 0
    last_plot = time.perf_counter()
    ep_start_wall = time.perf_counter()

    def do_reset(reason: str):
        nonlocal obs, ep, ep_start_wall
        # optional: print reason occasionally
        # print(f"[reset] ep={ep} reason={reason}")
        obs, _ = env.reset()
        ep += 1
        ep_start_wall = time.perf_counter()

    try:
        while t < DURATION:
            # --- custom reset checks (based on current obs) ---
            x, th, xd, thd = map(float, obs)

            if abs(x) > float(X_RESET_LIMIT):
                do_reset("x_limit")
                continue

            if IGNORE_ENV_DONE and abs(th) > float(THETA_RESET_LIMIT):
                do_reset("theta_limit")
                continue

            if (time.perf_counter() - ep_start_wall) > float(EPISODE_TIME_LIMIT):
                do_reset("time_limit")
                continue

            # choose action from current obs
            action = pd_with_small_noise(obs, noise_std=NOISE_STD)
            action = np.clip(action, umin, umax)

            s_k = obs.astype(np.float32).copy()

            terminated_any = False
            truncated_any = False
            obs_next = obs

            # hold action for steps_per_action internal steps
            for _ in range(steps_per_action):
                obs_next, r, terminated, truncated, info = env.step(action)
                terminated_any |= bool(terminated)
                truncated_any |= bool(truncated)

                # render pacing
                time.sleep(step_dt)

                # if we respect env termination, break immediately
                if (not IGNORE_ENV_DONE) and (terminated or truncated):
                    break

            s_kp1 = obs_next.astype(np.float32).copy()

            # log one sample at controller rate
            t_log.append(t)
            s_log.append(s_k)
            u_log.append(float(action[0]))
            sn_log.append(s_kp1)
            ep_log.append(ep)

            # advance
            t += dt_eff
            obs = obs_next

            # reset logic
            if (not IGNORE_ENV_DONE) and (terminated_any or truncated_any):
                do_reset("env_done")

            # plot update
            if SHOW_PLOT:
                now = time.perf_counter()
                if now - last_plot > PLOT_UPDATE_SEC and len(t_log) > 1:
                    S = np.asarray(s_log, dtype=np.float32)
                    tt = np.asarray(t_log, dtype=np.float32)

                    l_th.set_data(tt, S[:, 1])
                    l_thd.set_data(tt, S[:, 3])
                    l_x.set_data(tt, S[:, 0])
                    l_u.set_data(tt, np.asarray(u_log, dtype=np.float32))

                    axs[0].set_xlim(max(0.0, t - 10.0), max(10.0, t))
                    axs[0].set_ylim(*THETA_PLOT_LIM)
                    axs[1].relim(); axs[1].autoscale_view()
                    axs[2].relim(); axs[2].autoscale_view()
                    axs[3].set_ylim(float(umin[0]) * 1.1, float(umax[0]) * 1.1)

                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    last_plot = now

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        if SHOW_PLOT:
            plt.ioff()

    if len(t_log) < 2:
        print("No data collected; nothing saved.")
        return

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        SAVE_PATH,
        dt=np.float32(dt_eff),
        t=np.asarray(t_log, dtype=np.float32),
        s=np.asarray(s_log, dtype=np.float32),        # (N,4)
        u=np.asarray(u_log, dtype=np.float32),        # (N,)
        s_next=np.asarray(sn_log, dtype=np.float32),  # (N,4)
        episode_id=np.asarray(ep_log, dtype=np.int64),
    )
    print(f"Saved {SAVE_PATH} | N={len(t_log)} | episodes={ep+1} | dt={dt_eff:.6f}s")

    if SHOW_PLOT:
        plt.show()

if __name__ == "__main__":
    main()
