#!/usr/bin/env python3
import time
import math
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


def set_bottom(env):
    """Start from bottom: theta = pi, theta_dot = 0."""
    if hasattr(env.unwrapped, "state"):
        env.unwrapped.state = np.array([math.pi, 0.0], dtype=np.float64)


def obs_to_theta(obs):
    """obs = [cos(theta), sin(theta), theta_dot]."""
    c, s, thdot = float(obs[0]), float(obs[1]), float(obs[2])
    theta = math.atan2(s, c)  # in (-pi, pi]
    return theta, thdot


def main():
    # -------- parameters (keep it simple) --------
    dt = 0.05          # you want 0.1s sampling
    duration = 20.0   # seconds
    save_path = "pendulum_random_dt0p1.npz"

    env = gym.make("Pendulum-v1", render_mode="human")
    obs, _ = env.reset(seed=0)
    set_bottom(env)              # start from bottom
    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else obs

    # internal env dt (Pendulum-v1 is usually 0.05)
    env_dt = float(getattr(env.unwrapped, "dt", 0.05))
    steps_per_action = max(1, int(round(dt / env_dt)))
    dt_eff = steps_per_action * env_dt
    print(f"env_dt={env_dt:.3f}s, steps_per_action={steps_per_action}, dt_eff={dt_eff:.3f}s")

    # action bounds
    umin = float(env.action_space.low[0])
    umax = float(env.action_space.high[0])

    # -------- live plot setup --------
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    (line_th,) = ax1.plot([], [], lw=2)
    (line_thd,) = ax2.plot([], [], lw=2)
    (line_u,) = ax3.plot([], [], lw=2)

    ax1.set_ylabel("theta [rad]")
    ax2.set_ylabel("theta_dot [rad/s]")
    ax3.set_ylabel("u [torque]")
    ax3.set_xlabel("time [s]")
    for ax in (ax1, ax2, ax3):
        ax.grid(True)

    # -------- logs --------
    t_log, th_log, thd_log, u_log = [], [], [], []

    t = 0.0
    start_wall = time.perf_counter()
    last_plot_wall = 0.0

    try:
        while t < duration:
            # state at sampling instant
            theta, theta_dot = obs_to_theta(obs)

            # random action, held constant for dt_eff
            u = float(np.random.uniform(umin, umax))
            action = np.array([u], dtype=np.float32)

            # log (input and outputs)
            t_log.append(t)
            th_log.append(theta)
            thd_log.append(theta_dot)
            u_log.append(u)

            # step the env for steps_per_action substeps
            for _ in range(steps_per_action):
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
                    set_bottom(env)
                    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else obs
                    break
                time.sleep(env_dt)  # makes rendering watchable

            t += dt_eff

            # update plot ~10 Hz (lightweight)
            now = time.perf_counter()
            if now - last_plot_wall > 0.1:
                line_th.set_data(t_log, th_log)
                line_thd.set_data(t_log, thd_log)
                line_u.set_data(t_log, u_log)

                ax1.set_xlim(max(0.0, t - 10.0), max(10.0, t))  # sliding window
                ax1.set_ylim(-math.pi, math.pi)
                ax2.relim(); ax2.autoscale_view()
                ax3.set_ylim(umin * 1.1, umax * 1.1)

                fig.canvas.draw()
                fig.canvas.flush_events()
                last_plot_wall = now

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plt.ioff()

    # -------- save --------
    if len(t_log) > 1:
        np.savez(
            save_path,
            dt=np.float32(dt_eff),
            t=np.asarray(t_log, dtype=np.float32),
            theta=np.asarray(th_log, dtype=np.float32),
            theta_dot=np.asarray(thd_log, dtype=np.float32),
            u=np.asarray(u_log, dtype=np.float32),
        )
        print(f"Saved {save_path} with N={len(t_log)} samples, dt={dt_eff:.3f}s")
    else:
        print("No data collected; nothing saved.")

    plt.show()


if __name__ == "__main__":
    main()
