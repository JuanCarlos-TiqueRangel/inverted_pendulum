import time
import math
import warnings
import numpy as np
import gymnasium as gym

# Optional: silence that pygame/pkg_resources warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

def pd_upright_pendulum(obs, t):
    """
    Pendulum-v1:
      obs = [cos(theta), sin(theta), theta_dot]
      action = [torque] in [-2, 2] (typically)

    Goal: stabilize upright (theta ~ 0).
    """
    c, s, theta_dot = float(obs[0]), float(obs[1]), float(obs[2])
    theta = math.atan2(s, c)  # (-pi, pi]

    # PD gains (tune as needed)
    ktheta = 8.0
    ktheta_dot = 1.5

    # PD to upright: torque = -Kp*theta - Kd*theta_dot
    u = -(ktheta * theta + ktheta_dot * theta_dot)

    # Small periodic torque (optional "motion"/excitation)
    u += 0.3 * math.sin(0.8 * t)

    return np.array([u], dtype=np.float32)

def main():
    env = gym.make("Pendulum-v1", render_mode="human")
    obs, info = env.reset(seed=0)

    dt = float(getattr(env.unwrapped, "dt", 0.05))  # Pendulum-v1 is commonly 0.05
    t = 0.0

    try:
        for step in range(10_000):
            action = pd_upright_pendulum(obs, t)

            # Clip to env bounds (torque limits)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            t += dt

            if terminated or truncated:
                obs, info = env.reset()

            time.sleep(dt)
    finally:
        env.close()

if __name__ == "__main__":
    main()
