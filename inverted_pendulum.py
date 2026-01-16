import time
import numpy as np
import gymnasium as gym

def pd_balance_with_motion(obs, t):
    """
    obs = [x, theta, x_dot, theta_dot]
      x         : cart position (m)
      theta     : pole angle (rad)   (upright ~ 0)
      x_dot     : cart velocity (m/s)
      theta_dot : pole angular vel (rad/s)
    action = [force] in [-3, 3]
    """
    x, theta, x_dot, theta_dot = obs

    # Hand-tuned stabilizing gains (adjust if needed)
    kx = 0.0
    kx_dot = 1.0
    ktheta = 1.0
    ktheta_dot = 1.0

    # Stabilize upright
    u = -(kx * x + kx_dot * x_dot + ktheta * theta + ktheta_dot * theta_dot)

    # Add a small periodic "drive" so the cart moves while balancing
    u += 0.6 * np.sin(0.8 * t)

    return np.array([u], dtype=np.float32)

def main():
    # render_mode="human" shows the MuJoCo viewer window
    env = gym.make("Pendulum-v1", render_mode="human")

    obs, info = env.reset(seed=0)

    # Try to use the env's step time if available; otherwise default.
    dt = getattr(env.unwrapped, "dt", 0.02)
    t = 0.0

    try:
        for step in range(10_000):
            action = pd_balance_with_motion(obs, t)

            # Respect action bounds (force in [-3, 3]) :contentReference[oaicite:1]{index=1}
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            t += dt

            if terminated or truncated:
                obs, info = env.reset()

            # Slow down to real-time-ish so you can watch it
            time.sleep(dt)

    finally:
        env.close()

if __name__ == "__main__":
    main()
