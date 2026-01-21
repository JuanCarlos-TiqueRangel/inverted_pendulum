import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

import os
import time
import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
from PIL import Image

import gymnasium as gym

# GP / TF stack (same as your MountainCar)
import tensorflow as tf
import gpflow
from gpflow.inducing_variables import InducingPoints


gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)
tf.keras.backend.set_floatx("float64")

print("TF version:", tf.__version__)
print("GPflow version:", gpflow.__version__)

# ----------------------------
# Pendulum constants
# ----------------------------
ENV_NAME = "Pendulum-v1"
U_MIN, U_MAX = -2.0, 2.0

# For angle wrapping
def wrap_pi(theta):
    """Wrap angle to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def obs_to_theta_omega(obs):
    """
    Pendulum obs: [cos(theta), sin(theta), omega]
    Return:
        theta in (-pi, pi]
        omega (rad/s)
    """
    c, s, w = float(obs[0]), float(obs[1]), float(obs[2])
    theta = math.atan2(s, c)
    theta = wrap_pi(theta)
    return theta, w

def theta_omega_to_features(theta, omega, u, omega_scale=8.0):
    """
    GP input features (your style):
      X = [sin(theta), cos(theta), omega_feat, u]
      omega_feat = tanh(omega / omega_scale)
    """
    omega_feat = np.tanh(float(omega) / float(omega_scale))
    return np.array([np.sin(theta), np.cos(theta), omega_feat, float(u)], dtype=np.float64)

# ----------------------------
# Env builder
# ----------------------------
def make_env(render_mode=None, seed=0):
    """
    render_mode:
      None -> fastest
      "rgb_array" -> for recording/animation
      "human" -> live window
    """
    env = gym.make(ENV_NAME, render_mode=render_mode)
    env.reset(seed=seed)
    return env

# Quick sanity check
env_test = make_env(render_mode=None, seed=0)
obs, info = env_test.reset()
theta0, omega0 = obs_to_theta_omega(obs)
print("Sanity check:")
print("  obs:", obs)
print("  theta0:", theta0, "omega0:", omega0)
env_test.close()