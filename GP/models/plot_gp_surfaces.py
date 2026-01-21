#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import sys
from pathlib import Path

# Add main_folder to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from gp_dynamics import GPManager  # your GPManager with .load() and .dataset()


# ============================================================
# USER SETTINGS (edit here)
# ============================================================

# Pick which GP to visualize (example: output dim 3 = d(theta_dot)/dt if you trained dstate_dt)
GP_PATH = "models/gp_invpend_out1_dstate_dt.pt"

# Choose which 2 state dimensions to plot on the grid:
# state indices: 0=x, 1=theta, 2=x_dot, 3=theta_dot
GRID_DIMS = (1, 3)  # (theta, theta_dot) is usually the most informative for balancing

# Actions to show (top row) and the action for uncertainty plot (bottom)
A_VALUES = [-3.0, 0.0, 3.0]
A_UNCERT = 0.0

# Training-data overlay tolerances
ACTION_TOL = 0.25
OTHER_STATE_TOL = {
    0: 0.20,   # x tolerance
    2: 0.20,   # x_dot tolerance
    # you can also add tolerances for dims not in GRID_DIMS if you want stricter filtering
}

# Fixed values for non-plotted state dims.
# If None: uses the median of the training data for that dim (recommended).
FIXED_STATE = None

# Grid resolution
N_GRID = 80

# Label for Z axis
TITLE_ZLABEL = "GP output (e.g., d(theta_dot)/dt)"


# ============================================================
# Helpers
# ============================================================

STATE_NAMES = ["x", "theta", "x_dot", "theta_dot"]

def build_fixed_state_from_data(X_train_state: np.ndarray, fixed_state: dict | None):
    """
    X_train_state: (N,4) = [x, theta, x_dot, theta_dot]
    Returns a dict {dim_index: value} for dims 0..3.
    """
    if fixed_state is not None:
        # fill missing dims with median
        out = {}
        for d in range(4):
            if d in fixed_state:
                out[d] = float(fixed_state[d])
            else:
                out[d] = float(np.median(X_train_state[:, d]))
        return out

    # default: median for all dims
    return {d: float(np.median(X_train_state[:, d])) for d in range(4)}

def make_query_grid(P, R, a_fixed, grid_dims, fixed_vals):
    """
    Build X_grid for GP prediction.
    GP input is 5D: [x, theta, x_dot, theta_dot, u]
    We vary the two dims in grid_dims using (P,R) and keep the other state dims fixed.
    """
    d0, d1 = grid_dims

    Xg = np.zeros((P.size, 5), dtype=np.float32)

    # fill state dims 0..3 with fixed values
    for d in range(4):
        Xg[:, d] = fixed_vals[d]

    # overwrite the two grid dims
    Xg[:, d0] = P.ravel().astype(np.float32)
    Xg[:, d1] = R.ravel().astype(np.float32)

    # action (u)
    Xg[:, 4] = np.float32(a_fixed)
    return Xg

def overlay_mask(X_train, grid_dims, fixed_vals, a_fixed, action_tol, other_state_tol):
    """
    Select training points near:
      - action ≈ a_fixed
      - and (optionally) other fixed dims ≈ fixed_vals[d]
    We do NOT filter on the grid dims (we want full spread there).
    """
    # X_train: (N,5)
    mask = np.abs(X_train[:, 4] - a_fixed) < action_tol

    # filter on non-grid state dims (optional)
    d0, d1 = grid_dims
    for d, tol in other_state_tol.items():
        if d in (d0, d1):
            continue
        mask &= (np.abs(X_train[:, d] - fixed_vals[d]) < float(tol))

    return mask


# ============================================================
# Main plotting: 3 mean surfaces + uncertainty surface
# ============================================================

def plot_gp_surfaces_with_uncertainty_invpend(
    gp,
    a_values,
    a_uncert=0.0,
    grid_dims=(1, 3),
    fixed_state=None,
    action_tolerance=0.25,
    other_state_tolerance=None,
    title_zlabel="GP output",
    n_grid=80,
):
    if other_state_tolerance is None:
        other_state_tolerance = {}

    # ---------- 1) training data ----------
    X_train, Y_train = gp.dataset()   # X: (N,5), Y: (N,)
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32).reshape(-1)

    if X_train.shape[1] != 5:
        raise ValueError(f"Expected X_train dim 5 ([x,theta,x_dot,theta_dot,u]). Got {X_train.shape}")

    X_state = X_train[:, :4]  # (N,4)
    u_train = X_train[:, 4]

    d0, d1 = grid_dims
    if not (0 <= d0 <= 3 and 0 <= d1 <= 3 and d0 != d1):
        raise ValueError(f"grid_dims must be two distinct state indices in 0..3. Got {grid_dims}")

    fixed_vals = build_fixed_state_from_data(X_state, fixed_state)

    # grid ranges based on training data for the chosen dims
    p_min, p_max = float(X_state[:, d0].min()), float(X_state[:, d0].max())
    r_min, r_max = float(X_state[:, d1].min()), float(X_state[:, d1].max())

    p_grid = np.linspace(p_min, p_max, n_grid)
    r_grid = np.linspace(r_min, r_max, n_grid)
    P, R = np.meshgrid(p_grid, r_grid)

    # ---------- 2) figure layout ----------
    fig = plt.figure(figsize=(16, 10), dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 0.06],
        height_ratios=[1, 1],
        wspace=0.008,
        hspace=0.02,
        left=0.02,
        right=0.92,
        bottom=0.06,
        top=0.95,
    )

    ax_top = [fig.add_subplot(gs[0, i], projection="3d") for i in range(3)]
    cax_mean = fig.add_subplot(gs[0, 3])

    ax_uncert = fig.add_subplot(gs[1, 0:3], projection="3d")
    cax_std   = fig.add_subplot(gs[1, 3])

    label_fs = 18 * 0.6
    title_fs = 24 * 0.6
    tick_fs  = 10 * 0.6

    # ---------- 3) top row: mean surfaces ----------
    z_min, z_max = np.inf, -np.inf
    mean_surfaces = []

    for a_fixed in a_values:
        X_grid = make_query_grid(P, R, a_fixed, grid_dims, fixed_vals)
        Mean_t, _ = gp.predict_torch(X_grid)
        Mean_np = Mean_t.detach().cpu().numpy().reshape(P.shape)

        mean_surfaces.append((a_fixed, Mean_np))
        z_min = min(z_min, float(Mean_np.min()))
        z_max = max(z_max, float(Mean_np.max()))

    surfaces = []
    for ax, (a_fixed, Mean_np) in zip(ax_top, mean_surfaces):
        surf = ax.plot_surface(
            P, R, Mean_np,
            cmap="coolwarm",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
            vmin=z_min,
            vmax=z_max,
        )
        surfaces.append(surf)

        mask = overlay_mask(
            X_train,
            grid_dims=grid_dims,
            fixed_vals=fixed_vals,
            a_fixed=a_fixed,
            action_tol=action_tolerance,
            other_state_tol=other_state_tolerance,
        )
        print(f"[TOP] u={a_fixed: .2f} -> overlay points: {int(np.sum(mask))}")

        ax.scatter(
            X_state[mask, d0], X_state[mask, d1], Y_train[mask],
            color="k", s=20, alpha=0.8, label="data"
        )

        ax.set_xlabel(f"{STATE_NAMES[d0]}", fontsize=label_fs)
        ax.set_ylabel(f"{STATE_NAMES[d1]}", fontsize=label_fs)
        ax.set_zlabel(title_zlabel, fontsize=label_fs)
        ax.set_title(f"u = {a_fixed:.2f}", fontsize=title_fs)
        ax.set_zlim(z_min, z_max)

        ax.legend(fontsize=10)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.tick_params(axis="z", labelsize=tick_fs)

    sm_mean = ScalarMappable(cmap=surfaces[0].cmap, norm=surfaces[0].norm)
    sm_mean.set_array([])
    cb_mean = fig.colorbar(sm_mean, cax=cax_mean)
    cb_mean.set_label("GP mean", fontsize=label_fs)
    cb_mean.ax.tick_params(labelsize=tick_fs)

    # ---------- 4) bottom: mean height + std color ----------
    print(f"[BOTTOM] u={a_uncert: .2f} uncertainty surface")
    X_grid_unc = make_query_grid(P, R, a_uncert, grid_dims, fixed_vals)

    Mean_t, Var_t = gp.predict_torch(X_grid_unc)
    Mean_unc = Mean_t.detach().cpu().numpy().reshape(P.shape)
    Var_unc  = Var_t.detach().cpu().numpy().reshape(P.shape)
    Std_unc  = np.sqrt(np.maximum(Var_unc, 0.0))

    norm_std = plt.Normalize(vmin=float(Std_unc.min()), vmax=float(Std_unc.max()))
    colors_std = plt.cm.viridis(norm_std(Std_unc))

    ax_uncert.plot_surface(
        P, R, Mean_unc,
        facecolors=colors_std,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    mask_unc = overlay_mask(
        X_train,
        grid_dims=grid_dims,
        fixed_vals=fixed_vals,
        a_fixed=a_uncert,
        action_tol=action_tolerance,
        other_state_tol=other_state_tolerance,
    )
    print(f"[BOTTOM] overlay points: {int(np.sum(mask_unc))}")

    ax_uncert.scatter(
        X_state[mask_unc, d0], X_state[mask_unc, d1], Y_train[mask_unc],
        color="k", s=20, alpha=0.7, label=f"data (u≈{a_uncert})"
    )

    ax_uncert.set_xlabel(f"{STATE_NAMES[d0]}", fontsize=label_fs)
    ax_uncert.set_ylabel(f"{STATE_NAMES[d1]}", fontsize=label_fs)
    ax_uncert.set_zlabel(title_zlabel, fontsize=label_fs)
    ax_uncert.set_title(f"u = {a_uncert:.2f} — Mean (height), Std (color)", fontsize=title_fs)
    ax_uncert.tick_params(axis="both", labelsize=tick_fs)
    ax_uncert.tick_params(axis="z", labelsize=tick_fs)
    ax_uncert.legend(fontsize=10)

    sm_std = ScalarMappable(cmap="viridis", norm=norm_std)
    sm_std.set_array([])
    cb_std = fig.colorbar(sm_std, cax=cax_std)
    cb_std.set_label("GP predictive std", fontsize=label_fs)
    cb_std.ax.tick_params(labelsize=tick_fs)

    # Print the fixed values used (so you know what slice you plotted)
    fixed_info = ", ".join([f"{STATE_NAMES[d]}={fixed_vals[d]:+.3f}" for d in range(4) if d not in grid_dims])
    fig.suptitle(f"Fixed dims: {fixed_info}", fontsize=title_fs)

    plt.show()


# ============================================================
# Optional 1D slice: vary one dim, fix the rest
# ============================================================

def plot_1d_gp_slice_with_uncertainty_invpend(
    gp,
    vary_dim=1,            # default theta
    fixed_state=None,      # dict for dims 0..3
    action_fixed=0.0,
    n_points=250,
    action_tolerance=0.25,
    other_state_tolerance=None,
    title_prefix="GP slice",
):
    if other_state_tolerance is None:
        other_state_tolerance = {}

    X_train, Y_train = gp.dataset()
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32).reshape(-1)

    X_state = X_train[:, :4]
    fixed_vals = build_fixed_state_from_data(X_state, fixed_state)

    if not (0 <= vary_dim <= 3):
        raise ValueError("vary_dim must be 0..3")

    vmin, vmax = float(X_state[:, vary_dim].min()), float(X_state[:, vary_dim].max())
    v_grid = np.linspace(vmin, vmax, n_points)

    # Build query: [x, theta, x_dot, theta_dot, u]
    Xq = np.zeros((n_points, 5), dtype=np.float32)
    for d in range(4):
        Xq[:, d] = fixed_vals[d]
    Xq[:, vary_dim] = v_grid.astype(np.float32)
    Xq[:, 4] = np.float32(action_fixed)

    Mean_t, Var_t = gp.predict_torch(Xq)
    Mean = Mean_t.detach().cpu().numpy().ravel()
    Std  = np.sqrt(np.maximum(Var_t.detach().cpu().numpy().ravel(), 0.0))

    plt.figure(figsize=(10, 6))
    plt.plot(v_grid, Mean, lw=2, label="GP mean")
    plt.fill_between(v_grid, Mean - 2.0 * Std, Mean + 2.0 * Std, alpha=0.25, label="±2σ")

    # overlay training points near the slice (filter on action + non-vary dims)
    mask = (np.abs(X_train[:, 4] - action_fixed) < action_tolerance)
    for d, tol in other_state_tolerance.items():
        if d == vary_dim:
            continue
        mask &= (np.abs(X_train[:, d] - fixed_vals[d]) < float(tol))

    plt.scatter(X_state[mask, vary_dim], Y_train[mask], color="k", s=35, alpha=0.7, label="data near slice")

    fixed_info = ", ".join([f"{STATE_NAMES[d]}={fixed_vals[d]:+.3f}" for d in range(4) if d != vary_dim])
    plt.title(f"{title_prefix}: vary {STATE_NAMES[vary_dim]} (u={action_fixed:+.2f})\nfixed: {fixed_info}")
    plt.xlabel(STATE_NAMES[vary_dim])
    plt.ylabel("GP output")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# main
# ============================================================

def main():
    gp = GPManager.load(GP_PATH)

    plot_gp_surfaces_with_uncertainty_invpend(
        gp,
        a_values=A_VALUES,
        a_uncert=A_UNCERT,
        grid_dims=GRID_DIMS,
        fixed_state=FIXED_STATE,
        action_tolerance=ACTION_TOL,
        other_state_tolerance=OTHER_STATE_TOL,
        title_zlabel=TITLE_ZLABEL,
        n_grid=N_GRID,
    )

    # Optional: 1D slice example
    # plot_1d_gp_slice_with_uncertainty_invpend(
    #     gp,
    #     vary_dim=1,  # theta
    #     fixed_state=None,
    #     action_fixed=0.0,
    #     n_points=250,
    #     action_tolerance=0.25,
    #     other_state_tolerance={0: 0.2, 2: 0.5, 3: 1.0},
    #     title_prefix="GP slice",
    # )

if __name__ == "__main__":
    main()
