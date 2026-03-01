"""
Visualization utilities for drone trajectory tracking.

Provides 2D/3D plotting of:
- Actual vs reference trajectories
- Tracking error over time
- Control inputs over time
- Cost / performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional


def plot_trajectory_3d(
    actual: np.ndarray,
    reference: np.ndarray,
    title: str = "Drone Trajectory Tracking (3D)",
    save_path: Optional[str] = None,
):
    """
    Plot actual vs reference trajectories in 3D.

    Args:
        actual: (T, 3) actual positions [x, y, z]
        reference: (T, 3) reference positions [x, y, z]
        title: Plot title
        save_path: If given, save plot to this path
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(reference[:, 0], reference[:, 1], reference[:, 2],
            "r--", linewidth=2, label="Reference", alpha=0.7)
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2],
            "b-", linewidth=1.5, label="Actual")

    # Mark start and end
    ax.scatter(*actual[0, :3], color="green", s=100, marker="^", label="Start")
    ax.scatter(*actual[-1, :3], color="red", s=100, marker="v", label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()

    # Equal aspect ratio
    max_range = np.max([
        actual[:, 0].ptp(), actual[:, 1].ptp(), actual[:, 2].ptp(),
        reference[:, 0].ptp(), reference[:, 1].ptp(), reference[:, 2].ptp(),
    ]) / 2
    mid = np.mean(np.vstack([actual, reference]), axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(max(0, mid[2] - max_range), mid[2] + max_range)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory_2d(
    actual: np.ndarray,
    reference: np.ndarray,
    title: str = "Drone Trajectory (Top View)",
    save_path: Optional[str] = None,
):
    """
    Plot actual vs reference trajectories in 2D (top-down view).
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(reference[:, 0], reference[:, 1],
            "r--", linewidth=2, label="Reference", alpha=0.7)
    ax.plot(actual[:, 0], actual[:, 1],
            "b-", linewidth=1.5, label="Actual")

    ax.scatter(actual[0, 0], actual[0, 1], color="green", s=100, marker="^",
               label="Start", zorder=5)
    ax.scatter(actual[-1, 0], actual[-1, 1], color="red", s=100, marker="v",
               label="End", zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_tracking_results(
    times: np.ndarray,
    actual_pos: np.ndarray,
    ref_pos: np.ndarray,
    controls: np.ndarray,
    title_prefix: str = "",
    save_path: Optional[str] = None,
):
    """
    Comprehensive plot of tracking results.

    Creates a 4-panel figure:
    1. 3D trajectory
    2. Position tracking (x, y, z vs time)
    3. Tracking error vs time
    4. Control inputs vs time

    Args:
        times: (T,) time array
        actual_pos: (T, 3) actual positions
        ref_pos: (T, 3) reference positions
        controls: (T, 4) control inputs
        title_prefix: Prefix for plot titles (e.g., "MPC" or "MPPI")
        save_path: Optional save path
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)

    # --- Panel 1: 3D Trajectory ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
             "r--", linewidth=2, label="Reference", alpha=0.7)
    ax1.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2],
             "b-", linewidth=1.5, label="Actual")
    ax1.scatter(*actual_pos[0], color="green", s=80, marker="^", label="Start")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"{title_prefix} 3D Trajectory")
    ax1.legend(fontsize=8)

    # --- Panel 2: Position vs Time ---
    ax2 = fig.add_subplot(gs[0, 1])
    labels = ["X", "Y", "Z"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    for i in range(3):
        ax2.plot(times, actual_pos[:, i], color=colors[i],
                 linewidth=1.5, label=f"{labels[i]} actual")
        ax2.plot(times, ref_pos[:, i], "--", color=colors[i],
                 linewidth=1, alpha=0.7, label=f"{labels[i]} ref")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.set_title(f"{title_prefix} Position Tracking")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Tracking Error ---
    ax3 = fig.add_subplot(gs[1, 0])
    error = np.linalg.norm(actual_pos - ref_pos, axis=1)
    error_xyz = actual_pos - ref_pos
    ax3.plot(times, error, "k-", linewidth=1.5, label="||e||")
    for i in range(3):
        ax3.plot(times, np.abs(error_xyz[:, i]),
                 color=colors[i], linewidth=1, alpha=0.7,
                 label=f"|e_{labels[i]}|")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Error (m)")
    ax3.set_title(f"{title_prefix} Tracking Error (RMSE={np.sqrt(np.mean(error**2)):.4f} m)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Controls ---
    ax4 = fig.add_subplot(gs[1, 1])
    ctrl_labels = ["Thrust", "Roll Cmd", "Pitch Cmd", "Yaw Cmd"]
    for i in range(min(4, controls.shape[1])):
        ax4.plot(times[:len(controls)], controls[:, i],
                 linewidth=1, label=ctrl_labels[i])
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Control")
    ax4.set_title(f"{title_prefix} Control Inputs")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f"{title_prefix} Drone Trajectory Tracking Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_controllers(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    Compare MPC and MPPI tracking results side by side.

    Args:
        results: Dict with keys like "MPC", "MPPI", each containing:
            - "times": time array
            - "actual_pos": actual positions
            - "ref_pos": reference positions
            - "controls": control inputs
            - "compute_times": per-step computation times
        save_path: Optional save path
    """
    n_methods = len(results)
    fig = plt.figure(figsize=(8 * n_methods, 10))
    gs = GridSpec(3, n_methods, figure=fig)

    colors_map = {"MPC": "tab:blue", "MPPI": "tab:orange"}

    for col, (name, data) in enumerate(results.items()):
        times = data["times"]
        actual_pos = data["actual_pos"]
        ref_pos = data["ref_pos"]
        controls = data["controls"]
        color = colors_map.get(name, f"C{col}")

        # 3D Trajectory
        ax1 = fig.add_subplot(gs[0, col], projection="3d")
        ax1.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
                 "r--", linewidth=2, alpha=0.5)
        ax1.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2],
                 color=color, linewidth=1.5)
        ax1.set_title(f"{name}: 3D Trajectory")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # Tracking error
        ax2 = fig.add_subplot(gs[1, col])
        error = np.linalg.norm(actual_pos - ref_pos, axis=1)
        ax2.plot(times, error, color=color, linewidth=1)
        rmse = np.sqrt(np.mean(error ** 2))
        ax2.set_title(f"{name}: Error (RMSE={rmse:.4f} m)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Error (m)")
        ax2.grid(True, alpha=0.3)

        # Computation time
        if "compute_times" in data:
            ax3 = fig.add_subplot(gs[2, col])
            ct = data["compute_times"]
            ax3.plot(times[:len(ct)], ct, color=color, linewidth=0.5)
            avg_ct = np.mean(ct) * 1000
            ax3.set_title(f"{name}: Compute Time (avg={avg_ct:.1f} ms)")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Time (s)")
            ax3.grid(True, alpha=0.3)

    fig.suptitle("Controller Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
