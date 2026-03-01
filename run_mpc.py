#!/usr/bin/env python3
"""
Run MPC-based drone circle trajectory tracking with MuJoCo visualization.

Usage:
    python run_mpc.py [--radius 1.0] [--height 1.0] [--omega 0.5]
                      [--duration 20] [--render] [--save]
"""

import argparse
import time
import numpy as np

from drone_mpc.drone_env import DroneEnv
from drone_mpc.mpc_controller import MPCController
from drone_mpc.trajectory import CircleTrajectory, LemniscateTrajectory
from drone_mpc.visualization import plot_tracking_results


def main():
    parser = argparse.ArgumentParser(description="MPC Drone Circle Tracking")
    parser.add_argument("--radius", type=float, default=1.0, help="Circle radius (m)")
    parser.add_argument("--height", type=float, default=1.0, help="Flight height (m)")
    parser.add_argument("--omega", type=float, default=0.5, help="Angular speed (rad/s)")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration (s)")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    parser.add_argument("--trajectory", type=str, default="circle",
                        choices=["circle", "lemniscate"], help="Trajectory type")
    parser.add_argument("--horizon", type=int, default=25, help="MPC horizon")
    parser.add_argument("--dt-ctrl", type=float, default=0.02, help="Control timestep (s)")
    parser.add_argument("--dt-sim", type=float, default=0.005, help="Simulation timestep (s)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MPC Drone Trajectory Tracking")
    print("=" * 60)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Radius: {args.radius} m | Height: {args.height} m")
    print(f"  Angular speed: {args.omega} rad/s")
    print(f"  Duration: {args.duration} s")
    print(f"  MPC horizon: {args.horizon} | Control dt: {args.dt_ctrl} s")
    print("=" * 60)

    # Create environment
    env = DroneEnv(dt=args.dt_sim, render=args.render)

    # Create trajectory generator
    if args.trajectory == "circle":
        traj = CircleTrajectory(
            radius=args.radius, omega=args.omega,
            center=(0.0, 0.0), height=args.height,
        )
    else:
        traj = LemniscateTrajectory(
            scale=args.radius, omega=args.omega,
            center=(0.0, 0.0), height=args.height,
        )

    # Create MPC controller
    controller = MPCController(
        dt=args.dt_ctrl,
        horizon=args.horizon,
        mass=0.027,
        gravity=9.81,
    )

    # Initialize at start of trajectory
    ref0 = traj.get_reference(0.0)
    state = env.reset(pos=ref0[:3])

    if args.render:
        env.launch_viewer()

    # Simulation loop
    n_ctrl_steps = int(args.duration / args.dt_ctrl)
    sim_steps_per_ctrl = max(1, int(args.dt_ctrl / args.dt_sim))

    # Logging
    log_times = []
    log_pos = []
    log_ref = []
    log_ctrl = []
    log_compute_time = []

    print("\nRunning simulation...")
    t_sim = 0.0

    for step in range(n_ctrl_steps):
        t_sim = step * args.dt_ctrl

        # Get reference trajectory for MPC horizon
        ref_seq = traj.get_reference_sequence(t_sim, controller.N, controller.dt)

        # Compute control
        t_start = time.time()
        ctrl = controller.compute_control(state, ref_seq)
        t_compute = time.time() - t_start

        # Step simulation
        for _ in range(sim_steps_per_ctrl):
            state = env.step(ctrl)

        # Log
        ref_now = traj.get_reference(t_sim)
        pos = state[:3]
        error = np.linalg.norm(pos - ref_now[:3])

        log_times.append(t_sim)
        log_pos.append(pos.copy())
        log_ref.append(ref_now[:3].copy())
        log_ctrl.append(ctrl.copy())
        log_compute_time.append(t_compute)

        # Print progress
        if step % 50 == 0:
            print(f"  t={t_sim:6.2f}s | pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) | "
                  f"error={error:.4f}m | compute={t_compute*1000:.1f}ms")

    # Convert logs
    times = np.array(log_times)
    actual_pos = np.array(log_pos)
    ref_pos = np.array(log_ref)
    controls = np.array(log_ctrl)
    compute_times = np.array(log_compute_time)

    # Statistics
    errors = np.linalg.norm(actual_pos - ref_pos, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)
    avg_compute = np.mean(compute_times) * 1000

    print("\n" + "=" * 60)
    print("  Results:")
    print(f"    RMSE:         {rmse:.4f} m")
    print(f"    Max error:    {max_error:.4f} m")
    print(f"    Avg compute:  {avg_compute:.1f} ms")
    print("=" * 60)

    # Plot
    plot_tracking_results(
        times, actual_pos, ref_pos, controls,
        title_prefix="MPC",
        save_path=args.save,
    )

    env.close()

    return {
        "times": times,
        "actual_pos": actual_pos,
        "ref_pos": ref_pos,
        "controls": controls,
        "compute_times": compute_times,
    }


if __name__ == "__main__":
    main()
