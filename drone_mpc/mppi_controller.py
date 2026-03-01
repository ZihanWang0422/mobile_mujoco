"""
Model Predictive Path Integral (MPPI) Controller for quadrotor trajectory tracking.

MPPI is a sampling-based MPC algorithm that:
1. Samples many random control perturbations around the current control sequence
2. Rolls out the dynamics for each sample
3. Evaluates the cost of each trajectory
4. Computes a weighted average of the samples based on exponentiated negative costs

Key advantages over standard MPC:
- No need for gradient computation
- Handles non-convex cost functions naturally
- Embarrassingly parallel (GPU-friendly)
- Can handle complex dynamics without linearization

Reference:
    Williams et al., "Information Theoretic MPC for Model-Based Reinforcement
    Learning", ICRA 2017.
"""

import numpy as np
from typing import Optional, Dict, Any


class MPPIController:
    """
    MPPI (Model Predictive Path Integral) controller for Crazyflie 2 drone.

    Algorithm:
        1. Sample K perturbation sequences: δu_k ~ N(0, Σ)
        2. Form candidate controls: U_k = U_nominal + δu_k
        3. Rollout dynamics for each sample
        4. Compute cost S_k for each trajectory
        5. Compute weights: w_k = exp(-S_k / λ) / Σ exp(-S_k / λ)
        6. Update nominal: U_nominal += Σ w_k * δu_k
        7. Apply first control, shift sequence

    State: [x, y, z, vx, vy, vz, roll, pitch, yaw] (9D)
    Control: [thrust, roll_cmd, pitch_cmd, yaw_cmd] (4D)
    """

    def __init__(
        self,
        dt: float = 0.02,
        horizon: int = 30,
        n_samples: int = 256,
        temperature: float = 0.05,
        noise_sigma: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Q_terminal: Optional[np.ndarray] = None,
        mass: float = 0.027,
        gravity: float = 9.81,
        max_thrust: float = 0.35,
        max_moment: float = 1.0,
        smoothing_alpha: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            dt: Control timestep
            horizon: Prediction horizon
            n_samples: Number of control samples (K)
            temperature: MPPI temperature parameter (λ). Lower = more greedy.
            noise_sigma: Noise std for each control dim (4,). Controls exploration.
            Q: State cost matrix (9x9)
            R: Control cost matrix (4x4)
            Q_terminal: Terminal cost matrix (9x9)
            mass: Drone mass
            gravity: Gravity
            max_thrust: Maximum thrust (N)
            max_moment: Maximum moment magnitude
            smoothing_alpha: Savitzky-Golay smoothing factor for control sequence
            seed: Random seed for reproducibility
        """
        self.dt = dt
        self.N = horizon
        self.K = n_samples
        self.lam = temperature
        self.mass = mass
        self.gravity = gravity
        self.n_state = 9
        self.n_ctrl = 4
        self.smoothing_alpha = smoothing_alpha

        # Exploration noise standard deviations
        if noise_sigma is None:
            noise_sigma = np.array([0.03, 0.15, 0.15, 0.1])
        self.noise_sigma = noise_sigma

        # Cost matrices
        if Q is None:
            Q = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 10.0, 10.0, 5.0])
        if R is None:
            R = np.diag([1.0, 0.1, 0.1, 0.1])
        if Q_terminal is None:
            Q_terminal = Q * 3.0

        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal

        # Control: [thrust_N, roll_rad, pitch_rad, yaw_rate_rads]
        max_angle    = np.deg2rad(30.0)
        max_yaw_rate = np.deg2rad(180.0)
        self.u_min   = np.array([0.0,        -max_angle,  -max_angle,  -max_yaw_rate])
        self.u_max   = np.array([max_thrust,  max_angle,   max_angle,   max_yaw_rate])
        self.u_hover = np.array([mass * gravity, 0.0, 0.0, 0.0])

        # Attitude bandwidth time constants
        self.tau_roll  = 0.05
        self.tau_pitch = 0.05

        # Nominal control sequence: (N, 4), initialized to hover
        self.U_nominal = np.tile(self.u_hover, (self.N, 1))

        # Random state
        self.rng = np.random.default_rng(seed)

        # Storage for diagnostics
        self._best_cost = np.inf
        self._mean_cost = np.inf
        self._best_trajectory = None

    def _dynamics_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Vectorized RK4 step for simplified quadrotor dynamics.

        Args:
            x: State array, shape (K, 9) or (9,)
            u: Control array, shape (K, 4) or (4,)

        Returns:
            x_next: Next state, same shape as x
        """
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * self.dt * k1, u)
        k3 = self._dynamics(x + 0.5 * self.dt * k2, u)
        k4 = self._dynamics(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Simplified quadrotor dynamics (vectorized for batch processing).

        Args:
            x: (K, 9) or (9,) state
            u: (K, 4) or (4,) control

        Returns:
            dx: State derivative, same shape
        """
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]
            u = u[np.newaxis, :]

        # Unpack
        vx, vy, vz = x[:, 3], x[:, 4], x[:, 5]
        roll, pitch, yaw = x[:, 6], x[:, 7], x[:, 8]

        thrust    = u[:, 0]   # [N]
        roll_cmd  = u[:, 1]   # [rad]
        pitch_cmd = u[:, 2]   # [rad]
        yaw_rate  = u[:, 3]   # [rad/s]

        # Acceleration from thrust (body-frame to world-frame)
        ax = (thrust / self.mass) * (
            np.cos(yaw) * np.sin(pitch) + np.sin(yaw) * np.sin(roll)
        )
        ay = (thrust / self.mass) * (
            np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.sin(roll)
        )
        az = (thrust / self.mass) * np.cos(roll) * np.cos(pitch) - self.gravity

        # Attitude dynamics: roll/pitch first-order tracking; yaw = integrate rate
        droll  = (roll_cmd  - roll)  / self.tau_roll
        dpitch = (pitch_cmd - pitch) / self.tau_pitch
        dyaw   = yaw_rate

        dx = np.column_stack([vx, vy, vz, ax, ay, az, droll, dpitch, dyaw])

        if single:
            return dx[0]
        return dx

    def _rollout(
        self, x0: np.ndarray, U_samples: np.ndarray, references: np.ndarray
    ) -> np.ndarray:
        """
        Rollout dynamics for all samples and compute costs.

        Args:
            x0: Initial state (9,)
            U_samples: Control samples (K, N, 4)
            references: Reference states (N, 9)

        Returns:
            costs: Total cost for each sample (K,)
        """
        K = U_samples.shape[0]
        N = U_samples.shape[1]

        costs = np.zeros(K)
        x = np.tile(x0, (K, 1))  # (K, 9)

        best_traj_cost = np.inf
        best_traj_states = None

        for t in range(N):
            # State cost
            x_ref = references[t]  # (9,)
            x_err = x - x_ref  # (K, 9)
            state_cost = np.sum(x_err @ self.Q * x_err, axis=1)  # (K,)

            # Control cost
            u_err = U_samples[:, t, :] - self.u_hover  # (K, 4)
            ctrl_cost = np.sum(u_err @ self.R * u_err, axis=1)  # (K,)

            costs += state_cost + ctrl_cost

            # Step dynamics
            x = self._dynamics_step(x, U_samples[:, t, :])

            # Check for NaN / divergence
            diverged = np.any(np.isnan(x) | np.isinf(x), axis=1)
            costs[diverged] = 1e10

        # Terminal cost
        x_ref_terminal = references[-1]
        x_err_terminal = x - x_ref_terminal
        terminal_cost = np.sum(x_err_terminal @ self.Q_terminal * x_err_terminal, axis=1)
        costs += terminal_cost

        return costs

    def compute_control(
        self,
        state: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Run MPPI optimization and return the optimal control.

        Args:
            state: Current state (13,) full or (9,) reduced
            reference: Reference sequence (N, 6) or (N, 9)

        Returns:
            ctrl: Optimal control vector (4,)
        """
        # Convert full state to reduced
        if len(state) == 13:
            x0 = self._full_to_reduced_state(state)
        else:
            x0 = state.copy()

        # Build full reference
        ref_full = np.zeros((self.N, self.n_state))
        if reference.shape[1] == 6:
            ref_full[:, :3] = reference[:, :3]
            ref_full[:, 3:6] = reference[:, 3:6]
        elif reference.shape[1] == 9:
            ref_full = reference.copy()
        else:
            raise ValueError(f"Reference shape {reference.shape} not supported")

        # Sample control perturbations
        noise = self.rng.normal(0, 1, size=(self.K, self.N, self.n_ctrl))
        noise *= self.noise_sigma[np.newaxis, np.newaxis, :]

        # Form candidate control sequences
        U_samples = self.U_nominal[np.newaxis, :, :] + noise  # (K, N, 4)

        # Clip to control bounds
        U_samples = np.clip(U_samples, self.u_min, self.u_max)

        # Rollout and compute costs
        costs = self._rollout(x0, U_samples, ref_full)

        # MPPI weight computation
        # Shift costs for numerical stability
        cost_min = np.min(costs)
        weights = np.exp(-(costs - cost_min) / self.lam)
        weights_sum = np.sum(weights)

        if weights_sum < 1e-10:
            # All trajectories are terrible, fall back to hover
            weights = np.ones(self.K) / self.K
        else:
            weights /= weights_sum

        # Weighted average of perturbations
        weighted_noise = np.sum(
            weights[:, np.newaxis, np.newaxis] * noise, axis=0
        )  # (N, 4)

        # Update nominal control sequence
        self.U_nominal += weighted_noise

        # Optional: smooth the control sequence
        if self.smoothing_alpha > 0:
            self._smooth_controls()

        # Clip
        self.U_nominal = np.clip(self.U_nominal, self.u_min, self.u_max)

        # Extract first control
        ctrl = self.U_nominal[0].copy()

        # Shift nominal sequence (receding horizon)
        self.U_nominal = np.roll(self.U_nominal, -1, axis=0)
        self.U_nominal[-1] = self.u_hover  # Append hover as last element

        # Store diagnostics
        self._best_cost = cost_min
        self._mean_cost = np.mean(costs)

        return ctrl

    def _smooth_controls(self):
        """Apply exponential smoothing to the control sequence."""
        alpha = self.smoothing_alpha
        for t in range(1, self.N):
            self.U_nominal[t] = (
                alpha * self.U_nominal[t - 1] + (1 - alpha) * self.U_nominal[t]
            )

    def _full_to_reduced_state(self, state: np.ndarray) -> np.ndarray:
        """Convert 13D full state to 9D reduced state."""
        from .drone_env import quat_to_euler

        pos = state[0:3]
        quat = state[3:7]
        vel = state[7:10]
        euler = quat_to_euler(quat)
        return np.concatenate([pos, vel, euler])

    def reset(self):
        """Reset the controller state."""
        self.U_nominal = np.tile(self.u_hover, (self.N, 1))
        self._best_cost = np.inf
        self._mean_cost = np.inf
        self._best_trajectory = None

    def get_info(self) -> Dict[str, Any]:
        """Get controller diagnostics."""
        return {
            "type": "MPPI",
            "horizon": self.N,
            "n_samples": self.K,
            "temperature": self.lam,
            "best_cost": float(self._best_cost),
            "mean_cost": float(self._mean_cost),
            "dt": self.dt,
        }
