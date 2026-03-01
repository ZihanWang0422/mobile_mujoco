"""
Model Predictive Controller (MPC) for quadrotor trajectory tracking.

Uses CasADi for nonlinear optimization with a simplified quadrotor
dynamics model. The controller solves a finite-horizon optimal control
problem at each timestep and applies the first control input.

Dynamics model:
    - Position: dp = v
    - Velocity: dv = R(q) * [0, 0, T/m] + [0, 0, -g] + drag
    - Orientation: simplified using small-angle assumption for roll/pitch
    - Angular rate: first-order response to moment commands

The controller tracks [x, y, z, vx, vy, vz] references while
keeping the drone upright and minimizing control effort.
"""

import numpy as np
import casadi as ca
from typing import Optional, Dict, Any


class MPCController:
    """
    Nonlinear Model Predictive Controller for Crazyflie 2 drone.

    Solves the following optimization at each step:

        min  Σ_{k=0}^{N-1} [ (x_k - x_ref)^T Q (x_k - x_ref)
                             + (u_k - u_hover)^T R (u_k - u_hover) ]
             + (x_N - x_ref)^T Q_f (x_N - x_ref)

        s.t. x_{k+1} = f(x_k, u_k)         (dynamics)
             u_min <= u_k <= u_max           (actuator limits)
             x_0 = x_current                 (initial condition)

    State vector (9):  [x, y, z, vx, vy, vz, roll, pitch, yaw]
    Control vector (4): [thrust, roll_cmd, pitch_cmd, yaw_cmd]
    """

    def __init__(
        self,
        dt: float = 0.02,
        horizon: int = 25,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Q_terminal: Optional[np.ndarray] = None,
        mass: float = 0.027,
        gravity: float = 9.81,
        max_thrust: float = 0.35,
        max_moment: float = 1.0,
        verbose: bool = False,
    ):
        """
        Args:
            dt: Control timestep (seconds). Should be >= simulation dt.
            horizon: MPC prediction horizon (number of steps).
            Q: State cost weight matrix (9x9). Penalizes tracking error.
            R: Control cost weight matrix (4x4). Penalizes control effort.
            Q_terminal: Terminal state cost (9x9). Penalizes final state error.
            mass: Drone mass in kg.
            gravity: Gravitational acceleration in m/s^2.
            max_thrust: Maximum thrust in N.
            max_moment: Maximum moment command magnitude.
            verbose: Print solver output if True.
        """
        self.dt = dt
        self.N = horizon
        self.mass = mass
        self.gravity = gravity
        self.n_state = 9   # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.n_ctrl = 4    # [thrust, roll_cmd, pitch_cmd, yaw_cmd]

        # Cost matrices
        if Q is None:
            #          x     y     z    vx    vy    vz   roll  pitch  yaw
            Q = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 10.0, 10.0, 5.0])
        if R is None:
            R = np.diag([1.0, 0.1, 0.1, 0.1])
        if Q_terminal is None:
            Q_terminal = Q * 3.0  # Higher terminal cost for stability

        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal

        # Control bounds: [thrust_N, roll_rad, pitch_rad, yaw_rate_rads]
        max_angle = np.deg2rad(30.0)          # ±30° roll/pitch
        max_yaw_rate = np.deg2rad(180.0)      # ±180°/s yaw rate
        self.u_min = np.array([0.0,        -max_angle,  -max_angle,  -max_yaw_rate])
        self.u_max = np.array([max_thrust,  max_angle,   max_angle,   max_yaw_rate])

        # Hover equilibrium command
        self.u_hover = np.array([mass * gravity, 0.0, 0.0, 0.0])

        # Attitude bandwidth (inner PID approx): ~20 Hz → τ ≈ 0.05 s
        self.tau_roll  = 0.05
        self.tau_pitch = 0.05

        self.verbose = verbose

        # Build the CasADi optimization problem
        self._build_solver()

        # Warm start storage
        self._prev_u = None

    def _dynamics(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """
        Simplified quadrotor dynamics for MPC prediction model.

        State:   [x, y, z, vx, vy, vz, roll, pitch, yaw]
        Control: [thrust_N, roll_cmd_rad, pitch_cmd_rad, yaw_rate_cmd_rads]

        Attitude is modelled as a first-order system driven by the inner-loop
        PID (bandwidth ≈ 1/tau).  Yaw is integrated from yaw_rate command.
        """
        # Unpack state
        vx, vy, vz = x[3], x[4], x[5]
        roll, pitch, yaw = x[6], x[7], x[8]

        # Unpack control
        thrust    = u[0]          # [N]
        roll_cmd  = u[1]          # [rad]
        pitch_cmd = u[2]          # [rad]
        yaw_rate  = u[3]          # [rad/s]

        # Thrust direction in world frame (ZYX Euler, small-angle safe)
        ax = (thrust / self.mass) * (
            ca.cos(yaw) * ca.sin(pitch) + ca.sin(yaw) * ca.sin(roll)
        )
        ay = (thrust / self.mass) * (
            ca.sin(yaw) * ca.sin(pitch) - ca.cos(yaw) * ca.sin(roll)
        )
        az = (thrust / self.mass) * ca.cos(roll) * ca.cos(pitch) - self.gravity

        # Translational dynamics
        dx   = vx
        dy   = vy
        dz   = vz
        dvx  = ax
        dvy  = ay
        dvz  = az

        # Attitude dynamics — first-order tracking of commanded angles
        droll  = (roll_cmd  - roll)  / self.tau_roll
        dpitch = (pitch_cmd - pitch) / self.tau_pitch
        dyaw   = yaw_rate               # directly integrate rate command

        return ca.vertcat(dx, dy, dz, dvx, dvy, dvz, droll, dpitch, dyaw)

    def _build_solver(self):
        """Build the CasADi NLP solver."""
        nx = self.n_state
        nu = self.n_ctrl
        N = self.N

        # Decision variables
        # States: X = [x_0, x_1, ..., x_N] -> (N+1) * nx
        # Controls: U = [u_0, u_1, ..., u_{N-1}] -> N * nu
        X = ca.SX.sym("X", nx, N + 1)
        U = ca.SX.sym("U", nu, N)
        P = ca.SX.sym("P", nx + N * nx)  # [x_init, x_ref_0, ..., x_ref_{N-1}]

        # Cost function
        cost = 0.0
        constraints = []
        lbg = []
        ubg = []

        # Initial condition constraint
        constraints.append(X[:, 0] - P[:nx])
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        Q = ca.DM(self.Q)
        R = ca.DM(self.R)
        Q_f = ca.DM(self.Q_terminal)
        u_hover = ca.DM(self.u_hover)

        for k in range(N):
            # State reference at step k
            x_ref = P[nx + k * nx: nx + (k + 1) * nx]

            # Stage cost
            x_err = X[:, k] - x_ref
            u_err = U[:, k] - u_hover
            cost += ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_err.T, R, u_err])

            # Dynamics constraint (RK4 integration)
            x_k = X[:, k]
            u_k = U[:, k]

            k1 = self._dynamics(x_k, u_k)
            k2 = self._dynamics(x_k + self.dt / 2 * k1, u_k)
            k3 = self._dynamics(x_k + self.dt / 2 * k2, u_k)
            k4 = self._dynamics(x_k + self.dt * k3, u_k)
            x_next = x_k + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            constraints.append(X[:, k + 1] - x_next)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

        # Terminal cost (use last reference)
        x_ref_terminal = P[nx + (N - 1) * nx: nx + N * nx]
        x_err_terminal = X[:, N] - x_ref_terminal
        cost += ca.mtimes([x_err_terminal.T, Q_f, x_err_terminal])

        # Flatten decision variables
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*constraints)

        # Bounds on decision variables
        n_vars = (N + 1) * nx + N * nu
        self._lbx = np.full(n_vars, -np.inf)
        self._ubx = np.full(n_vars, np.inf)

        # State bounds (mild, to keep solver stable)
        for k in range(N + 1):
            idx = k * nx
            # Position bounds
            self._lbx[idx:idx+3] = -50.0
            self._ubx[idx:idx+3] = 50.0
            # Velocity bounds
            self._lbx[idx+3:idx+6] = -10.0
            self._ubx[idx+3:idx+6] = 10.0
            # Angle bounds
            self._lbx[idx+6:idx+9] = -np.pi / 3
            self._ubx[idx+6:idx+9] = np.pi / 3

        # Control bounds
        ctrl_start = (N + 1) * nx
        for k in range(N):
            idx = ctrl_start + k * nu
            self._lbx[idx:idx+nu] = self.u_min
            self._ubx[idx:idx+nu] = self.u_max

        # Create solver
        nlp = {"x": opt_vars, "f": cost, "g": g, "p": P}
        opts = {
            "ipopt.print_level": 3 if self.verbose else 0,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "print_time": 1 if self.verbose else 0,
        }
        self._solver = ca.nlpsol("mpc_solver", "ipopt", nlp, opts)

        # Store dimensions
        self._nx = nx
        self._nu = nu
        self._n_vars = n_vars
        self._n_params = nx + N * nx
        self._X = X
        self._U = U
        self._P = P

    def compute_control(
        self,
        state: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Solve MPC optimization and return the first control input.

        Args:
            state: Current state [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13,)
                   OR reduced state [x, y, z, vx, vy, vz, roll, pitch, yaw] (9,)
            reference: Reference sequence (N, 6) = [x, y, z, vx, vy, vz] per step
                       OR (N, 9) with full state reference

        Returns:
            ctrl: Control vector [thrust, roll_moment, pitch_moment, yaw_moment] (4,)
        """
        # Convert full state to reduced state if needed
        if len(state) == 13:
            x_mpc = self._full_to_reduced_state(state)
        else:
            x_mpc = state.copy()

        # Build reference parameter vector
        ref_full = np.zeros((self.N, self.n_state))
        if reference.shape[1] == 6:
            # Only position + velocity reference, zero attitude desired
            ref_full[:, :3] = reference[:, :3]  # position
            ref_full[:, 3:6] = reference[:, 3:6]  # velocity
            # roll, pitch, yaw = 0 (hover orientation)
        elif reference.shape[1] == 9:
            ref_full = reference
        else:
            raise ValueError(f"Reference shape {reference.shape} not supported")

        # Parameter vector: [x_init, ref_0, ref_1, ..., ref_{N-1}]
        p = np.concatenate([x_mpc, ref_full.flatten()])

        # Initial guess (warm start)
        x0 = np.zeros(self._n_vars)
        if self._prev_u is not None:
            # Shift previous solution
            ctrl_start = (self.N + 1) * self._nx
            for k in range(self.N - 1):
                x0[ctrl_start + k * self._nu: ctrl_start + (k + 1) * self._nu] = \
                    self._prev_u[k + 1]
            x0[ctrl_start + (self.N - 1) * self._nu:] = self._prev_u[-1]
        else:
            # Initialize with hover
            ctrl_start = (self.N + 1) * self._nx
            for k in range(self.N):
                x0[ctrl_start + k * self._nu: ctrl_start + (k + 1) * self._nu] = \
                    self.u_hover

        # Initialize states with current state
        for k in range(self.N + 1):
            x0[k * self._nx: (k + 1) * self._nx] = x_mpc

        # Solve
        sol = self._solver(
            x0=x0, lbx=self._lbx, ubx=self._ubx,
            lbg=0.0, ubg=0.0, p=p
        )

        # Extract solution
        opt_x = np.array(sol["x"]).flatten()
        ctrl_start = (self.N + 1) * self._nx

        # Store all controls for warm start
        self._prev_u = []
        for k in range(self.N):
            uk = opt_x[ctrl_start + k * self._nu: ctrl_start + (k + 1) * self._nu]
            self._prev_u.append(uk.copy())
        self._prev_u = np.array(self._prev_u)

        # Return first control
        u_opt = self._prev_u[0].copy()

        # Map from MPC control to actuator commands
        ctrl = self._mpc_to_actuator_ctrl(u_opt)
        return ctrl

    def _full_to_reduced_state(self, state: np.ndarray) -> np.ndarray:
        """
        Convert 13D full state to 9D MPC state.

        Full:    [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        Reduced: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        """
        from .drone_env import quat_to_euler

        pos = state[0:3]
        quat = state[3:7]
        vel = state[7:10]

        euler = quat_to_euler(quat)
        return np.concatenate([pos, vel, euler])

    def _mpc_to_actuator_ctrl(self, u_mpc: np.ndarray) -> np.ndarray:
        """
        Map MPC internal control to the outer-loop command consumed by AttitudePID.

        MPC internal: [thrust_N, roll_cmd_rad, pitch_cmd_rad, yaw_rate_cmd_rads]
        Outer cmd:    same format — passed directly to CascadeController.inner.compute()

        NOTE: this does NOT write to MuJoCo ctrl[].
              The run_*.py scripts feed this into AttitudePID which produces
              the final [body_thrust, x_moment, y_moment, z_moment].
        """
        # Clamp roll/pitch to safe range (±30°)
        max_angle = np.deg2rad(30.0)
        thrust  = float(np.clip(u_mpc[0], 0.0, 0.35))
        roll    = float(np.clip(u_mpc[1], -max_angle, max_angle))
        pitch   = float(np.clip(u_mpc[2], -max_angle, max_angle))
        yaw_rate = float(np.clip(u_mpc[3], -np.pi, np.pi))
        return np.array([thrust, roll, pitch, yaw_rate])

    def reset(self):
        """Reset warm-start state."""
        self._prev_u = None

    def get_predicted_trajectory(self) -> Optional[np.ndarray]:
        """
        Get the predicted state trajectory from the last solve.

        Returns:
            trajectory: (N+1, 9) predicted states or None if not available
        """
        if self._prev_u is None:
            return None
        return self._predicted_states

    def get_info(self) -> Dict[str, Any]:
        """Get controller info / diagnostics."""
        return {
            "type": "MPC",
            "horizon": self.N,
            "dt": self.dt,
            "n_state": self.n_state,
            "n_ctrl": self.n_ctrl,
        }
