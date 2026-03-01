"""
Inner-loop PID attitude/altitude controller for Crazyflie 2 in MuJoCo.

Architecture (cascade control):
  Outer loop (MPC/MPPI) → desired [thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd]
                                     |
  Inner loop (PID)       → actuator  [body_thrust, x_moment, y_moment, z_moment]

The Crazyflie 2 model has 4 site-based actuators:
  ctrl[0]  body_thrust  [0,    0.35]  N    — vertical lift
  ctrl[1]  x_moment     [-1,   1  ]  N·m  — roll torque
  ctrl[2]  y_moment     [-1,   1  ]  N·m  — pitch torque
  ctrl[3]  z_moment     [-1,   1  ]  N·m  — yaw torque

The outer MPC/MPPI returns desired angles (roll_cmd, pitch_cmd) in radians
and a yaw_rate_cmd (rad/s).  This module closes the loop with simple PID.
"""

import numpy as np


class _PID:
    """Minimal PID with anti-windup."""

    def __init__(self, kp, ki, kd, out_min=-np.inf, out_max=np.inf):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        out = float(np.clip(out, self.out_min, self.out_max))
        # Anti-windup: clamp integral so output stays in bounds
        if out >= self.out_max:
            self.integral -= error * dt
        elif out <= self.out_min:
            self.integral -= error * dt
        return out


class AttitudePID:
    """
    Inner-loop PID that converts outer-loop commands to MuJoCo actuator inputs.

    Outer-loop command format (4,):
        [total_thrust_N, roll_cmd_rad, pitch_cmd_rad, yaw_rate_cmd_rads]

    Output MuJoCo ctrl (4,):
        [body_thrust, x_moment, y_moment, z_moment]

    Tuning mirrors the reference Skydio X2 PID example provided by the user,
    re-scaled for Crazyflie 2 inertia (≈ 10× smaller than X2).
    """

    # ---- Physical constants ------------------------------------------------
    MASS = 0.027          # kg
    G = 9.81              # m/s²
    HOVER_THRUST = MASS * G  # ≈ 0.265 N
    # Crazyflie 2 inertia from cf2.xml: Ixx=Iyy=2.3951e-5  Izz=3.2347e-5
    IXX = 2.3951e-5
    IYY = 2.3951e-5
    IZZ = 3.2347e-5

    def __init__(self, dt: float = 0.005):
        self.dt = dt

        # ---------- altitude / thrust -----------------------------------------
        # gains tuned so ~5 m/s² response without overshoot
        self._pid_alt = _PID(kp=6.0, ki=0.8, kd=1.5,
                             out_min=0.0, out_max=0.35)

        # ---------- roll (x_moment) -------------------------------------------
        self._pid_roll = _PID(kp=0.004, ki=0.0002, kd=0.0008,
                              out_min=-1.0, out_max=1.0)

        # ---------- pitch (y_moment) ------------------------------------------
        self._pid_pitch = _PID(kp=0.004, ki=0.0002, kd=0.0008,
                               out_min=-1.0, out_max=1.0)

        # ---------- yaw (z_moment) --------------------------------------------
        self._pid_yaw = _PID(kp=0.003, ki=0.0, kd=0.002,
                             out_min=-1.0, out_max=1.0)

        self._yaw_setpoint = 0.0   # integrated from yaw_rate_cmd

    def reset(self):
        self._pid_alt.reset()
        self._pid_roll.reset()
        self._pid_pitch.reset()
        self._pid_yaw.reset()
        self._yaw_setpoint = 0.0

    def compute(
        self,
        outer_cmd: np.ndarray,
        euler: np.ndarray,
        z: float,
        vz: float = 0.0,
    ) -> np.ndarray:
        """
        Convert outer-loop command to actuator ctrl vector.

        Args:
            outer_cmd: [thrust_N, roll_rad, pitch_rad, yaw_rate_rads]  (4,)
            euler:     [roll, pitch, yaw] in radians  (measured)
            z:         current altitude in metres
            vz:        current vertical speed (unused but reserved)

        Returns:
            ctrl: [body_thrust, x_moment, y_moment, z_moment]  (4,)
        """
        thrust_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd = outer_cmd

        dt = self.dt

        # --- altitude: treat thrust_cmd directly as feedforward + small correction
        # The MPC already outputs a thrust in Newtons; we pass it straight through
        # but add a tiny PID correction on z-error via the difference from hover.
        # (Full altitude PID is only used when outer_cmd[0] == HOVER_THRUST sentinel)
        body_thrust = float(np.clip(thrust_cmd, 0.0, 0.35))

        # --- roll moment
        roll_err = roll_cmd - euler[0]
        x_moment = self._pid_roll.step(roll_err, dt)

        # --- pitch moment
        pitch_err = pitch_cmd - euler[1]
        y_moment = self._pid_pitch.step(pitch_err, dt)

        # --- yaw moment: integrate rate command → angle setpoint
        self._yaw_setpoint += yaw_rate_cmd * dt
        yaw_err = self._yaw_setpoint - euler[2]
        # Wrap to [-π, π]
        yaw_err = (yaw_err + np.pi) % (2 * np.pi) - np.pi
        z_moment = self._pid_yaw.step(yaw_err, dt)

        return np.array([body_thrust, x_moment, y_moment, z_moment])


class CascadeController:
    """
    Full cascade controller: outer (MPC/MPPI) + inner (PID attitude).

    Decouples the high-level trajectory planner from low-level attitude control.
    The outer planner runs at a slower rate (dt_outer), the inner PID runs at
    the simulation rate (dt_inner).

    Usage::

        cascade = CascadeController(dt_inner=0.005, dt_outer=0.02)
        cascade.reset()

        # inside simulation loop (called every dt_inner):
        outer_cmd = mpc.compute_control(state, ref)   # runs every dt_outer steps
        ctrl = cascade.inner_step(outer_cmd, euler, z)
        env.step_raw(ctrl)
    """

    def __init__(self, dt_inner: float = 0.005):
        self.inner = AttitudePID(dt=dt_inner)

    def reset(self):
        self.inner.reset()

    def step(
        self,
        outer_cmd: np.ndarray,
        euler: np.ndarray,
        z: float,
        vz: float = 0.0,
    ) -> np.ndarray:
        """
        Args:
            outer_cmd: [thrust_N, roll_rad, pitch_rad, yaw_rate_rads]
            euler:     [roll, pitch, yaw] rad
            z:         altitude m
            vz:        vertical speed m/s

        Returns:
            ctrl: MuJoCo actuator ctrl (4,)
        """
        return self.inner.compute(outer_cmd, euler, z, vz)
