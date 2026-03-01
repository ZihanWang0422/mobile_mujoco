"""
Trajectory generators for drone tracking.
Provides circle, lemniscate (figure-8), and custom trajectory generators.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class TrajectoryBase(ABC):
    """Base class for trajectory generators."""

    @abstractmethod
    def get_reference(self, t: float) -> np.ndarray:
        """
        Get reference state at time t.

        Returns:
            ref: [x, y, z, vx, vy, vz] reference state (6,)
        """
        pass

    def get_reference_sequence(self, t: float, N: int, dt: float) -> np.ndarray:
        """
        Get a sequence of reference states for MPC horizon.

        Args:
            t: Current time
            N: Prediction horizon length
            dt: Time step

        Returns:
            refs: (N, 6) reference state sequence
        """
        refs = np.zeros((N, 6))
        for i in range(N):
            refs[i] = self.get_reference(t + i * dt)
        return refs


class CircleTrajectory(TrajectoryBase):
    """
    Circular trajectory in the horizontal plane.

    The drone flies in a circle at a fixed altitude with configurable
    radius, angular speed, center position, and height.
    """

    def __init__(
        self,
        radius: float = 1.0,
        omega: float = 0.5,
        center: Tuple[float, float] = (0.0, 0.0),
        height: float = 1.0,
    ):
        """
        Args:
            radius: Circle radius in meters
            omega: Angular speed in rad/s
            center: (cx, cy) center of the circle in the x-y plane
            height: Flying altitude in meters
        """
        self.radius = radius
        self.omega = omega
        self.center = center
        self.height = height

    def get_reference(self, t: float) -> np.ndarray:
        """Get position and velocity reference at time t."""
        cx, cy = self.center
        r = self.radius
        w = self.omega

        # Position
        x = cx + r * np.cos(w * t)
        y = cy + r * np.sin(w * t)
        z = self.height

        # Velocity (derivative of position)
        vx = -r * w * np.sin(w * t)
        vy = r * w * np.cos(w * t)
        vz = 0.0

        return np.array([x, y, z, vx, vy, vz])


class LemniscateTrajectory(TrajectoryBase):
    """
    Lemniscate (figure-8) trajectory.

    Creates a figure-8 pattern using a lemniscate of Bernoulli,
    with configurable scale, speed, and height.
    """

    def __init__(
        self,
        scale: float = 1.5,
        omega: float = 0.4,
        center: Tuple[float, float] = (0.0, 0.0),
        height: float = 1.0,
        warmup_t: float = 0.0,
    ):
        """
        Args:
            scale: Scale factor for the figure-8
            omega: Angular speed in rad/s
            center: (cx, cy) center of the figure-8
            height: Flying altitude in meters
            warmup_t: Soft-start duration (s). During [0, warmup_t] the
                      reference velocity is ramped from 0 to full speed
                      while position is held at t=0 start point, so the
                      drone accelerates gently before entering the loop.
        """
        self.scale = scale
        self.omega = omega
        self.center = center
        self.height = height
        self.warmup_t = warmup_t

    def get_reference(self, t: float) -> np.ndarray:
        cx, cy = self.center
        a = self.scale
        w = self.omega

        if self.warmup_t > 0 and t < self.warmup_t:
            # Warmup phase: stay at the figure-8 entry point (theta=0),
            # ramp reference velocity from 0 to full using ease-in
            alpha = t / self.warmup_t        # 0 → 1
            ramp  = alpha * alpha            # quadratic ease-in

            theta0 = 0.0
            denom0 = 1.0 + np.sin(theta0) ** 2
            x = cx + a * np.cos(theta0) / denom0
            y = cy + a * np.sin(theta0) * np.cos(theta0) / denom0
            z = self.height

            # Full-speed velocity at theta=0 (for reference scaling)
            dt_num = 1e-4
            theta1 = w * dt_num
            denom1 = 1.0 + np.sin(theta1) ** 2
            x1 = cx + a * np.cos(theta1) / denom1
            y1 = cy + a * np.sin(theta1) * np.cos(theta1) / denom1
            full_vx = (x1 - x) / dt_num
            full_vy = (y1 - y) / dt_num
            return np.array([x, y, z, full_vx * ramp, full_vy * ramp, 0.0])

        # Normal lemniscate (time shifted by warmup_t so theta=0 at t=warmup_t)
        theta = w * (t - self.warmup_t)

        # Lemniscate parametric equations
        denom = 1.0 + np.sin(theta) ** 2
        x = cx + a * np.cos(theta) / denom
        y = cy + a * np.sin(theta) * np.cos(theta) / denom
        z = self.height

        # Numerical velocity approximation
        dt_num = 1e-4
        theta2 = theta + w * dt_num
        denom2 = 1.0 + np.sin(theta2) ** 2
        x2 = cx + a * np.cos(theta2) / denom2
        y2 = cy + a * np.sin(theta2) * np.cos(theta2) / denom2

        vx = (x2 - x) / dt_num
        vy = (y2 - y) / dt_num
        vz = 0.0

        return np.array([x, y, z, vx, vy, vz])


class HelixTrajectory(TrajectoryBase):
    """
    Helical (spiral) trajectory combining circular motion with vertical ascent.
    """

    def __init__(
        self,
        radius: float = 1.0,
        omega: float = 0.5,
        climb_rate: float = 0.1,
        center: Tuple[float, float] = (0.0, 0.0),
        start_height: float = 0.5,
        max_height: float = 2.5,
    ):
        self.radius = radius
        self.omega = omega
        self.climb_rate = climb_rate
        self.center = center
        self.start_height = start_height
        self.max_height = max_height

    def get_reference(self, t: float) -> np.ndarray:
        cx, cy = self.center
        r = self.radius
        w = self.omega

        x = cx + r * np.cos(w * t)
        y = cy + r * np.sin(w * t)
        z = min(self.start_height + self.climb_rate * t, self.max_height)

        vx = -r * w * np.sin(w * t)
        vy = r * w * np.cos(w * t)
        vz = self.climb_rate if z < self.max_height else 0.0

        return np.array([x, y, z, vx, vy, vz])
