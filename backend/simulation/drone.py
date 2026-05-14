"""
Drone Physics Simulation
========================
A simplified 6-DOF drone model with realistic physics:
- Position (x, y, z)
- Velocity (vx, vy, vz)
- Orientation (roll, pitch, yaw)
- Angular velocity (p, q, r)

The drone is controlled by 4 rotors (quadcopter layout):
    Rotor 1 (front-left)  Rotor 2 (front-right)
    Rotor 3 (rear-left)   Rotor 4 (rear-right)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DroneState:
    """Complete drone state at a given moment."""
    # Position (meters)
    x: float = 0.0
    y: float = 0.0
    z: float = 10.0  # starts at 10m altitude
    
    # Velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # Orientation (radians)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    # Angular velocity (rad/s)
    p: float = 0.0  # roll rate
    q: float = 0.0  # pitch rate
    r: float = 0.0  # yaw rate
    
    # Path tracking
    path_x: list = field(default_factory=list)
    path_y: list = field(default_factory=list)
    path_z: list = field(default_factory=list)
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for LNN input."""
        return np.array([
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            self.roll, self.pitch, self.yaw,
            self.p, self.q, self.r
        ], dtype=np.float32)
    
    def copy(self) -> 'DroneState':
        """Create a copy without path history."""
        return DroneState(
            x=self.x, y=self.y, z=self.z,
            vx=self.vx, vy=self.vy, vz=self.vz,
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
            p=self.p, q=self.q, r=self.r
        )


class DronePhysics:
    """
    Quadcopter physics model.
    
    Based on simplified Newtonian mechanics with:
    - Gravity: 9.81 m/s²
    - Air drag: proportional to velocity squared
    - Rotor thrust: directly controlled
    - Cross-coupling between rotors for realistic torque
    """
    
    # Physical constants
    MASS = 1.2          # kg
    GRAVITY = 9.81      # m/s²
    ARM_LENGTH = 0.25   # m (distance from center to rotor)
    DRAG_COEFF = 0.15   # aerodynamic drag coefficient
    MAX_THRUST = 15.0   # N per rotor (total hover needs ~12N for 1.2kg)
    INERTIA_XX = 0.015  # kg·m² (roll inertia)
    INERTIA_YY = 0.015  # kg·m² (pitch inertia)
    INERTIA_ZZ = 0.022  # kg·m² (yaw inertia)
    TORQUE_COEFF = 0.02 # rotor torque coefficient
    
    # Hover thrust per rotor (each rotor supports 1/4 of weight)
    HOVER_THRUST = MASS * GRAVITY / 4.0
    
    def __init__(self):
        self.state = DroneState()
        self.dt = 0.05  # 20 Hz simulation
        self.time = 0.0
        self.total_distance = 0.0
        self.crashes = 0
        self.waypoints_reached = 0
        
    def reset(self, x=0.0, y=0.0, z=10.0):
        """Reset drone to starting position."""
        self.state = DroneState(x=x, y=y, z=z)
        self.time = 0.0
        self.total_distance = 0.0
        self.crashes = 0
        self.state.path_x = [x]
        self.state.path_y = [y]
        self.state.path_z = [z]
    
    def apply_control(self, thrusts: np.ndarray, wind_force: np.ndarray = None) -> DroneState:
        """
        Apply rotor thrusts and advance physics by one time step.
        
        Args:
            thrusts: Array [T1, T2, T3, T4] — rotor thrust in [0, 1] (normalized)
            wind_force: External wind force [Fx, Fy, Fz] in Newtons
        
        Returns:
            Updated drone state
        """
        s = self.state
        
        # Scale thrusts to actual Newtons
        T = np.clip(thrusts, 0.0, 1.0) * self.MAX_THRUST
        T1, T2, T3, T4 = T
        
        # Total thrust (upward)
        total_thrust = T1 + T2 + T3 + T4
        
        # Torques from differential thrust
        # Roll: left rotors (1,3) vs right rotors (2,4)
        tau_roll  = self.ARM_LENGTH * ((T1 + T3) - (T2 + T4))
        # Pitch: front rotors (1,2) vs rear rotors (3,4)
        tau_pitch = self.ARM_LENGTH * ((T3 + T4) - (T1 + T2))
        # Yaw: counter-rotating pairs
        tau_yaw   = self.TORQUE_COEFF * ((T1 + T4) - (T2 + T3))
        
        # ---- Rotation dynamics ----
        # Angular accelerations
        alpha_roll  = tau_roll  / self.INERTIA_XX
        alpha_pitch = tau_pitch / self.INERTIA_YY
        alpha_yaw   = tau_yaw   / self.INERTIA_ZZ
        
        # Integrate angular velocity
        s.p += alpha_roll  * self.dt
        s.q += alpha_pitch * self.dt
        s.r += alpha_yaw   * self.dt
        
        # Damping (gyroscopic effects)
        s.p *= 0.92
        s.q *= 0.92
        s.r *= 0.95
        
        # Integrate orientation
        s.roll  += s.p * self.dt
        s.pitch += s.q * self.dt
        s.yaw   += s.r * self.dt
        
        # Clamp angles to prevent flip
        s.roll  = np.clip(s.roll,  -0.7, 0.7)
        s.pitch = np.clip(s.pitch, -0.7, 0.7)
        
        # ---- Translation dynamics ----
        # Thrust vector in body frame → world frame
        # Simplified: thrust acts mostly upward, tilted by roll/pitch
        Fx_thrust = total_thrust * np.sin(s.pitch)
        Fy_thrust = -total_thrust * np.sin(s.roll) * np.cos(s.pitch)
        Fz_thrust = total_thrust * np.cos(s.roll) * np.cos(s.pitch)
        
        # Gravity
        Fz_gravity = -self.MASS * self.GRAVITY
        
        # Air drag (opposes motion)
        Fx_drag = -self.DRAG_COEFF * s.vx * abs(s.vx)
        Fy_drag = -self.DRAG_COEFF * s.vy * abs(s.vy)
        Fz_drag = -self.DRAG_COEFF * s.vz * abs(s.vz)
        
        # Wind disturbance
        Fx_wind, Fy_wind, Fz_wind = (wind_force if wind_force is not None else [0, 0, 0])
        
        # Net forces
        Fx = Fx_thrust + Fx_drag + Fx_wind
        Fy = Fy_thrust + Fy_drag + Fy_wind
        Fz = Fz_thrust + Fz_gravity + Fz_drag + Fz_wind
        
        # Accelerations (F = ma)
        ax = Fx / self.MASS
        ay = Fy / self.MASS
        az = Fz / self.MASS
        
        # Integrate velocity
        prev_vx, prev_vy, prev_vz = s.vx, s.vy, s.vz
        s.vx += ax * self.dt
        s.vy += ay * self.dt
        s.vz += az * self.dt
        
        # Clamp velocities (physical limit)
        max_vel = 15.0
        s.vx = np.clip(s.vx, -max_vel, max_vel)
        s.vy = np.clip(s.vy, -max_vel, max_vel)
        s.vz = np.clip(s.vz, -max_vel, max_vel)
        
        # Integrate position
        prev_x, prev_y, prev_z = s.x, s.y, s.z
        s.x += s.vx * self.dt
        s.y += s.vy * self.dt
        s.z += s.vz * self.dt
        
        # Ground collision
        if s.z < 0.5:
            s.z = 0.5
            s.vz = max(0, s.vz)
            if abs(s.vz) > 3.0:
                self.crashes += 1
        
        # Update tracking
        dist = np.sqrt((s.x - prev_x)**2 + (s.y - prev_y)**2 + (s.z - prev_z)**2)
        self.total_distance += dist
        self.time += self.dt
        
        # Store path (keep last 200 points)
        s.path_x.append(s.x)
        s.path_y.append(s.y)
        s.path_z.append(s.z)
        if len(s.path_x) > 200:
            s.path_x.pop(0)
            s.path_y.pop(0)
            s.path_z.pop(0)
        
        return s
    
    def get_error_to_target(self, target: np.ndarray) -> np.ndarray:
        """
        Compute position and velocity error to target waypoint.
        
        Args:
            target: [tx, ty, tz] — target position
        
        Returns:
            Error vector [ex, ey, ez, evx, evy, evz]
        """
        s = self.state
        pos_error = target - np.array([s.x, s.y, s.z])
        vel_error = np.array([-s.vx, -s.vy, -s.vz])  # want zero velocity at target
        return np.concatenate([pos_error, vel_error])
    
    def build_lnn_input(self, target: np.ndarray, wind: np.ndarray) -> np.ndarray:
        """
        Build the full input vector for the LNN controller.
        
        Input features (12 total):
        - Position error (3): ex, ey, ez
        - Velocity (3): vx, vy, vz
        - Orientation (3): roll, pitch, yaw
        - Wind sensor (3): wx, wy, wz (normalized)
        """
        s = self.state
        pos_error = target - np.array([s.x, s.y, s.z])
        velocity = np.array([s.vx, s.vy, s.vz])
        orientation = np.array([s.roll, s.pitch, s.yaw])
        wind_norm = wind / (np.linalg.norm(wind) + 1e-6) * min(np.linalg.norm(wind) / 5.0, 1.0)
        
        features = np.concatenate([pos_error, velocity, orientation, wind_norm])
        return features.astype(np.float32)
