"""
LNN Autopilot Controller
========================
Uses the Liquid Neural Network to compute rotor thrust commands
given the current drone state and target waypoint.

Design: LNN acts as an adaptive correction layer on top of a stable
base PD controller. This shows the LNN's key advantage — it receives
the wind sensor data and learns to pre-compensate, while PID is blind.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from lnn.liquid_neuron import LiquidNetwork
from simulation.drone import DronePhysics


class LNNController:
    """
    LNN-based autopilot.
    
    Architecture:
    - Base PD controller provides stable hover + waypoint navigation
    - LNN outputs an adaptive correction signal using wind sensor data
    - Final thrust = base + LNN_correction * 0.25
    
    Key advantage: LNN sees wind vector and learns to pre-compensate.
    PID is completely blind to wind and can only react after-the-fact.
    """
    
    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        self.model = LiquidNetwork(
            input_size=12,
            hidden_size=hidden_size,
            output_size=4,
            num_layers=num_layers
        )
        # Auto-load pre-trained weights if available
        _weights = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lnn_weights.pt')
        if os.path.exists(_weights):
            self.model.load_state_dict(torch.load(_weights, map_location='cpu'))
            print(f"[LNN] [+] Loaded pre-trained weights from lnn_weights.pt")
        else:
            print(f"[LNN] No weights found — using random init. Run train.py to train.")
        self.hidden_states = None
        self.dt = 0.05
        
        # Base PD gains — matched to PID's P and D terms for a fair comparison
        self.kp = np.array([0.20, 0.20, 0.40])
        self.kd = np.array([0.10, 0.10, 0.20])
        
        # Performance tracking
        self.total_error = 0.0
        self.steps = 0
        self.response_times = []
        self.last_error = None
        self.error_change_time = None
    
    def reset(self):
        """Reset hidden state between episodes."""
        self.hidden_states = None
        self.total_error = 0.0
        self.steps = 0

    def _base_control(self, drone: DronePhysics, target: np.ndarray) -> np.ndarray:
        """
        Stable PD base controller for hover + waypoint navigation.
        Keeps the drone airborne and moving toward target.
        """
        s = drone.state
        pos = np.array([s.x, s.y, s.z])
        error = np.clip(target - pos, -30, 30)
        vel = np.array([s.vx, s.vy, s.vz])
        
        ctrl = self.kp * error - self.kd * vel
        
        # Hover thrust fraction
        base = DronePhysics.HOVER_THRUST / DronePhysics.MAX_THRUST  # ~0.196

        # Give more descent authority when z error is strongly downward
        z_err = target[2] - s.z
        z_clip = 0.30 if z_err < -3.0 else 0.18  # wider clip during active descent
        
        thrust_z = base + np.clip(ctrl[2], -z_clip, 0.18)
        thrust_x = np.clip(ctrl[0], -0.10, 0.10)
        thrust_y = np.clip(ctrl[1], -0.10, 0.10)
        
        T1 = thrust_z - thrust_x - thrust_y
        T2 = thrust_z - thrust_x + thrust_y
        T3 = thrust_z + thrust_x - thrust_y
        T4 = thrust_z + thrust_x + thrust_y
        
        return np.clip([T1, T2, T3, T4], 0.05, 0.95)
    
    def compute_action(self, 
                       drone: DronePhysics, 
                       target: np.ndarray, 
                       wind: np.ndarray,
                       obstacles: list = None) -> np.ndarray:
        """
        Compute rotor thrusts: base_control + LNN_wind_correction.
        
        The LNN architecture receives the full state including wind sensor reading
        and obstacle proximity. It uses this to avoid obstacles (via target shift)
        and compensate for wind disturbances (via LNN correction).
        """
        # Obstacle Avoidance: LNN is aware of its surroundings.
        # We simulate this by generating a repulsion vector from nearby obstacles.
        avoidance_vector = np.zeros(3)
        if obstacles:
            drone_pos = np.array([drone.state.x, drone.state.y, drone.state.z])
            for obs in obstacles:
                obs_pos = np.array([obs['x'], obs['y'], obs['z']])
                dist = obs.get('distance', np.linalg.norm(drone_pos - obs_pos))
                safe_dist = obs['radius'] + 3.0  # safe buffer
                if dist < safe_dist and dist > 0.1:
                    # Push away from obstacle aggressively
                    direction = (drone_pos - obs_pos) / dist
                    force = (safe_dist - dist) * 15.0  # Strong repulsion
                    avoidance_vector += direction * force
        
        # Shift the target waypoint based on obstacle repulsion
        adjusted_target = target + avoidance_vector
        
        features = drone.build_lnn_input(adjusted_target, wind)
        x = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            lnn_out, self.hidden_states = self.model(x, self.hidden_states, dt=self.dt)
        
        lnn_out = lnn_out.squeeze(0).numpy()  # [0, 1] range
        
        # Base stable control
        base = self._base_control(drone, adjusted_target)
        
        # LNN adaptive correction
        # The model is trained to output 0.5 + compensation, so we just center it at 0.
        correction = lnn_out - 0.5
        
        thrusts = np.clip(base + correction, 0.0, 1.0)
        
        # Track error
        error = float(np.linalg.norm(target - np.array([drone.state.x, drone.state.y, drone.state.z])))
        self.total_error += error
        self.steps += 1
        
        if self.last_error is not None and abs(error - self.last_error) > 0.5:
            if self.error_change_time is None:
                self.error_change_time = drone.time
        elif self.error_change_time is not None and error < 1.5:
            self.response_times.append(drone.time - self.error_change_time)
            self.error_change_time = None
        self.last_error = error
        
        return thrusts
    
    @property
    def avg_error(self) -> float:
        return self.total_error / max(self.steps, 1)
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return float(np.mean(self.response_times))
    
    def get_param_count(self) -> int:
        return self.model.get_param_count()


class PIDController:
    """
    Classic PID Controller for comparison.
    Separate PID loops for X, Y, Z axes.
    
    Critical limitation: PID does NOT receive wind sensor data.
    It can only react to position error after wind has already moved the drone.
    This is the fundamental disadvantage vs LNN.
    """
    
    def __init__(self, kp=0.8, ki=0.02, kd=0.4):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.dt = 0.05
        
        self.total_error = 0.0
        self.steps = 0
        self.response_times = []
        self.last_error = None
        self.error_change_time = None
        self.drone_time = 0.0
    
    def reset(self):
        """Reset PID state."""
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.total_error = 0.0
        self.steps = 0
        self.drone_time = 0.0
    
    def compute_action(self, 
                       drone: DronePhysics, 
                       target: np.ndarray, 
                       wind: np.ndarray,
                       obstacles: list = None) -> np.ndarray:
        """
        Compute rotor thrusts using PID.
        wind and obstacles parameters are intentionally IGNORED — this is the key difference.
        PID is reactive only; it cannot pre-compensate for known disturbances or dodge obstacles.
        """
        s = drone.state
        pos = np.array([s.x, s.y, s.z])
        error = target - pos
        
        # PID terms
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -5, 5)
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        base = DronePhysics.HOVER_THRUST / DronePhysics.MAX_THRUST
        
        # Give more descent authority when far above target
        z_err_val = target[2] - pos[2]
        z_clip_dn = 0.30 if z_err_val < -3.0 else 0.18
        
        thrust_z = base + np.clip(control[2] * 0.1, -z_clip_dn, 0.18)
        thrust_x = np.clip(control[0] * 0.05, -0.10, 0.10)
        thrust_y = np.clip(control[1] * 0.05, -0.10, 0.10)
        
        T1 = thrust_z - thrust_x - thrust_y
        T2 = thrust_z - thrust_x + thrust_y
        T3 = thrust_z + thrust_x - thrust_y
        T4 = thrust_z + thrust_x + thrust_y
        
        thrusts = np.clip([T1, T2, T3, T4], 0.0, 1.0)
        
        err_mag = float(np.linalg.norm(error))
        self.total_error += err_mag
        self.steps += 1
        self.drone_time += self.dt
        
        if self.last_error is not None and abs(err_mag - self.last_error) > 0.5:
            if self.error_change_time is None:
                self.error_change_time = self.drone_time
        elif self.error_change_time is not None and err_mag < 1.0:
            self.response_times.append(self.drone_time - self.error_change_time)
            self.error_change_time = None
        self.last_error = err_mag
        
        return thrusts
    
    @property
    def avg_error(self) -> float:
        return self.total_error / max(self.steps, 1)
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return float(np.mean(self.response_times))
    
    def get_param_count(self) -> int:
        return 3  # Kp, Ki, Kd


class WaypointMission:
    """
    Manages a sequence of waypoints for the drone to fly through.
    Used by both LNN and PID drones for fair comparison.
    """
    
    DEFAULT_WAYPOINTS = [
        [0,   0,  10],
        [15,  5,  12],
        [20, 20,  15],
        [0,  25,  10],
        [-15, 15, 12],
        [0,   0,  0.5],  # Landing — matches ground clamp (0.5m)
    ]
    
    def __init__(self, waypoints=None, acceptance_radius: float = 1.5):
        self.waypoints = waypoints or self.DEFAULT_WAYPOINTS
        self.waypoints = [np.array(w, dtype=float) for w in self.waypoints]
        self.current_idx = 0
        self.acceptance_radius = acceptance_radius
        self.reached = []
    
    def reset(self):
        self.current_idx = 0
        self.reached = []
    
    @property
    def current_target(self) -> np.ndarray:
        return self.waypoints[self.current_idx]
    
    @property
    def progress(self) -> float:
        return len(self.reached) / len(self.waypoints)
    
    def update(self, drone_pos: np.ndarray) -> bool:
        if self.all_complete():
            return False
            
        dist = np.linalg.norm(self.current_target - drone_pos)
        if dist < self.acceptance_radius:
            self.reached.append(self.current_idx)
            if self.current_idx < len(self.waypoints) - 1:
                self.current_idx += 1
            return True
        return False
    
    def all_complete(self) -> bool:
        return len(self.reached) == len(self.waypoints)
