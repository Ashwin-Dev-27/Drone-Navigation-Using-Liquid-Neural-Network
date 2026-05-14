"""
Environment Simulation
======================
Models unpredictable real-world conditions:
1. Wind gusts — sudden directional force bursts
2. Turbulence — continuous random perturbation
3. Moving obstacles — spheres drifting through the flight zone
4. Weather zones — regions of dense wind
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class WindGust:
    """A discrete wind gust event."""
    direction: np.ndarray   # unit vector [dx, dy, dz]
    magnitude: float        # m/s (force in Newtons = mass * magnitude roughly)
    duration: float         # seconds
    elapsed: float = 0.0   # time since gust started
    
    @property
    def is_active(self) -> bool:
        return self.elapsed < self.duration
    
    def get_force(self) -> np.ndarray:
        if not self.is_active:
            return np.zeros(3)
        # Ramp up then ramp down
        progress = self.elapsed / self.duration
        envelope = 4 * progress * (1 - progress)  # parabolic, peaks at midpoint
        return self.direction * self.magnitude * envelope


@dataclass
class Obstacle:
    """A moving spherical obstacle."""
    x: float
    y: float
    z: float
    radius: float
    vx: float = 0.0  # velocity
    vy: float = 0.0
    vz: float = 0.0
    
    def update(self, dt: float, bounds: Tuple[float, float]):
        """Move obstacle and bounce off bounds."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        
        # Bounce off walls
        if not (bounds[0] < self.x < bounds[1]):
            self.vx *= -1
        if not (bounds[0] < self.y < bounds[1]):
            self.vy *= -1
        if not (2 < self.z < 30):
            self.vz *= -1
    
    def check_collision(self, drone_x: float, drone_y: float, drone_z: float) -> bool:
        """Check if drone is inside obstacle sphere."""
        dist = np.sqrt(
            (self.x - drone_x)**2 +
            (self.y - drone_y)**2 +
            (self.z - drone_z)**2
        )
        return dist < (self.radius + 0.5)  # 0.5m drone radius
    
    def to_dict(self) -> dict:
        return {
            'x': round(self.x, 2),
            'y': round(self.y, 2),
            'z': round(self.z, 2),
            'radius': self.radius
        }


class Environment:
    """
    Simulation environment with dynamic weather and obstacles.
    
    Controls:
    - Turbulence intensity (0 = calm, 1 = extreme)
    - Gust frequency and magnitude
    - Number and speed of moving obstacles
    """
    
    BOUNDS = (-50, 50)  # flight zone limits in x/y
    
    def __init__(self, 
                 turbulence: float = 0.3,
                 gust_frequency: float = 0.05,
                 num_obstacles: int = 5,
                 seed: int = 42):
        """
        Args:
            turbulence: Continuous noise level [0, 1]
            gust_frequency: Probability of a new gust per step
            num_obstacles: Number of moving obstacles
            seed: Random seed for reproducibility
        """
        self.turbulence = turbulence
        self.gust_frequency = gust_frequency
        self.num_obstacles = num_obstacles
        self.rng = np.random.RandomState(seed)
        
        self.gusts: List[WindGust] = []
        self.obstacles: List[Obstacle] = []
        self.base_wind = np.array([0.5, 0.2, 0.0])  # constant background wind
        self.current_wind = np.zeros(3)
        self.dt = 0.05
        self.time = 0.0
        
        self._spawn_obstacles()
    
    def _spawn_obstacles(self):
        """Create initial set of moving obstacles."""
        self.obstacles = []
        for _ in range(self.num_obstacles):
            self.obstacles.append(Obstacle(
                x=self.rng.uniform(-30, 30),
                y=self.rng.uniform(-30, 30),
                z=self.rng.uniform(5, 20),
                radius=self.rng.uniform(1.5, 4.0),
                vx=self.rng.uniform(-2, 2),
                vy=self.rng.uniform(-2, 2),
                vz=self.rng.uniform(-0.5, 0.5)
            ))
    
    def _maybe_spawn_gust(self):
        """Randomly spawn a wind gust."""
        if self.rng.random() < self.gust_frequency:
            # Random direction with slight upward/downward bias
            direction = self.rng.randn(3)
            direction[2] *= 0.3  # reduce vertical component
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            self.gusts.append(WindGust(
                direction=direction,
                magnitude=self.rng.uniform(2.0, 8.0),  # Newtons
                duration=self.rng.uniform(0.5, 3.0)    # seconds
            ))
    
    def step(self) -> np.ndarray:
        """
        Advance environment by one time step.
        
        Returns:
            Total wind force vector [Fx, Fy, Fz] in Newtons
        """
        self.time += self.dt
        
        # Update gusts
        self._maybe_spawn_gust()
        gust_force = np.zeros(3)
        active_gusts = []
        for gust in self.gusts:
            gust.elapsed += self.dt
            if gust.is_active:
                gust_force += gust.get_force()
                active_gusts.append(gust)
        self.gusts = active_gusts
        
        # Turbulence: continuous random noise
        turbulence_force = self.rng.randn(3) * self.turbulence * 2.0
        turbulence_force[2] *= 0.4  # less vertical turbulence
        
        # Base wind (slowly varies)
        self.base_wind += self.rng.randn(3) * 0.01
        self.base_wind = np.clip(self.base_wind, -1.5, 1.5)
        
        # Total wind
        self.current_wind = self.base_wind + gust_force + turbulence_force
        
        # Update moving obstacles
        for obs in self.obstacles:
            obs.update(self.dt, self.BOUNDS)
        
        return self.current_wind
    
    def check_collisions(self, drone_x: float, drone_y: float, drone_z: float) -> List[dict]:
        """Return list of collided obstacles."""
        return [
            obs.to_dict()
            for obs in self.obstacles
            if obs.check_collision(drone_x, drone_y, drone_z)
        ]
    
    def get_nearby_obstacles(self, 
                              drone_x: float, drone_y: float, drone_z: float, 
                              radius: float = 15.0) -> List[dict]:
        """Return obstacles within sensing radius."""
        nearby = []
        for obs in self.obstacles:
            dist = np.sqrt(
                (obs.x - drone_x)**2 +
                (obs.y - drone_y)**2 +
                (obs.z - drone_z)**2
            )
            if dist < radius:
                nearby.append({**obs.to_dict(), 'distance': round(dist, 2)})
        return nearby
        
    def is_area_clear(self, x: float, y: float, radius: float = 5.0) -> bool:
        """Check if a cylindrical area is free of obstacles for landing."""
        for obs in self.obstacles:
            # Only consider horizontal distance for a landing cylinder
            dist = np.sqrt((obs.x - x)**2 + (obs.y - y)**2)
            if dist < (obs.radius + radius):
                return False
        return True
    
    def set_difficulty(self, level: str):
        """Adjust difficulty preset."""
        presets = {
            'calm':    (0.05, 0.01, 2),
            'normal':  (0.3,  0.05, 5),
            'stormy':  (0.7,  0.12, 8),
            'extreme': (1.0,  0.20, 12),
        }
        if level in presets:
            self.turbulence, self.gust_frequency, self.num_obstacles = presets[level]
            self._spawn_obstacles()
    
    def get_state(self) -> dict:
        """Return environment state for API/frontend."""
        return {
            'wind': {
                'x': round(float(self.current_wind[0]), 3),
                'y': round(float(self.current_wind[1]), 3),
                'z': round(float(self.current_wind[2]), 3),
                'magnitude': round(float(np.linalg.norm(self.current_wind)), 3)
            },
            'gusts': len(self.gusts),
            'turbulence': self.turbulence,
            'obstacles': [obs.to_dict() for obs in self.obstacles],
            'time': round(self.time, 2)
        }
