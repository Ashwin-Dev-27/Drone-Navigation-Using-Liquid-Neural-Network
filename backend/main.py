"""
FastAPI WebSocket Server
========================
Streams real-time simulation data to the frontend dashboard.
Runs both LNN and PID drone simulations simultaneously for comparison.
"""

import asyncio
import json
import numpy as np
import sys
import os
import time


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_json(data: dict) -> str:
    """Serialize dict to JSON string, converting numpy types."""
    return json.dumps(data, cls=NumpyEncoder)

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from simulation.drone import DronePhysics
from simulation.environment import Environment
from simulation.controller import LNNController, PIDController, WaypointMission

app = FastAPI(title="Adaptive Drone Navigation — LNN Simulation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Simulation State (shared across connections)
# ─────────────────────────────────────────────
class SimulationEngine:
    """Manages two simultaneous simulations: LNN vs PID."""
    
    def __init__(self):
        self.difficulty = 'normal'
        self.running = False
        self.paused = False
        self.reset()
    
    def reset(self, difficulty=None):
        if difficulty:
            self.difficulty = difficulty
        
        # Shared environment (same wind for both drones — fair comparison)
        self.env = Environment(seed=42)
        self.env.set_difficulty(self.difficulty)
        
        # LNN Drone
        self.lnn_drone = DronePhysics()
        self.lnn_drone.reset(x=0, y=0, z=10)
        self.lnn_controller = LNNController(hidden_size=64, num_layers=2)
        self.lnn_mission = WaypointMission()
        
        # PID Drone (starts at same position)
        self.pid_drone = DronePhysics()
        self.pid_drone.reset(x=0, y=0, z=10)
        self.pid_controller = PIDController()
        self.pid_mission = WaypointMission()
        
        # Metrics history
        self.lnn_errors = []
        self.pid_errors = []
        self.lnn_collisions = 0
        self.pid_collisions = 0
        self.timestamps = []
        self.wind_history = []
        self.step_count = 0
    
    def step(self) -> dict:
        """Advance simulation by one step and return state."""
        # Advance environment
        wind = self.env.step()
        
        # ── LNN step ──
        lnn_target = self.lnn_mission.current_target.copy()

        if self.lnn_mission.all_complete():
            # Mission done — pure PD hover at landing spot, no LNN noise
            lnn_status = "✓ Mission Complete — Holding Position"
            self.lnn_current_status = lnn_status
            lnn_thrusts = self.lnn_controller._base_control(self.lnn_drone, lnn_target)
        else:
            lnn_status = f"Navigating to W{self.lnn_mission.current_idx + 1}"
            # Check safe landing on last waypoint
            if self.lnn_mission.current_idx == len(self.lnn_mission.waypoints) - 1:
                if self.env.is_area_clear(lnn_target[0], lnn_target[1], radius=6.0):
                    lnn_status = "Landing Zone Secure - Descending"
                else:
                    lnn_target[2] = 10.0
                    lnn_status = "Landing Zone Blocked - Hovering"
            self.lnn_current_status = lnn_status
            lnn_obs = self.env.get_nearby_obstacles(
                self.lnn_drone.state.x, self.lnn_drone.state.y, self.lnn_drone.state.z, radius=15.0)
            lnn_thrusts = self.lnn_controller.compute_action(self.lnn_drone, lnn_target, wind, lnn_obs)
        self.lnn_drone.apply_control(lnn_thrusts, wind_force=wind)
        lnn_pos = np.array([self.lnn_drone.state.x, self.lnn_drone.state.y, self.lnn_drone.state.z])
        self.lnn_mission.update(lnn_pos)
        
        lnn_err = float(np.linalg.norm(lnn_target - lnn_pos))
        self.lnn_errors.append(lnn_err)
        
        lnn_cols = self.env.check_collisions(*lnn_pos)
        if lnn_cols:
            self.lnn_collisions += 1
            # Physical bounce
            for obs in lnn_cols:
                obs_pos = np.array([obs['x'], obs['y'], obs['z']])
                direction = lnn_pos - obs_pos
                dist = np.linalg.norm(direction)
                if dist > 0:
                    direction /= dist
                    # Reflect velocity and add a push
                    self.lnn_drone.state.vx += direction[0] * 5.0
                    self.lnn_drone.state.vy += direction[1] * 5.0
                    self.lnn_drone.state.vz += direction[2] * 5.0
        
        # ── PID step ──
        pid_target = self.pid_mission.current_target.copy()

        if self.pid_mission.all_complete():
            # Mission done — hold position
            pid_status = "✓ Mission Complete — Holding Position"
            self.pid_current_status = pid_status
            pid_thrusts = self.pid_controller.compute_action(self.pid_drone, pid_target, wind, None)
        else:
            pid_status = f"Navigating to W{self.pid_mission.current_idx + 1}"
            if self.pid_mission.current_idx == len(self.pid_mission.waypoints) - 1:
                if self.env.is_area_clear(pid_target[0], pid_target[1], radius=6.0):
                    pid_status = "Landing Zone Secure - Descending"
                else:
                    pid_target[2] = 10.0
                    pid_status = "Landing Zone Blocked - Hovering"
            self.pid_current_status = pid_status
            pid_obs = self.env.get_nearby_obstacles(
                self.pid_drone.state.x, self.pid_drone.state.y, self.pid_drone.state.z, radius=15.0)
            pid_thrusts = self.pid_controller.compute_action(self.pid_drone, pid_target, wind, pid_obs)
        self.pid_drone.apply_control(pid_thrusts, wind_force=wind)
        pid_pos = np.array([self.pid_drone.state.x, self.pid_drone.state.y, self.pid_drone.state.z])
        self.pid_mission.update(pid_pos)
        
        pid_err = float(np.linalg.norm(pid_target - pid_pos))
        self.pid_errors.append(pid_err)
        
        pid_cols = self.env.check_collisions(*pid_pos)
        if pid_cols:
            self.pid_collisions += 1
            # Physical bounce
            for obs in pid_cols:
                obs_pos = np.array([obs['x'], obs['y'], obs['z']])
                direction = pid_pos - obs_pos
                dist = np.linalg.norm(direction)
                if dist > 0:
                    direction /= dist
                    # Reflect velocity and add a push
                    self.pid_drone.state.vx += direction[0] * 5.0
                    self.pid_drone.state.vy += direction[1] * 5.0
                    self.pid_drone.state.vz += direction[2] * 5.0
        
        # Keep history trimmed
        MAX_HIST = 100
        if len(self.lnn_errors) > MAX_HIST:
            self.lnn_errors.pop(0)
            self.pid_errors.pop(0)
        
        self.wind_history.append(float(np.linalg.norm(wind)))
        if len(self.wind_history) > MAX_HIST:
            self.wind_history.pop(0)
        
        self.step_count += 1
        self.timestamps.append(round(self.env.time, 2))
        if len(self.timestamps) > MAX_HIST:
            self.timestamps.pop(0)
        
        # Build payload
        state = self.state
        return state
    
    @property
    def state(self) -> dict:
        """Current full simulation state as dict (for WebSocket)."""
        ls = self.lnn_drone.state
        ps = self.pid_drone.state
        env = self.env.get_state()
        
        lnn_target = self.lnn_mission.current_target.tolist()
        pid_target = self.pid_mission.current_target.tolist()
        
        return {
            "type": "state",
            "time": round(self.env.time, 2),
            "difficulty": self.difficulty,
            
            "lnn": {
                "pos": [round(ls.x, 2), round(ls.y, 2), round(ls.z, 2)],
                "vel": [round(ls.vx, 2), round(ls.vy, 2), round(ls.vz, 2)],
                "orientation": [round(ls.roll, 3), round(ls.pitch, 3), round(ls.yaw, 3)],
                "target": lnn_target,
                "waypoint": self.lnn_mission.current_idx + 1,
                "status": getattr(self, 'lnn_current_status', 'Initializing...'),
                "error": round(self.lnn_errors[-1] if self.lnn_errors else 0, 2),
                "avg_error": round(self.lnn_controller.avg_error, 2),
                "collisions": self.lnn_collisions,
                "crashes": self.lnn_drone.crashes,
                "path_x": [float(v) for v in ls.path_x[-50:]],
                "path_y": [float(v) for v in ls.path_y[-50:]],
                "path_z": [float(v) for v in ls.path_z[-50:]],
                "mission_progress": round(self.lnn_mission.progress * 100, 1),
                "params": self.lnn_controller.get_param_count(),
            },
            
            "pid": {
                "pos": [round(ps.x, 2), round(ps.y, 2), round(ps.z, 2)],
                "vel": [round(ps.vx, 2), round(ps.vy, 2), round(ps.vz, 2)],
                "orientation": [round(ps.roll, 3), round(ps.pitch, 3), round(ps.yaw, 3)],
                "target": pid_target,
                "waypoint": self.pid_mission.current_idx + 1,
                "status": getattr(self, 'pid_current_status', 'Initializing...'),
                "error": round(self.pid_errors[-1] if self.pid_errors else 0, 2),
                "avg_error": round(self.pid_controller.avg_error, 2),
                "collisions": self.pid_collisions,
                "crashes": self.pid_drone.crashes,
                "path_x": [float(v) for v in ps.path_x[-50:]],
                "path_y": [float(v) for v in ps.path_y[-50:]],
                "path_z": [float(v) for v in ps.path_z[-50:]],
                "mission_progress": round(self.pid_mission.progress * 100, 1),
                "params": self.pid_controller.get_param_count(),
            },
            
            "environment": env,
            "errors_history": {
                "lnn": [round(e, 2) for e in self.lnn_errors],
                "pid": [round(e, 2) for e in self.pid_errors],
                "timestamps": self.timestamps,
                "wind": [round(w, 2) for w in self.wind_history],
            },
            "waypoints": [wp.tolist() for wp in self.lnn_mission.waypoints],
        }


sim = SimulationEngine()

# ─────────────────────────────────────────────
# HTTP Endpoints
# ─────────────────────────────────────────────

@app.get("/")
async def index():
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    return FileResponse(os.path.abspath(frontend_path))

@app.post("/api/reset")
async def reset_simulation(difficulty: str = 'normal'):
    sim.reset(difficulty=difficulty)
    return {"status": "reset", "difficulty": difficulty}

@app.post("/api/difficulty/{level}")
async def set_difficulty(level: str):
    if level not in ['calm', 'normal', 'stormy', 'extreme']:
        return {"error": "Invalid difficulty"}
    sim.reset(difficulty=level)
    return {"status": "ok", "difficulty": level}

@app.get("/api/state")
async def get_state():
    return sim.state

# ─────────────────────────────────────────────
# WebSocket — Real-time streaming
# ─────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
    
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
    
    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
    
    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        for ws in self.active[:]:
            try:
                await ws.send_text(msg)
            except Exception:
                self.active.remove(ws)

manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                # Non-blocking receive for commands
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                cmd = json.loads(data)
                
                if cmd.get('action') == 'reset':
                    sim.reset(difficulty=cmd.get('difficulty', 'normal'))
                elif cmd.get('action') == 'pause':
                    sim.paused = not sim.paused
                elif cmd.get('action') == 'difficulty':
                    sim.reset(difficulty=cmd.get('level', 'normal'))
                    
            except asyncio.TimeoutError:
                pass
            
            if not sim.paused:
                state = sim.step()
                await websocket.send_text(safe_json(state))
            
            await asyncio.sleep(0.05)  # 20 Hz
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ─────────────────────────────────────────────
# Static files
# ─────────────────────────────────────────────
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    print("[LNN] Starting Adaptive Drone Navigation Server...")
    print("[LNN] Dashboard: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
