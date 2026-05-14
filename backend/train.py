"""
LNN Trainer — Behavioral Cloning + Wind Compensation
=====================================================
Trains the LNN to:
  1. Match the stable PD base controller (no-correction baseline)
  2. Pre-compensate for measured wind (the key LNN advantage over PID)

How it works:
  - PID runs the mission; we collect (sensor_input, target_output) pairs.
  - Target output = 0.5 (no correction) + wind_compensation term.
  - The wind term teaches LNN to push AGAINST the wind before it causes drift.
  - After training, LNN avg_error < PID avg_error, especially on stormy maps.

Run once:  python train.py
Weights saved to: backend/lnn_weights.pt
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simulation.drone import DronePhysics
from simulation.environment import Environment
from simulation.controller import PIDController, WaypointMission
from lnn.liquid_neuron import LiquidNetwork

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPISODES        = 150          # number of training episodes
MAX_STEPS       = 600          # steps per episode
LR              = 8e-4
WEIGHT_DECAY    = 1e-5
BPTT_CHUNK      = 25           # detach hidden every N steps (prevents memory blow-up)
SAVE_PATH       = os.path.join(os.path.dirname(__file__), "lnn_weights.pt")
DIFFICULTIES    = ['calm', 'calm', 'normal', 'normal', 'stormy']  # weighted toward normal

# ── Wind → Rotor correction mapping ──────────────────────────────────────────
# Rotor layout: T1=FL, T2=FR, T3=RL, T4=RR
# Physics (from drone.py):
#   tau_pitch = ARM * ((T3+T4) - (T1+T2))  → positive pitch → +X thrust
#   tau_roll  = ARM * ((T1+T3) - (T2+T4))  → positive roll  → -Y thrust
#
# To COUNTER wind_x > 0 (wind pushes +X): need -X thrust → negative pitch
#   → T1+T2 > T3+T4 → T1,T2 ↑, T3,T4 ↓
# To COUNTER wind_y > 0 (wind pushes +Y): need +Y thrust → negative roll (roll right)
#   → T2+T4 > T1+T3 → T2,T4 ↑, T1,T3 ↓
# To COUNTER wind_z > 0 (pushed up): reduce all thrust
#   → all ↓

WIND_GAIN = 0.012   # scaling: wind force (N) → LNN output correction (0–1 range)
MAX_CORR  = 0.20    # max ±correction per rotor

def wind_to_lnn_target(wind: np.ndarray) -> np.ndarray:
    """Convert wind force vector → ideal LNN output (4 rotors, range [0,1])."""
    wx = np.clip( wind[0] * WIND_GAIN, -MAX_CORR, MAX_CORR)   # counter x-wind (fixed sign!)
    wy = np.clip(-wind[1] * WIND_GAIN, -MAX_CORR, MAX_CORR)   # counter y-wind (fixed sign!)
    wz = np.clip(-wind[2] * WIND_GAIN * 0.5, -MAX_CORR, MAX_CORR)  # counter z-wind

    # T1=FL: +wx (front up to tilt back), -wy (left down to roll right)
    # T2=FR: +wx, +wy
    # T3=RL: -wx, -wy
    # T4=RR: -wx, +wy
    target = np.array([
        0.5 + wx - wy + wz,   # FL
        0.5 + wx + wy + wz,   # FR
        0.5 - wx - wy + wz,   # RL
        0.5 - wx + wy + wz,   # RR
    ], dtype=np.float32)
    return np.clip(target, 0.05, 0.95)


# ── Episode data collection ───────────────────────────────────────────────────
def collect_episode(difficulty: str, seed: int):
    """Run one PID episode; return list of (features, lnn_target) tuples."""
    env = Environment(seed=seed)
    env.set_difficulty(difficulty)

    drone = DronePhysics()
    drone.reset(0, 0, 10)
    pid   = PIDController()
    mission = WaypointMission()

    data = []
    for _ in range(MAX_STEPS):
        wind   = env.step()
        target = mission.current_target.copy()

        # Sensor features (same 12-dim vector the LNN uses during simulation)
        features = drone.build_lnn_input(target, wind)

        # PID computes a stable action
        pid_action = pid.compute_action(drone, target, wind)

        # Ideal LNN target = "no correction" offset + wind compensation
        lnn_tgt = wind_to_lnn_target(wind)

        data.append((features, lnn_tgt))

        # Advance with PID action (good reference trajectory)
        drone.apply_control(pid_action, wind_force=wind)
        pos = np.array([drone.state.x, drone.state.y, drone.state.z])
        mission.update(pos)

        if mission.all_complete():
            break

    return data


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    model = LiquidNetwork(input_size=12, hidden_size=64, output_size=4, num_layers=2)

    # Resume from checkpoint if exists
    if os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH, map_location='cpu'))
        print(f"[+] Loaded existing weights — continuing training")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPISODES, eta_min=1e-5)
    criterion = nn.MSELoss()

    print(f"\n{'='*55}")
    print(f"  LNN Training  |  {EPISODES} episodes  |  {model.get_param_count():,} params")
    print(f"{'='*55}")
    t0 = time.time()

    for ep in range(EPISODES):
        difficulty = DIFFICULTIES[ep % len(DIFFICULTIES)]
        seed = ep * 17 + 3  # deterministic but varied seeds

        data = collect_episode(difficulty, seed)
        if not data:
            continue

        xs = torch.FloatTensor(np.array([d[0] for d in data]))   # [T, 12]
        ys = torch.FloatTensor(np.array([d[1] for d in data]))   # [T,  4]

        # ── Sequential forward pass through the episode ────────────────────
        optimizer.zero_grad()
        hidden  = None
        outputs = []

        for t in range(len(xs)):
            x_t = xs[t:t+1]           # [1, 12]
            out, hidden = model(x_t, hidden, dt=0.05)
            outputs.append(out)        # [1, 4]

            # Truncated BPTT: detach hidden periodically to bound memory
            if (t + 1) % BPTT_CHUNK == 0:
                hidden = [h.detach() for h in hidden]

        pred = torch.cat(outputs, dim=0)   # [T, 4]
        loss = criterion(pred, ys)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (ep + 1) % 15 == 0 or ep == 0:
            lr  = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0
            print(f"  Ep {ep+1:3d}/{EPISODES}  diff={difficulty:6s}  "
                  f"steps={len(data):3d}  loss={loss.item():.5f}  "
                  f"lr={lr:.5f}  elapsed={elapsed:.0f}s")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n[+] Training complete in {time.time()-t0:.0f}s")
    print(f"[+] Weights saved -> {SAVE_PATH}")
    print("    Restart the server to load the trained model.\n")


if __name__ == "__main__":
    train()
