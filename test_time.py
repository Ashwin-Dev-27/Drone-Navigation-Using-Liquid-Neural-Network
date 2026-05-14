from backend.main import SimulationEngine

sim = SimulationEngine()
sim.reset(difficulty='normal')

for i in range(10000):
    state = sim.step()
    if len(sim.lnn_mission.reached) == len(sim.lnn_mission.waypoints):
        print(f"Mission complete at T+{state['time']:.1f} seconds")
        break
