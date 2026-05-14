from backend.main import SimulationEngine

sim = SimulationEngine()
sim.reset(difficulty='calm')

for i in range(1000):
    state = sim.step()
    if i % 100 == 0:
        print(f"T+{state['time']:.1f}s | LNN Pos: {state['lnn']['pos']} | LNN Target: {state['lnn']['target']} | LNN Error: {state['lnn']['error']} | Prog: {state['lnn']['mission_progress']}")
    if state['lnn']['mission_progress'] >= 100:
        break
