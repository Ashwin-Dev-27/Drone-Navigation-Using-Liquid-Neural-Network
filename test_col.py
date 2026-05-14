from backend.main import SimulationEngine
from backend.simulation.environment import Obstacle
sim = SimulationEngine()
sim.env.obstacles.append(Obstacle(0, 0, 10, radius=5)) # drone starts at 0,0,10
state = sim.step()
print("LNN Collisions:", state['lnn']['collisions'])
