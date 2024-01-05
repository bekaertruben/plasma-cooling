from wrapper_sim import Simulation

import warnings
warnings.filterwarnings("error")

sim = Simulation(iterations=100)
sim.begin(1, 1, number_of_saves=100, fields="pic", gamma_drag={"syn": 1})
sim.run()
sim.end(name="test")


print(sim.pos_history.shape)