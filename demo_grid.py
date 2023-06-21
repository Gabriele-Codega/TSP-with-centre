import numpy as np
import matplotlib.pyplot as plt

from TSPC.grid_model import TSPCgrid

tsp = TSPCgrid(7,5,5)

print(tsp)

tsp.generate_nodes()
tsp.generate_edges()
tsp.find_shortest_paths()
tsp.find_t_paths()

print('Optimising with tpaths')
tsp.optimise(20)

print('Tour: ',tsp.tour)

print('Optimising with direct paths')
tsp.optimise_direct_path(20)

print('Tour: ',tsp.tour)
fig, ax = tsp.plot()
plt.show()
