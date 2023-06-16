import numpy as np
import matplotlib.pyplot as plt

from TSPC.model import TSPCeuclidean

N = 30
nodes = np.random.random((N,2))

tsp = TSPCeuclidean(nodes, p =81, r = 3.6)

print(tsp)

tsp.find_t_nodes()
tsp.find_t_dist()
tsp.find_t_energy()

tsp.prepare_lkh()

tsp.solve_lkh()

fig, ax = tsp.plot(triangular=True)
plt.show()