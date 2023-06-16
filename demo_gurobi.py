import numpy as np
import matplotlib.pyplot as plt

from TSPC.model import TSPCeuclidean

N = 30
nodes = np.random.random((N,2))

tsp = TSPCeuclidean(nodes, p =81, r = 3.6)

print(tsp)

tsp.prepare_gurobi()

tsp.solve_gurobi()

fig, ax = tsp.plot(triangular=True)
plt.show()