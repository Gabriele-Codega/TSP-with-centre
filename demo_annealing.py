import numpy as np
import time
import sys, argparse

from TSPC.model import TSPCeuclidean
from TSPC.annealing import SimulatedAnnealing

def main(N = 30, r = 6):
    nodes = np.random.random((N,2))

    tsp = TSPCeuclidean(nodes, r=r, K = 5)

    tsp.prepare_gurobi()
    tsp.prepare_lkh(tpath=True)

    sa = SimulatedAnnealing(10000,tf=1e-6,tsp = tsp,triangular=True)

    print('Running annealing...')
    ti = time.monotonic()
    sa.simulate()
    t = time.monotonic()-ti
    print('Annealing done.')
    print(f'Simulated annealing: E = {sa.energy:.3f}, t = {t:.3f} s')

    l,c,t = tsp.solve_gurobi()
    print(f'Gurobi IP: E = {l+ r * c:.3f}, t = {t:.3f} s')

    l,c,t = tsp.solve_lkh()
    print(f'LKH: E = {l+ r * c:.3f}, t = {t:.3f} s')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Running with default parameters: N = 30, r = 6.\n")
        main()
    else:
        parser = argparse.ArgumentParser(description="Compare effectiveness and efficiency of annealing, Gurobi and LKH")
        parser.add_argument('-N', type = int, help='Problem size',default=30)
        parser.add_argument('-r', type = float, help='Parameter for the energy function (E = L + r*C)',default=6)
        
        args = vars(parser.parse_args())
        main(**args)

