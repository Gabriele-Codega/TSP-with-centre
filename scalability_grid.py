import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from tqdm import tqdm
    bar = True
except ImportError:
    print('Running with no progressbar')
    bar = False

import sys
import argparse

from TSPC.grid_model import TSPCgrid


def scalability_grid(nxmin = 4, nxmax = 8, nymin = 5, nymax = 11, n_exp = 10, K = 5, tau = 25):
    nxs = range(nxmin,nxmax+1)
    nys = range(nymin,nymax+1)
    I = range(n_exp)
    ts = np.zeros((len(nxs)+len(nys)-1,n_exp,2))

    if bar: pbar = tqdm(total=(len(nxs)+len(nys)-1)*n_exp)
    Ny = nys[0]
    j = 0
    for Nx in nxs:
        tsp = TSPCgrid(Nx,Nx,K)
        tsp.generate_nodes()

        for i in I:
            tsp.generate_edges()
            tsp.find_shortest_paths()
            tsp.find_t_paths()
            l,c,t = tsp.optimise_direct_path(tau,verbose=False)
            ts[j,i,0] = t

            l,c,t = tsp.optimise(tau,verbose=False)
            ts[j,i,1] = t

            if bar: pbar.update(1)
        j += 1

    Nx = nxs[-1]
    for Ny in nys[1:]:
        tsp = TSPCgrid(Nx,Ny,K)
        tsp.generate_nodes()

        for i in I:
            tsp.generate_edges()
            tsp.find_shortest_paths()
            tsp.find_t_paths()
            l,c,t = tsp.optimise_direct_path(tau,verbose=False)
            ts[j,i,0] = t

            l,c,t = tsp.optimise(tau,verbose=False)
            ts[j,i,1] = t
            if bar: pbar.update(1)
        j += 1

    if bar: pbar.close()

    cwd = Path.cwd()/'data'
    if not cwd.is_dir():
        cwd.mkdir()
    np.save('./data/grid_scalability.npy',ts)

    t_aver = np.mean(ts, axis=1)
    Ns = np.concatenate([[nys[0] * nx for nx in nxs],[nxs[-1] * ny for ny in nys[1:]]])
    fig,ax = plt.subplots()
    ax.plot(Ns,t_aver.T[0],'o', label = 'K=1')
    ax.plot(Ns,t_aver.T[1],'o', label = f'K={K:}')

    ax.legend()

    plt.show()




if __name__ == '__main__':
    if len(sys.argv) == 1:
        cont = input("Running the script with default values may take a good while.\n" + \
                    "Learn how to change the parameters by running '"+ sys.argv[0] + " -h '\n" + \
                    "Do you wish to continue? [y/n] ")
        if cont == 'n':
            sys.exit()
        else:
            scalability_grid()
    else:
        parser = argparse.ArgumentParser(description="Run a scalability test on grid network.\n" +
                                         "Given two lists Nxs and Nys, the problem size scales by fixing Ny = Nys[0] and varying Nx and then by fixing Nx = Nxs[-1] amd varying Ny")
        parser.add_argument('--nxmin', type = int, help='Minimum number of nodes along x',default=4)
        parser.add_argument('--nxmax', type = int, help='Maximum number of nodes along x',default=8)
        parser.add_argument('--nymin', type = int, help='Minimum number of nodes along y',default=5)
        parser.add_argument('--nymax', type = int, help='Maximum number of nodes along y',default=11)
        parser.add_argument('--n_exp', type = int, help='Number of instances for each size',default=10)
        parser.add_argument('-K', type = int, help='Number of triangular nodes to use',default = 5)
        parser.add_argument('--tau', type = float, help='Upper bound on L',default = 25)

        args = vars(parser.parse_args())
        scalability_grid(**args)
