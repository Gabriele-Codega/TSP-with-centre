import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    bar = True
except ImportError:
    print('Running with no progressbar')
    bar = False

from pathlib import Path
import time
import sys
import argparse

from TSPC.model import TSPCeuclidean


def scalability_euclidean(minsize = 10, maxsize = 80, step = 10, ks = [1,5], n_exp = 10,save_data = True):
    Ns = list(range(minsize,maxsize+1,step))

    t_aver = []
    if bar: pbar = tqdm(total=len(Ns)*len(ks)*n_exp*3)
    for N in Ns:

        node_instances = np.random.random((n_exp,N,2))

        ## compute the lower bound on l as the solution to tspc with r=0
        llos = []
        for nodes in node_instances:
            tsp = TSPCeuclidean(nodes,r=0,p=81)
            tsp.compute_metrics()
            tsp.write_par()
            l,c,t = tsp.solve_lkh(verbose=False)
            llos.append(l)
        llo = np.mean(llos)

        taus = [1.5*llo, 4*llo, 7*llo]


        ls = np.zeros((len(ks),node_instances.shape[0],len(taus)))
        cs = np.zeros((len(ks),node_instances.shape[0],len(taus)))
        ts = np.zeros((len(ks),node_instances.shape[0],len(taus)))
        prets = np.zeros((len(ks),node_instances.shape[0]))

        i = 0
        for K in ks:
            j = 0
            for nodes in node_instances:
                tsp = TSPCeuclidean(nodes, K = K, p=81)

                ti = time.monotonic()
                tsp.find_t_nodes()
                tsp.find_t_dist()
                tf = time.monotonic()
                prets[i,j] = tf-ti

                k = 0
                for tau in taus:
                    tsp.define_IP(tau=tau,objective='C')
                    l,c,t = tsp.solve_gurobi(verbose=False)
                    ls[i,j,k] = l
                    cs[i,j,k] = c
                    ts[i,j,k] = t
                    if bar: pbar.update(1)
                    k += 1

                j += 1
            i += 1

        t_aver.append(np.mean(ts,axis=1))

        if save_data:
            cwd = Path.cwd()/'data'
            if not cwd.is_dir():
                cwd.mkdir()
            file1 = f"./data/scalability_N{N:}_L.npy"
            file2 = f"./data/scalability_N{N:}_C.npy"
            file3 = f"./data/scalability_N{N:}_t.npy"
            file4 = f"./data/scalability_N{N:}_tpre.npy"
            try:
                np.save(file1,ls)
                np.save(file2,cs)
                np.save(file3,ts)
                np.save(file4,prets)
            except:
                print("Couldn't save data :(")

    if bar: pbar.close()

    t_aver = np.array(t_aver).reshape((len(Ns),len(ks),3))
    axs = []
    fig,axs = plt.subplots(1,3)
    for i,k in zip(range(len(ks)),ks):
        axs[0].plot(Ns,t_aver[:,i,0],'o', label = f'K={k:}')
        axs[1].plot(Ns,t_aver[:,i,1],'o', label = f'K={k:}')
        axs[2].plot(Ns,t_aver[:,i,2],'o', label = f'K={k:}')
    for i in range(len(axs)):
        axs[i].legend()

    fig.tight_layout()
    plt.show()




if __name__ == '__main__':
    if len(sys.argv) == 1:
        cont = input("Running the script with default values may take a good while.\n" + \
                    "Learn how to change the parameters by running '"+ sys.argv[0] + " -h '\n" + \
                    "Do you wish to continue? [y/n] ")
        if cont == 'n':
            sys.exit()
        else:
            scalability_euclidean()
    else:
        parser = argparse.ArgumentParser(description="Run a scalability test on euclidean graph.")
        parser.add_argument('-m', '--minsize', type = int, help='Minimum problem size',default=10)
        parser.add_argument('-M', '--maxsize', type = int, help='Maximum problem size',default=80)
        parser.add_argument('-s', '--step', type = int, help='Step between problem sizes',default=10)
        parser.add_argument('-ks', type = int, nargs='*', help='Values of K to be tested',default=[1,5])
        parser.add_argument('--n_exp', type = int, help='Number of instances for each size',default=10)
        parser.add_argument('--save_data', type = int, help='Whether to save the data to a file. 1 for True, 0 for False',default=1)

        args = vars(parser.parse_args())
        scalability_euclidean(**args)
