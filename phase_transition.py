import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import argparse

from TSPC.model import TSPCeuclidean


def phase_transition_test(rmin = 0, rmax = 10, step = 0.2, size = 50, n_exp = 10, save_data = True):

    node_instances = np.random.random((n_exp,size,2))

    rs = np.arange(rmin,rmax+0.01,step)

    ls_lkh = np.zeros((node_instances.shape[0],rs.shape[0]))
    cs_lkh = np.zeros((node_instances.shape[0],rs.shape[0]))
    ls_grb = np.zeros((node_instances.shape[0],rs.shape[0]))
    cs_grb = np.zeros((node_instances.shape[0],rs.shape[0]))
    i = 0
    for nodes in tqdm(node_instances, desc='Instance number'):
        tsp = TSPCeuclidean(nodes,scale = 1000, p = 81)
        tsp.find_t_nodes()
        tsp.find_t_dist()
        j = 0
        for r in tqdm(rs,desc = 'r'):
            tsp.params = {'r': r}
            tsp.find_t_energy()

            # gurobi
            tsp.define_IP()
            l,c,t = tsp.solve_gurobi(verbose = False)

            ls_grb[i,j] = l
            cs_grb[i,j] = c

            # lkh
            tsp.prepare_lkh()

            l,c,t = tsp.solve_lkh(verbose=False)
            ls_lkh[i,j] = l
            cs_lkh[i,j] = c
            j += 1
        i += 1


    if save_data:
        cwd = Path.cwd()/'data'
        if not cwd.is_dir():
            cwd.mkdir()        
        of_llkh = "./data/l_lkh.dat"
        of_clkh = "./data/c_lkh.dat"
        of_lgrb = "./data/l_grb.dat"
        of_cgrb = "./data/c_grb.dat"
        try:
            np.save(of_llkh,ls_lkh)
            np.save(of_clkh,cs_lkh)
            np.save(of_lgrb,ls_grb)
            np.save(of_cgrb,cs_grb)
        except:
            print("Couldn't save data :(")

    l_aver_lkh = np.mean(ls_lkh,axis = 0)
    c_aver_lkh = np.mean(cs_lkh,axis = 0)
    l_aver_grb = np.mean(ls_grb,axis = 0)
    c_aver_grb = np.mean(cs_grb,axis = 0)


    fig,ax = plt.subplots()
    ax.scatter(rs,l_aver_lkh, color = 'navy',label = 'LKH')
    ax.scatter(rs,l_aver_grb, color = 'orange', s = 10, label = 'IP')
    ax.set_xlabel("r")
    ax.set_ylabel("L")
    ax.legend()
    plt.show()
    ## uncomment to automatically save the picture
    # fig.savefig("./pics/Lvr_ipvlkh.png")

    error = np.abs(l_aver_lkh-l_aver_grb)/l_aver_grb
    imaxerr = np.argmax(error)
    print(f"Maximum error is {error[imaxerr]:1.3f}, for r={rs[imaxerr]:2.1f}")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        cont = input("Running the script with default values may take a good while.\n" + \
                    "Learn how to change the parameters by running '"+ sys.argv[0] + " -h '\n" + \
                    "Do you wish to continue? [y/n] ")
        if cont == 'n':
            sys.exit()
        else:
            phase_transition_test()
    else:
        parser = argparse.ArgumentParser(description="Solve TSPC for different values of r to see its influence on the phase transition.")
        parser.add_argument('--rmin', type = float, help='Minimum value of r',default=0)
        parser.add_argument('--rmax', type = float, help='Maximum value of r',default=10)
        parser.add_argument('--step', type = float, help='Step between values of r',default=2)
        parser.add_argument('--size', type = int, help='Size of the problem to be solved (i.e. number of nodes)',default=50)
        parser.add_argument('--n_exp', type = int, help='Number of instances for each size',default=10)
        parser.add_argument('--save_data', type = int, help='Whether to save the data to a file. 1 for True, 0 for False',default=1)

        args = vars(parser.parse_args())
        phase_transition_test(**args)
