import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import gurobipy as gp

from . import metrics
from . import utils



class TSPCeuclidean:
    def __init__(self,nodes, e = metrics.E, d = metrics.da, K = 5, scale = 100 ,**kwargs) -> None:
        """
        Initialise a model object on euclidean graph.

        Parameters
        ----------
        nodes : array of shape (N,2)
            coordinates of the nodes
        e : function, optional
            one of the energy functions defined in metrics.py, by default metrics.E.
        d : function, optional
            one of the edge distances defined in metrics.py, by default metrics.da
        K : int, optional
            number of triangular paths, by default 5
        scale : int, optional
            factor that multiplies E when writing the distance matrix for LKH, by default 100
        """
        self.nodes = nodes
        self.scale = scale
        self.size = nodes.shape[0]

        self.K = K

        self.energy = e
        self.distance = d

        self._params = {}
        self.params = kwargs

        self.tour = None
        self.used_nodes = []
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self,params):
        for key, value in params.items():
            self._params[key] = value

    def find_t_nodes(self,**kwargs):
        """
        Finds the triangular nodes for each pair of nodes.
        Builds:
            - _t_nodes: array of shape (N,N,K,2) with coordinates for the K tnodes per each pair of real nodes
            - _tl: array of shape (N,N,K) with the lengths of the K triangular paths
        """
        K = self.K
        N = self.size
        self._t_nodes = np.zeros((N,N,K,2))
        self._tl = np.zeros((N,N,K))
        for i in range(N):
            for j in range(i,N):
                temp = utils.find_t_nodes(self.nodes[i],self.nodes[j],K=K)
                self._t_nodes[i,j] = temp[:,:2]
                self._tl[i,j] = temp[:,2]
                ## lower half
                self._t_nodes[j,i] = self._t_nodes[i,j]
                self._tl[j,i] = self._tl[i,j]
        
    def find_t_dist(self,**kwargs):
        """
        Builds:
            - _tc: an array of shape (N,N,K) with the distance from the centre for triangular paths between pairs of nodes
        """
        K = self.K
        N = self.size
        self._tc = np.zeros((N,N,K))
        for i in range(N):
            for j in range(i,N):
                for k in range(K):
                    self._tc[i,j,k] = metrics.t_path_dist(self.nodes[i],self.nodes[j],self._t_nodes[i,j,k,:2])
                    ## lower half
                    self._tc[j,i,k] = self._tc[i,j,k]

    def find_t_energy(self,**kwargs):
        """
        Computes the energy matrix _te by calling the model's energy function.
        Requires that `_tl` and `_tc` have been built. This can be done by calling `find_t_nodes` and `find_t_dist`
        """
        self._te = self.energy(self._tl,self._tc,**self.params)

    def find_t_energy_min(self,**kwargs):
        """
        For each pair of nodes, finds the minimum energy of the corresponding triangular paths.
        """
        self._Emin = np.min(self._te,axis=2)
        self._imin = np.argmin(self._te,axis=2).reshape((self.size,self.size,1))

    def find_t_len_min(self,**kwarg):
        """
        For each pair of nodes, finds the length of the minimum-energy triangular path between them.
        Requires a call to `find_t_energy_min`.
        """
        self._Lmin = np.take_along_axis(self._tl,self._imin,axis=2).reshape((self.size,self.size))

    def find_t_dist_min(self,**kwarg):
        """
        For each pair of nodes, finds the distance from centre of the minimum-energy triangular path between them.
        Requires a call to `find_t_energy_min`.
        """
        self._Cmin = np.take_along_axis(self._tc,self._imin,axis=2).reshape((self.size,self.size))

    def find_t_nodes_min(self,**kwargs):
        """
        For each pair of nodes, finds the coordinates of the tnode corresponding to the minimum-energy triangular path between them.
        Requires a call to `find_t_energy_min`.
        """
        self._imin = self._imin.reshape((self.size,self.size,1,1))
        self._t_nodesmin = np.take_along_axis(self._t_nodes,self._imin,axis = 2).reshape((self.size,self.size,2))


    def compute_L(self):
        """
        Computes a distance matrix L using euclidean distance.
        Ignores triangular nodes.
        """
        self.L = distance.cdist(self.nodes,self.nodes,metric='euclidean')

    def compute_C(self):
        """
        Computes a distance matrix C using the distance from centre for each edge.
        Ignores triangular nodes.
        """
        self.C = distance.cdist(self.nodes,self.nodes,metric = self.distance,**self.params)

    def compute_E(self):
        """
        Computes an energy matrix E using the object's energy function.
        Ignores triangular nodes.
        """
        self.E = self.energy(self.L,self.C,**self.params)

    def compute_metrics(self):
        """
        Computes L, C, E.
        Ignores triangular nodes.
        """
        self.compute_L()
        self.compute_C()
        self.compute_E()

    def prepare_lkh(self, tpath = True, **kwargs):
        """
        Computes L, C, E for an LKH run.
        Requires `_tl`,`_tc`,`_te`, which can be computed with `find_t_nodes`,`find_t_dist`,`find_t_energy`.
        """
        if tpath:
            self.find_t_energy_min()
            self.find_t_len_min()
            self.find_t_dist_min()
            self.find_t_nodes_min()
        
            self.E = self._Emin
            self.L = self._Lmin
            self.C = self._Cmin
            self.used_nodes = self._t_nodesmin
        else:
            self.compute_metrics()

        self.write_par()

    def write_par(self):
        """
        Writes the parameters for LKH on a "tspc.tsp" file. Has to be called before solving.
        """
        nodes_df = pd.DataFrame({'labels': list(range(1,self.size+1)),'x' : self.nodes[:,0],'y' : self.nodes[:,1]})
        outf = open("tspc.tsp",mode = "w")
        print("NAME: TSPC \n",
                "TYPE: TSP \n",
                f"DIMENSION: {self.size:} \n",
                "EDGE_WEIGHT_TYPE: EXPLICIT \n",
                "EDGE_WEIGHT_FORMAT: FULL_MATRIX\n",
                "NODE_COORD_TYPE: TWOD_COORDS \n",
                "DISPLAY_DATA_TYPE: COORD_DISPLAY\n",
                "\n",
                "NODE_COORD_SECTION",
                sep = "",
                file = outf)
        nodes_df.to_csv(outf,sep= " ",header=False,index = False)
        print("\nEDGE_WEIGHT_SECTION",file=outf)
        np.savetxt(outf, (self.scale*self.E).astype(int),fmt='%d') ### Emin and E are different things rn
        print("EOF",file = outf)
        outf.close()

    def solve_lkh(self, verbose = True):
        """
        Runs LKH, retrieves optimal path and computes total length and distance from center.

        Parameters
        ----------
        verbose: bool
            if `True` the function will print a message before and after completion of LKH. `True` by default.

        Returns
        -------
        length: float
            length of the tour
        dist_from_c: float
            distance from the centre of the tour
        elapsed_time: float
            time required by LKH
        """
        ## run the actual algorithm
        if verbose : print("Running LKH")
        subprocess.run("./run_lkh.sh")
        if verbose : print("LKH done.\n")

        ## read the best tour
        inf = open("tspc_tour.dat",'r')
        lines = inf.readlines()
        i = lines.index("TOUR_SECTION\n")+1
        tour = [int(line.strip())-1 for line in lines[i:i+self.size]]
        tour.append(tour[0])
        inf.close()
        self.tour = tour

        ## length and distance from center for the tour
        length = 0
        dist_from_c = 0
        for i in range(len(self.tour)-1):
            length += self.L[self.tour[i],self.tour[i+1]]
            dist_from_c += self.C[self.tour[i],self.tour[i+1]]

        ## lkh execution time
        with open("lkh.log",'r') as f:
            line = f.readlines()[-1]
        i = line.index("=")
        j = line.index("sec.")
        elapsed_time = float(line[i+1:j].strip())
        
        return length,dist_from_c,elapsed_time

    def prepare_gurobi(self,tau = None,objective = 'E', **kwargs):
        """
        Prepares to run Gurobi optimisation.
        Computes all required metrics and defines an IP.

        Parameters
        ----------
        tau: float,optional
            an upper bound to the length of the tour, by default None. If none, the problem is unconstraied.
        objective : str, optional
            specifier for the objective function, by default 'E'. Other possible value is 'C'.
 
        """
        self.find_t_nodes()
        self.find_t_dist()
        self.find_t_energy()

        self.define_IP(tau,objective = 'E')

    def define_IP(self, tau = None,objective = 'E', maxtime = 120, gap = 1e-4, **kwargs):
        """
        Defines the IP problem for Gurobi.

        Note: to solve an IP with direct paths, the object's K property should be set to 1.

        Parameters
        ----------
        tau : float, optional
            upper bound on the total length, by default None. If None, the unconstrained problem is defined
        objective : str, optional
            specifier for the objective function, by default 'E'. Other possible value is 'C'.
        maxtime: int, optional
            time limit, in seconds. By default 120
        gap: float, optional
            relative gap at which the search for an optimum stops. By default 1e-4

        Raises
        ------
        ValueError
            unrecognised objective function. 
        """
        K = self.K
        N = self.size
        I = range(N)
        J = range(N)

        self._m = gp.Model()
        self._m.ModelSense = gp.GRB.MINIMIZE

        x = self._m.addVars([(i,j,k) for i in I for j in J for k in range(K)], vtype = gp.GRB.BINARY)

        # visit each node once
        for i in I:
            self._m.addConstr(gp.quicksum(x[i,j,k] for j in J for k in range(K)) == 1) 
            self._m.addConstr(gp.quicksum(x[j,i,k] for j in I for k in range(K)) == 1) 
        
        # constraint of total length
        if not (tau is None):
            self._m.addConstr(gp.quicksum(self._tl[i,j,k]* x[i,j,k] for i in I for j in J for k in range(K)) <= tau)

        # set the objective
        if objective == 'E':
            self._m.setObjective(gp.quicksum(self._te[i,j,k]*x[i,j,k] for i in I for j in J for k in range(K)))
        elif objective == 'C':
            self._m.setObjective(gp.quicksum(self._tc[i,j,k]*x[i,j,k] for i in I for j in J for k in range(K)))
        else:
            raise ValueError('Objective must be either E or C.')

        # internal variables useful for the subtour elimination
        self._m._vars = x
        self._m._K = K
        self._m._size = N

        # suppress outputs, set stopping criteria and allow lazy constraints
        self._m.setParam(gp.GRB.Param.OutputFlag,0)
        self._m.Params.MIPGap = gap
        self._m.Params.TimeLimit = maxtime
        self._m.Params.lazyConstraints = 1

    def solve_gurobi(self, verbose = True):
        """
        Runs Gurobi optimiser.
        Saves the optimal tour and the used tnode for each edge. 

        Parameters
        ----------
        verbose: bool
            if `True` the function will print a message before and after completion of Gurobi `optimize()`. `True` by default.

        Returns
        -------
        length: float
            length of the tour
        dist_from_c: float
            distance from the centre of the tour
        runtime: float
            time required by optimisation
        """
       
        if verbose: print("Running Gurobi optimisation...")
        self._m.optimize(utils.subtourelim)
        if verbose: print("Gurobi done.\n")

        ## save the tour
        arcs = gp.tuplelist((i,j,k) for i,j,k in self._m._vars if self._m._vars[i,j,k].X > 0.5)
        tour = []
        tour.append(arcs[0][0])

        for i in range(1,len(arcs)):
            tour.append(arcs[tour[i-1]][1])
        tour.append(tour[0])
        self.tour = tour

        ## save tnodes and lengths
        self.used_nodes = np.zeros((self.size,self.size,2))
        length = 0
        dist_from_c = 0
        for i,j,k in arcs:
            self.used_nodes[i,j] = self._t_nodes[i,j,k]
            length += self._tl[i,j,k]
            dist_from_c += self._tc[i,j,k]

        return length, dist_from_c, self._m.Runtime



    def plot(self,triangular = False):
        """
        Plots the graph with the current stored path.

        Parameters
        ----------
        triancular: bool, optional
            if True, the triangular paths are plotted. By default is False.

        Returns
        -------
        fig: 
            matplotlib figure object
        ax:
            matplotlib axis object
        """
        fig,ax = plt.subplots()
        if triangular:
            pts = self.nodes.T
            ax.plot(0.5,0.5,'o',color = 'black',label="Center")

            path = []
            for i in range(len(self.tour)-1):
                path.append(self.nodes[self.tour[i]])
                path.append(self.used_nodes[self.tour[i],self.tour[i+1]])
            path.append(path[0])
            path = np.array(path)
            ax.scatter(path.T[0],path.T[1],color='y',label="Triangular nodes")
            ax.plot(pts[0],pts[1],'o', color = 'red',label="Destination nodes")
            ax.plot(path.T[0],path.T[1])
            ax.legend()
        else:
            pts = self.nodes.T
            ax.scatter(0.5,0.5,color = 'black',label = "Center")
            ax.scatter(pts[0],pts[1],color = 'red',label="Destination nodes")
            if self.tour is not None:
                ax.plot(pts[0,self.tour],pts[1,self.tour])
        return fig,ax

    def __str__(self) -> str:
        outstr = "Travelling Salesman Problem with a center.\n" + \
            f"Size: {self.size:}\n" + \
            "Distance function: "+self.distance.__name__+"\n" + \
            "Energy function: "+self.energy.__name__+"\n"+ \
            f"Number of triangular paths: {self.K:}\n" + \
            f"Scale: {self.scale}\n" + \
            "Other parameters: "+str(self.params)+"\n"
        return outstr  
