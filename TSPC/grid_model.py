import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pth
from itertools import combinations
import gurobipy as gp

from . import utils

class TSPCgrid:
    def __init__(self,Nx,Ny,K) -> None:
        self.Nx = Nx
        self.Ny = Ny
        self.size = Nx*Ny

        self.hH = 1/(Nx-1)
        self.hV = 1/(Ny-1)
        self.hD = np.sqrt(self.hH**2 + self.hV**2)

        self.num_tpaths = K


    def generate_nodes(self):
        """
        Genetares nodes and assigns a random centre.
        """
        Nx = self.Nx
        Ny = self.Ny

        xs = np.linspace(0,1,Nx)
        ys = np.linspace(0,1,Ny)
        X,Y = np.meshgrid(xs,ys)
        self.nodes = np.vstack([X.ravel(),Y.ravel()]).T
        self.labels = list(range(Nx*Ny))
        self.c_label = np.random.choice(self.labels)
        self.center = self.nodes[self.c_label]


    def generate_edges(self):
        """
        Generates the edges of the graph according to some predefined probabilities.
        Builds and adjecency matrix `edges` and an adjecency list `adj`.
        """
        Nx = self.Nx
        Ny = self.Ny
        hH = self.hH
        hV = self.hV
        hD = self.hD
        labels = self.labels

        edges = np.full((Nx*Ny,Nx*Ny),np.inf)

        valid = False
        while valid is False:
            for i,j in combinations(labels,2):
                xi, yi = i%Nx, i//Nx
                xj, yj = j%Nx, j//Nx
                dx = np.abs(xi - xj)
                dy = np.abs(yi - yj)
                if (dx <= 1) and (dy <= 1):
                    if dx + dy > 1:
                        r = np.random.random(1) 
                        if r < 0.5:
                            edges[i,j] = hD
                    elif dy == 0:
                        r = np.random.random(1)
                        if r < 0.9:
                            edges[i,j] = hV
                    else:
                        r = np.random.random(1)
                        if r < 0.9:
                            edges[i,j] = hH

            for i in range(self.size-1):
                for j in range(i+1,self.size):
                    edges[j,i] = edges[i,j]
            
            invalid_row = [np.allclose(row,np.inf) for row in edges]
            if True in invalid_row:
                valid = False
            else:
                valid = True

        np.fill_diagonal(edges,0)

        adj = [[] for _ in range(Nx*Ny)]
        for src,dest in combinations(labels,2):
            w = edges[src,dest]
            if w < np.inf:
                adj[src].append((w,dest))
                adj[dest].append((w,src))

        self.edges = edges
        self.adj = adj


    def find_shortest_paths(self):
        """
        Finds shortest paths (in the sense of TSPC) between each pair of nodes.

        Builds:
            - (N,N) matrix L with path length
            - (N,N) matrix P of lists of predecessors in the shortest paths
            - (N,N) matrix PE of lists of edges traversed in each shortest path
            - (N,N) matrix C with the minimum distance from the center of shortest paths between each pair of nodes
            - (N,N) matrix shortest_path with the index of the shortest path between two nodes (used to access PE)
        """
        labels = self.labels
        c = self.c_label
        size = self.size

        L = np.zeros((size,size)) ## length of shortest path
        P = np.zeros_like(L,dtype = list) ## parents
        PE = np.zeros_like(L,dtype = list) ## lists of edges for each shortest path


        for s in labels:
            d, p = utils.dijkstra(self.adj,labels,s)
            L[s] = d
            P[s] = p

        for i in range(size):
            for j in range(size):
                path = []
                paths = []
                utils.find_paths(i,j,path,paths,P[i])
                path_edges = []
                for path in paths:
                    ed = [(path[i],path[i+1]) for i in range(len(path)-1)]
                    path_edges.append(ed)
                PE[i,j] = path_edges


        ## quantities for the average distance of each edge
        d = np.zeros((size,size))
        mx = np.zeros((size,size))
        mi = np.zeros((size,size))
        g = np.zeros((size,size))

        for k in range(size):
            for l in range(size):
                d[k,l] = (L[k,l]-np.abs(L[k,c]-L[l,c]))*0.5
                mx[k,l] = np.max([L[k,c],L[l,c]])
                mi[k,l] = np.min([L[k,c],L[l,c]])
                if np.abs(L[k,c]-L[l,c]) >= L[k,l]:
                    g[k,l] = mi[k,l] + L[k,l]*0.5
                else:
                    g[k,l] = (d[k,l]*(mx[k,l]+d[k,l]*0.5) + (L[k,l]-d[k,l])*(mi[k,l]+(L[k,l]-d[k,l])*0.5))/L[k,l]


        C = np.zeros((size,size)) ## min distance from center among all the shortest paths
        shortest_path = np.zeros((size,size)) ## shortest path (the one with minimum C)

        for i in range(size):
            for j in range(size):
                if j != i:
                    Cs = []
                    for path in PE[i,j]:
                        CC = 0
                        for k,l in path:
                            CC += g[k,l]*L[k,l]
                        Cs.append(CC)
                    idx = np.argmin(Cs)
                    C[i,j] = (np.min(Cs))/L[i,j]
                    shortest_path[i,j] = idx
        for i in range(size):
            C[i,i] = L[i,c]

        self.L = L
        self.P = P
        self.PE = PE
        self.C = C
        self.shortest_path = shortest_path

    def find_t_paths(self):
        """
        Find the triangular paths for each pair of nodes.
        Requires L and C to be defined for the model. These can be computed with `find_shortest_paths`.

        Builds:
          - an array of shape (N,N,K) tL with the path lengths
          - an array of shape (N,N,K) tC with the path distance from centre
          - an array of shape (N,N,K) tnodes with the triangular nodes for each pair of nodes
          - a matrix feasible_t of dictionaries with feasible tnodes and respective C
        """
        K = self.num_tpaths
        labels = self.labels
        L = self.L
        C = self.C
        c = self.c_label
        size = self.size

        feasible_t = np.zeros((size,size),dtype=dict) ## feasible triangular nodes for each pair of nodes
        for i in range(size):
            for j in range(size):
                feasible = {}
                for t in range(len(labels)):
                    if t==i or t==j:
                        feasible[t] = C[i,j]
                    else:
                        c_dist = (C[i,t]*L[i,t]+C[t,j]*L[t,j])/(L[i,t]+L[t,j])
                        if L[i,t]+L[t,j] <= L[i,c]+L[c,j] and  c_dist <= C[i,j]:
                            feasible[t] = c_dist
                feasible_t[i,j] = dict(sorted(feasible.items(), key = lambda item:item[1]))

        tL = np.full((size,size,K),np.inf) ## tpahth length
        tC = np.full((size,size,K),np.inf) ## tpath distance from center
        tnodes = np.full((size,size,K),None) ## tnodes for each pair of nodes

        for i in range(size):
            for j in range(size):
                keys = list(feasible_t[i,j].keys())
                l = len(keys)
                if l <= K:
                    keep = keys
                else:
                    keep = [keys[0],keys[-1]]
                    for k in range(3,K+1):
                        node = keys[(1+(k-2)*int(np.floor(l/K)))-1]
                        keep.append(node)
                length = []
                dist = []
                for node in keep:
                    length.append(L[i,node]+L[node,j])
                    dist.append(feasible_t[i,j][node])
                for k in range(len(keep)):
                    tL[i,j,k] = length[k]
                    tC[i,j,k] = dist[k]
                    tnodes[i,j,k] = keep[k]

            self.tL = tL
            self.tC = tC
            self.tnodes = tnodes
            self.feasible_t = feasible_t

    def optimise(self, tau,verbose = True):
        """Define the constrained Integer Program with triangular nodes, solve using Gurobi and save the tour.

        Parameters
        ----------
        tau : float
            Upper bound on the total length
        verbose: bool
            if True, print a message before and after optimisation. True by default

        Returns
        -------
        length: float
            Total length of the tour
        dist_from_c: float
            Total distance from the center
        runtime: float
            Total runtime for Gurobi optimisation
        """
        N = self.size
        I = range(N)
        J = range(N)
        K = self.num_tpaths

        model = gp.Model()
        model.ModelSense = gp.GRB.MINIMIZE

        x = model.addVars([(i,j,k) for i in I for j in J for k in range(K)], vtype = gp.GRB.BINARY)
        for i in I:
            model.addConstr(gp.quicksum(x[i,j,k] for j in J for k in range(K)) == 1) 
            model.addConstr(gp.quicksum(x[j,i,k] for j in I for k in range(K)) == 1) 

        model.addConstr(gp.quicksum(self.tL[i,j,k]* x[i,j,k] for i in I for j in J for k in range(K) if self.tL[i,j,k]!=np.inf) <= tau,name='Lbound')

        for i in I:
            for j in J:
                for k in range(K):
                    if self.tnodes[i,j,k] is None:
                        model.addConstr(x[i,j,k] == 0)

        model.setObjective(gp.quicksum(self.tC[i,j,k]*x[i,j,k] for i in I for j in J for k in range(K) if self.tC[i,j,k]!=np.inf and not np.isnan(self.tC[i,j,k])))
        
        # internal variables useful for the subtour elimination
        model._vars = x
        model._K = K
        model._size = N

        # suppress outputs
        model.setParam(gp.GRB.Param.OutputFlag,0)
        model.Params.lazyConstraints = 1
        model.Params.MIPGap = 5e-4
        model.Params.TimeLimit = 3*60


        ls = []
        cs = []

        if verbose: print("Running Gurobi optimisation...")
        model.optimize(utils.subtourelim)
        if verbose: print("Gurobi done.")


        ## save the tour
        arcs = gp.tuplelist((i,j,k) for i,j,k in model._vars if model._vars[i,j,k].X > 0.5)
        tour = []
        tour.append(arcs[0][0])

        for i in range(1,len(arcs)):
            tour.append(arcs[tour[i-1]][1])
        tour.append(tour[0])
        self.tour = tour

        ## save tnodes and lengths
        used_nodes = np.zeros((N,N))
        length = 0
        dist_from_c = 0
        for i,j,k in arcs:
            used_nodes[i,j] = self.tnodes[i,j,k]
            length += self.tL[i,j,k]
            dist_from_c += self.tC[i,j,k]
        self.used_nodes = used_nodes
        ls.append(length)
        cs.append(dist_from_c)

        return length,dist_from_c,model.Runtime
    

    def optimise_direct_path(self,tau,verbose = True):
        """Define the constrained Integer Program without triangular nodes, solve using Gurobi and save the tour.

        Parameters
        ----------
        tau : float
            Upper bound on the total length
        verbose: bool
            if True, print a message before and after optimisation. True by default


        Returns
        -------
        length: float
            Total length of the tour
        dist_from_c: float
            Total distance from the center
        runtime: float
            Total runtime for Gurobi optimisation
        """
        N = self.size
        I = range(N)
        J = range(N)
        K = 1

        model = gp.Model()
        model.ModelSense = gp.GRB.MINIMIZE

        x = model.addVars([(i,j,k) for i in I for j in J for k in range(K)], vtype = gp.GRB.BINARY)
        for i in I:
            model.addConstr(gp.quicksum(x[i,j,k] for j in J for k in range(K)) == 1) 
            model.addConstr(gp.quicksum(x[j,i,k] for j in I for k in range(K)) == 1) 
        model.addConstr(gp.quicksum(self.L[i,j]* x[i,j,k] for i in I for j in J for k in range(K) if self.L[i,j]!=np.inf) <= tau,name='Lbound')

        model.setObjective(gp.quicksum(self.C[i,j]*x[i,j,k] for i in I for j in J for k in range(K) if self.C[i,j]!=np.inf and not np.isnan(self.C[i,j])))

        # internal variables useful for the subtour elimination
        model._vars = x
        model._K = K
        model._size = N

        # suppress outputs
        model.setParam(gp.GRB.Param.OutputFlag,0)
        model.Params.lazyConstraints = 1
        model.Params.MIPGap = 5e-4
        model.Params.TimeLimit = 3*60

        ls_d = []
        cs_d = []
        
        if verbose: print("Running Gurobi optimisation...")
        model.optimize(utils.subtourelim)
        if verbose: print("Gurobi done.")


        ## save the tour
        arcs = gp.tuplelist((i,j,k) for i,j,k in model._vars if model._vars[i,j,k].X > 0.5)
        tour = []
        tour.append(arcs[0][0])

        for i in range(1,len(arcs)):
            tour.append(arcs[tour[i-1]][1])
        tour.append(tour[0])
        self.tour = tour

        ## save lengths
        length = 0
        dist_from_c = 0
        for i,j,k in arcs:
            length += self.L[i,j]
            dist_from_c += self.C[i,j]
        ls_d.append(length)
        cs_d.append(dist_from_c)

        return length,dist_from_c,model.Runtime
    
    def plot(self):
        """ 
        Plots the grid.
        """
        labels = self.labels
        center = self.center
        nodes = self.nodes
        edges = self.edges

        fig, ax = plt.subplots(figsize=(7,7))
        for i,j in combinations(labels,2):
            if edges[i,j] < np.inf:
                plt.plot([nodes.T[0,i],nodes.T[0,j]],[nodes.T[1,i],nodes.T[1,j]],'-', color = 'b')
        for node, label in zip(nodes,labels):
            if np.allclose(node,center):
                rect = pth.Rectangle(center-0.03,0.06,0.06,facecolor = 'lime',edgecolor='black',zorder = 2)
                ax.add_patch(rect)
                ax.text(center[0],center[1],f"{label:}",horizontalalignment='center',verticalalignment='center')
            else:
                circle  = pth.Circle(node,radius = 0.03,edgecolor = 'black',facecolor='white',zorder = 2)
                ax.add_patch(circle)
                ax.text(node[0],node[1],f"{label:}",horizontalalignment='center',verticalalignment='center')
        ax.set_xlim(-0.1,1.1)
        ax.set_ylim(-0.1,1.1)
        ax.set_aspect('equal')

        return fig, ax
    
    def __str__(self) -> str:
        outstr = "Travelling Salesman Problem with a center.\n" + \
            f"Size: {self.size:}\n" + \
            f"Grid shape: ({self.Nx:},{self.Ny:})\n" + \
            f"Number of triangular paths: {self.num_tpaths:}\n"
        return outstr  