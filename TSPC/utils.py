import gurobipy as gp
import numpy as np



### ------ Gurobi utility functions ------ ###

def subtourelim(model, where):
    size = model._size
    if where == gp.GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j, k) for i, j, k in model._vars.keys() if vals[i, j, k] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected,size)
        if len(tour) < size:
            # add subtour elimination constr. for every pair of cities in subtour
            model.cbLazy(gp.quicksum(model._vars[i, j, k] for i in tour for j in tour for k in range(model._K)) <= len(tour)-1)

# Given a tuplelist of edges, find the shortest subtour

def subtour(edges,size):
    unvisited = list(range(size))
    cycle = list(range(size)) # Dummy - guaranteed to be replaced
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j, k  in edges.select(current, '*','*') if j in unvisited]
        if len(thiscycle) <= len(cycle):
            cycle = thiscycle # New shortest subtour
    return cycle


### ------- Grid utility functions ------- ###

def dijkstra(adj,labels,s):
    """
    Modified Dijkstra to find all shortest paths in a grid.

    Parameters
    ----------
    adj : list[list]
        adjecency list for the graph
    labels : list
        list of labels for the nodes (assumed to be of the form `range(N)`). Items are used as indices.
    s : int
        label of the source node

    Returns
    -------
    d: float
        length of the shortest paths
    p: list[list]
        parents for each shortest path from s to all the other nodes
    """
    d = []
    p = [[] for _ in adj]
    flag = []

    for i in labels:
        d.append(np.inf)
        flag.append(0)
    d[s] = 0
    p[s] = [-1]

    q = dict(zip(labels,d))

    while q:
        m = 999
        for i in q:
            if m > d[i] and flag[i] != -1:
                m = d[i]
                u = i
        flag[u] = -1
        q.pop(u)

        for w,v in adj[u]:
            if d[v] > d[u] + w:
                d[v] = d[u] + w
                p[v] = [u]
            elif np.isclose(d[v],d[u]+w):
                p[v].append(u)
    return d,p



def find_paths(a,b,path,paths,p):
    """
    Finds all possible paths between nodes a,b given a list of parents.

    Parameters
    ----------
    a : int
        label of the first node
    b : int
        label of the second node
    path : list
        current path
    paths : list[list]
        list of all paths
    p : list[list]
        parents for each shortest path from a to all the other nodes
    """
    if b == -1:
        paths.append(path.copy())
        return
    for par in p[b]:
        path.append(b)
        find_paths(a,par,path,paths,p)
        path.pop()
