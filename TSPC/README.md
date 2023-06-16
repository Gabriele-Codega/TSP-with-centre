# TSPC
## model.py
A class to represent the model defined on an euclidean graph with nodes in the square $[0,1] \times [0,1]$ and centre at $(0.5,0.5)$.

The methods include
- methods to find the triangular paths, their lengths and distances from the centre
- methods to define a Gurobi model (constrained or unconstrained), optimise it and retrieve the optimal tour
- methods to write the .tsp file required by LKH, solve the problem with LKH and retrieve the solution
- a method to visualise the nodes and the optimal path

## grid_model.py
A class to represent the model defined on a grid network.

The methods include
- methods to generate the grid with the desired dimensions, assign a random centre and generate random edges
- methods to find the shortest paths, both in the sense of euclidean distance and in the sense of the TSPC problem
- a method to find candidate triangular paths along with their length and distance from centre
- methods to define Gurobi models and optimise them, with or without triangular paths, and retrieve the optimal path
- a method to visualise the grid

## metrics.py
Includes the definition of distance and energy functions employed in the paper.

## utils.py
Utility functions that include
- a function to find the triangular nodes for euclidean models
- callback functions for the lazy implementation of subtour elimination constraints in Gurobi models
- functions to find shortest paths in a graph.