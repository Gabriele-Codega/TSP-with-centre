# Travelling Salesman Problem with a centre
This repository includes code written to study and reproduce the results from [this article](https://doi.org/10.1016/j.cor.2022.105748) by Yuchen Luo et al.

## TSPC
TSPC includes the implementation of two classes and various utility functions.

Note that, other than some _pretty much_ standard modules, it also requires Gurobipy.

## LKH-2.0.10
An empty directory which should be replaced by the directory found at [this website](http://webhotel4.ruc.dk/~keld/research/LKH/).

Quoting the website
>> LKH is an effective implementation of the Lin-Kernighan heuristic for solving the traveling salesman problem.

## .py files
Some sample code that shows how to setup and solve a problem with the TSPC code.

## Other files
- Shell script called by a class method to run the LKH algorithm
- .tsp file that contains parameters for the LKH algorithm