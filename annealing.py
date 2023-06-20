import numpy as np

from TSPC.model import TSPCeuclidean

class SimulatedAnnealing:
    def __init__(self,M,tf, tsp: TSPCeuclidean, triangular = False) -> None:
        """Instantiate a Simuated Annealing object.

        Parameters
        ----------
        M : int
            maximum number of epochs (iterations)
        tf : float
            temperature to reach at the end of the process
        tsp : TSPCeuclidean
            instance of a TSPC problem to be solved
        triangular : bool, optional
            whether to use triangular paths, by default False
        """
        self._problem = tsp
        self._triangular = triangular

        if triangular:
            self._solution = np.concatenate([np.random.choice(tsp.size,(tsp.size,1),False),np.random.choice(tsp.K,(tsp.size,1),True)],axis=1)
            self._energy = self._getEnergy(self._solution)
        else:
            self._solution = np.random.choice(tsp.size,tsp.size,False)
            self._energy = self._getEnergy(self._solution)

        self._T0 = self._findT0()
        self._T = self._T0
        self._Tf = tf

        self._M = M

        self._alpha = (tf/self._T0)**(1/(M-1))

    def simulate(self):
        """Annealing process
        """
        while self._T > self._Tf:
            for _ in range(self._problem.size):
                
                newTour = self._findNeighbour(self._solution)

                newE = self._getEnergy(newTour)
                diff = newE - self._energy

                if diff < 0:
                    self._solution = np.copy(newTour)
                    self._energy = newE
                else:
                    if np.random.random() < np.exp(-diff/self._T):
                        self._solution = np.copy(newTour)
                        self._energy = newE
            self._cooldown()

    @property
    def solution(self):
        return self._solution
    @property
    def energy(self):
        return self._energy


    def _cooldown(self):
        """Reduce temperature of the system
        """
        self._T = self._T * self._alpha

    def _getEnergy(self,tour)-> float:
        """Compute the energy

        Parameters
        ----------
        tour : numpy ndarray
            tour. If triangular paths are allowed, should be of shape (N,2) such that tour[i,0] is the i-th destination node
            and tour[i,1] is the triangular node used to go from tour[i,0] to tour[i+1,0]

        Returns
        -------
        e : float
            energy along the path
        """
        e = 0
        if self._triangular:
            for i in range(self._solution.shape[0]-1):
                e += self._problem._te[tour[i,0],tour[i+1,0],tour[i,1]]
            e += self._problem._te[tour[-1,0],tour[0,0],tour[-1,1]]
        else:
            for i in range(len(self._solution)-1):
                e += self._problem.E[tour[i],tour[i+1]]
            e += self._problem.E[tour[-1],tour[0]]
        return e
    
    def _findT0(self)->float:
        """Find suitable initial temperature

        Returns
        -------
        T0 : float
            initial temperature
        """
        e0 = self._energy
        e = []
        for _ in range(len(self._solution)):
            i,j = np.random.choice(len(self._solution),2)
            tour = np.copy(self._solution)
            tour[i], tour[j] = tour[j], tour[i]
            e.append(self._getEnergy(tour))
        diff = [ee-e0 for ee in e if ee > e0]
        meanInc = np.mean(diff)
        T0 = -meanInc/np.log(0.9) ## 0.9 is the desired probability of accepting an uphill move early on
        return T0
    
    def _findNeighbour(self,tour)->np.ndarray:
        """Finds a neighbour of a given tour. Two nodes are swapped and, if triangular paths are allowed, 
            a random triangular node is also assigned to the swapped nodes.

        Parameters
        ----------
        tour : numpy ndarray
            tour. If triangular paths are allowed, should be of shape (N,2) such that tour[i,0] is the i-th destination node
            and tour[i,1] is the triangular node used to go from tour[i,0] to tour[i+1,0]

        Returns
        -------
        newTour : numpy ndarray
            neighbour of the given tour
        """
        newTour = np.copy(tour)
        i,j = np.random.choice(len(tour),2)
        if self._triangular:
            newTour[[i,j]] = newTour[[j,i]]
            ki, kj = np.random.choice(self._problem.K,2,replace=True)
            newTour[i,1] = ki
            newTour[j,1] = kj
        else:
            newTour[i], newTour[j] = newTour[j], newTour[i]
        return newTour
