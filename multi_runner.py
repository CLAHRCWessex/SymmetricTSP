# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 20:59:46 2017

@author: tm3y13
"""

from abc import ABC, abstractmethod
import numpy as np

from init_solutions import random_tour

class MultiRunner(object):
    
    def __init__(self, solver):
        self.solver = solver
        self.solutions = []
        self.costs = []
        self.best_indexs = []
        
    def run(self, n):
        """
        Re-run solver n times using a different initial solution
        each time.  Init solution is generated randomly each time.
        """
        for x in range(n):
        
            init = random_tour(self.solver.solution)
            #print(init)
            self.solver.set_init_solution(init)
            self.solver.solve()
            self.solutions.append(self.solver.best_solutions)
            self.costs.append(self.solver.best_cost)
            #print(self.solver.best_cost)
            
        self.save_best_solution()
            
    
    def save_best_solution(self):
        bcost = min(self.costs)
        self.best_indexs = [i for i,x in enumerate(self.costs) if x == bcost]
        
    def get_best_solutions(self):
        return self.costs[self.best_indexs[0]], [self.solutions[x] for x in self.best_indexs]




class ILSPertubation(ABC):
    @abstractmethod
    def perturb(self, tour):
        pass


class ILSHomeBaseAcceptanceLogic(ABC):
    @abstractmethod
    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        pass
    

class HigherQualityHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Accept if candidate is better than home_base
    '''

    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        if candidate_cost > home_cost
            return candidate, candidate_cost
        else:
            return home_base, home_cost

class RandomHomeBase(ILSHomeBaseAcceptanceLogic):
    '''
    Random walk homebase
    '''
    def new_home_base(self, home_base, home_cost, candidate, candidate_cost):
        return candidate, candidate_cost

class DoubleBridgePertubation(ILSPertubation):
    '''
        Perform a random 4-opt ("double bridge") move on a tour.
        
         E.g.
        
            A--B             A  B
           /    \           /|  |\
          H      C         H------C
          |      |   -->     |  |
          G      D         G------D
           \    /           \|  |/
            F--E             F  E
        
        Where edges AB, CD, EF and GH are chosen randomly.

    '''

    def perturb(self, tour):
        '''
        Perform a random 4-opt ("double bridge") move on a tour.
        
        Returns:
        --------
        numpy.array, vector. representing the tour

        Parameters:
        --------
        tour - numpy.array, vector representing tour between cities e.g.
               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        '''

        n = len(tour)
        pos1 = np.random.randint(1, int(n/4)+1) 
        pos2 = pos1 + np.random.randint(1, int(n/4)+1) 
        pos3 = pos2 + np.random.randint(1, int(n/4)+1) 

        p1 = np.concatenate((tour[:pos1] , tour[pos3:]), axis=None)
        p2 = np.concatenate((tour[pos2:pos3] , tour[pos1:pos2]), axis=None)
        #this bit will need updating if switching to alternative operation.
        return np.concatenate((p1, p2, p1[0]), axis=None)


if __name__ == '__main__':
    p = DoubleBridgePertubation()

    tour = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    result = p.perturb(tour)

    print(result)


class IteratedLocalSearch(object):
    
    def __init__(self, local_search, accept=None, perturb=None):
        '''
        Constructor Method

        Parameters:
        --------
        local_search -- hill climbing solver or similar
        perturb -- ILSPertubation, conatins logic for pertubation from the local optimimum in each iteration

        '''
        self._local_search = local_search
        if accept == None:
            self._accepter = RandomHomeBase()
        else:
            self._accepter = accept

        if perturb == None:
            self._perturb = DoubleBridgePertubation()
        else:
            self._perturber = perturb
            
        self._solutions = []
        self.costs = []
        self.best_indexs = []
        
    def run(self, n):
        """
        Re-run solver n times using a different initial solution
        each time.  Init solution is generated randomly each time.

        The potential power of iteratedl ocal search lies in its biased sampling of the set of local optima.

        """

        current = random_tour(self.solver.solution)
        home_base = current
        self._solutions[current]
        best = current
        best_cost = np.inf
        self._negate = -1

        for x in range(n):

            #Hill climb from new starting point
            self._local_search.set_init_solution(current)
            self._local_search.solve()
            #will need to refactor 2Opt search from decent to ascent....maybe...
            iteration_best_cost = self._local_search.best_cost * self._negate

            if iteration_best_cost > best_cost:
                best_cost = iteration_best_cost
                self._solutions = self._local_search.best_solutions

            elif iteration_best_cost == best_cost:
                self._solutions.append(self._local_search.best_solutions)
                [self._solutions.append(i) for i in self._local_search.best_solutions]


            
            current = self._pertuber.perturb(home_base)


            
            #print(self.solver.best_cost)
            
        self.save_best_solution()
    


    def new_home_base(home_base, current):
        pass

    
    def save_best_solution(self):
        bcost = min(self.costs)
        self.best_indexs = [i for i,x in enumerate(self.costs) if x == bcost]
        
    def get_best_solutions(self):
        return self.costs[self.best_indexs[0]], [self._solutions[x] for x in self.best_indexs]
            
            
            