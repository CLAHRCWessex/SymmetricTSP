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
        if candidate_cost > home_cost:
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



class IteratedLocalSearch(object):
    
    def __init__(self, objective, local_search, accept=None, perturb=None, maximisation=True):
        '''
        Constructor Method

        Parameters:
        --------
        local_search -- hill climbing solver or similar
        perturb -- ILSPertubation, conatins logic for pertubation from the local optimimum in each iteration

        '''
        self._objective = objective
        self._local_search = local_search
        if accept == None:
            self._accepter = RandomHomeBase()
        else:
            self._accepter = accept

        if perturb == None:
            self._perturber = DoubleBridgePertubation()
        else:
            self._perturber = perturb

        if maximisation:
            self._negate = 1
        elif not maximisation:
            self._negate = -1
        else:
            raise ValueError('maximisation must be of type bool (True|False) ')
            
        self._solutions = []
        self._best_cost = np.inf * self._negate
        
        
    def run(self, n):
        """
        Re-run solver n times using a different initial solution
        each time.  Init solution is generated randomly each time.

        The potential power of iteratedl ocal search lies in its biased sampling of the set of local optima.

        """

        current = self._local_search.solution
        np.random.shuffle(current) # random tour
        
        home_base = current
        home_base_cost = self._objective.evaluate(current) * self._negate
        self._best_cost = home_base_cost 
        self._solutions.append(current)
                
        for x in range(n):

            #Hill climb from new starting point
            self._local_search.set_init_solution(current)
            self._local_search.solve()
            current = self._local_search.best_solutions[0]

            #will need to refactor 2Opt search from decent to ascent....maybe...
            iteration_best_cost = self._local_search.best_cost * self._negate

            if iteration_best_cost > self._best_cost:
                self._best_cost = iteration_best_cost
                self._solutions = self._local_search.best_solutions

            elif iteration_best_cost == self._best_cost:
                self._solutions.append(self._local_search.best_solutions)
                [self._solutions.append(i) for i in self._local_search.best_solutions]

            home_base, home_base_cost = self._accepter.new_home_base(home_base, home_base_cost, 
                                                     current, iteration_best_cost)
            current = self._perturber.perturb(home_base)
            
        
    def get_best_solutions(self):
        return self._best_cost * self._negate, self._solutions

  
            
            
            