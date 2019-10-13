# -*- coding: utf-8 -*-
"""
Functions and classes to enable bruteforce solution of the TSP.
Note Bruteforce is inefficient after tours exceed 5 cities
"""


import itertools as ite
import numpy as np

from objective import tour_cost
from init_solutions import random_tour
from tsp_utility import append_base, trim_base
    

class BruteForceSolver(object):
    """
    Enumerates all permutations of a given list of cities and calculates
    waits. Note for n cities there are n! permutations!  BruteForceSolver is
    too slow for tours beyond 5 cities.
    """
    def __init__(self, init_solution, matrix):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self.matrix = matrix
        self.init_solution = init_solution
        #list of best solutions in case of multiple optima
        self.best_solutions = [init_solution]
        self.best_cost = tour_cost(init_solution, matrix)
    
    def all_permutations(self, tour, base_city):
        """
        Returns a list of lists containing all permutations of a
        tour.  The base_city is appended to each_list
        """
        return [append_base(list(x), base_city) for x in ite.permutations(tour)]
    
        
    
    def solve(self):
        """
        Enumerate all costs to find the minimum.
        Store solution(s)
        """
        trimmed_tour, base_city = trim_base(self.init_solution)
        perms = self.all_permutations(trimmed_tour, base_city)
        
        #print(p) for p in perms]  #uncomment if want to see all perms
    
        for current_tour in perms:
            cost = tour_cost(current_tour, self.matrix)
            #print(cost)
            
            if self.best_cost == cost:
                self.best_cost = cost
                self.best_solutions.append(current_tour)
                
            elif self.best_cost > cost:
                self.best_cost = cost
                self.best_solutions = [current_tour]
                
                
                
            
class RandomSearch(object):
    """
    A simple global optimisation algorithm - encapsulates Random Search.  
    The algorithm is completely explorative and randomly
    samples a tour and compares if it is better than the current
    best.
    """
    def __init__(self, init_solution, matrix, max_iter=1000):
        """
        Constructor Method

        Parameters:
        ---------
        init_solution -- initial tour
        matrix -- matrix of travel costs
        """
        self._matrix = matrix
        self._init_solution = init_solution
        self._max_iter = max_iter
        #list of best solutions in case of multiple optima
        self.best_solutions = [init_solution]
        self.best_cost = tour_cost(init_solution, matrix)

    def solve(self):
        '''
        Random search.
        
        Loop until all iterations are complete.  
        Sample a random tour on each iterations and compare to best.
        
        '''
        
        for iteration in range(self._max_iter):
            sample_tour = random_tour(self._init_solution)
            sample_cost = tour_cost(sample_tour, self._matrix)

            if self.best_cost == sample_cost:
                self.best_solutions.append(sample_tour)

            elif sample_cost < self.best_cost:
                self.best_solutions = [sample_tour]
                self.best_cost = sample_cost
        
            
