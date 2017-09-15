# -*- coding: utf-8 -*-
"""
Local search algorithms for TSP (a.k.a neighbourhood search)
Implementations:
1. Ordinary Search - find first improvement
"""

from objective import tour_cost
from tsp_utility import append_base, trim_base

class OrdinaryDecent(object):
    
    def __init__(self, init_solution, matrix):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self.matrix = matrix
        self.solution = init_solution
        #list of best solutions in case of multiple optima
        self.best_solutions = [init_solution]
        self.best_cost = tour_cost(self.solution, matrix)
        
    def solve(self):
        
        improvement = True
        
        while(improvement):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    
                    self.swap_cities(city1, city2)
                    
                    new_cost = tour_cost(self.solution, self.matrix)
                    
                    if (new_cost == self.best_cost):
                        self.best_solutions.append(self.solution)
                        improvement = True
                    elif (new_cost < self.best_cost):
                        self.best_cost = new_cost
                        self.best_solutions = [self.solution]
                        improvement = True
                    else:
                        self.swap_cities(city1, city2)
                        
                        
                    
                    
                    
    def swap_cities(self, city1, city2):
        self.solution[city1], self.solution[city2] = \
            self.solution[city2], self.solution[city1]
        
                    
            
    
    
    