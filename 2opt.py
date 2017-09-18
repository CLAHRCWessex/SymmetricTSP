# -*- coding: utf-8 -*-
"""
local search implemented with 2-opt swap

@author: Tom Monks
"""

from objective import tour_cost


class LocalSearchArgs(object):
    """
    Argument class for local search classes
    """
    def __init__(self):
        pass
    
    
class OrdinaryDecent2Opt(object):
    """
    
    Local (neighbourhood) search implemented as first improvement 
    
    """   
    def __init__(self, args):
        """
        Constructor Method
        @init_solution = initial tour
        @matrix = matrix of travel costs
        """
        self.matrix = args.matrix
        self.set_init_solution(args.init_solution)
        self.swapper = args.swapper
        
    
    def set_init_solution(self, solution):  
        self.solution = solution
        self.best_solutions = [solution]
        self.best_cost = tour_cost(self.solution, self.matrix)        
    
    def solve(self):
        
        improvement = True
        
        while(improvement):
            improvement = False
            
            for city1 in range(1, len(self.solution) - 1):
                
                for city2 in range(city1 + 1, len(self.solution) - 1):
                    
                    self.reverse_sub_list(self.solution, city1, city2)
                    
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
                 
                    
    def reverse_sublist(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end
        """
        lst[start:end] = lst[start:end][::-1]
        return lst

                    