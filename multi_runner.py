# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 20:59:46 2017

@author: tm3y13
"""

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
        
       
            
            
            
            