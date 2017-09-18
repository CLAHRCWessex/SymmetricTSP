# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 07:48:50 2017

@author: tm3y13
"""
from objective import tour_cost


class NearestNeighbour(object):
    
    def __init__(self, cities, matrix):
        self.cities = cities
        self.matrix = matrix
        self.solution = []
        self.best_cost = 99999
        self.best_solution = self.solution
        
    def solve(self):
        """
        Constructs a tour based on nearest neighbour method.
        Assume that first city in tour is base city.
        """
        
        from_city = self.cities[0]
        
        self.solution.append(from_city)
        
        
        for i in range(len(self.cities) - 2):
            to_city = self.closest_city_not_in_tour(from_city)
            self.solution.append(self.cities[to_city])
            print(self.solution)
            from_city = to_city
        
        self.solution.append(self.cities[0])
        self.best_cost = tour_cost(self.solution, self.matrix)
        self.best_solution = self.solution
                   
            
                    
    def closest_city_not_in_tour(self, from_city):
        min_cost = 999
        min_index = from_city
        
        for to_city in range(len(self.cities) - 1):
            
            if (min_cost > self.matrix[from_city][to_city]):
                              
                if (self.cities[to_city] not in self.solution):
                    min_index = to_city
                    min_cost = self.matrix[from_city][to_city]
        
        return min_index
        
        
            
                           
            
            
        
        