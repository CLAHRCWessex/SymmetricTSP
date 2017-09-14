# -*- coding: utf-8 -*-
"""
Encapsulates objective functions for TSP
"""

import numpy as np

def symmetric_tour_list(n_cities):
    """
    Returns a numpy array representiung a symmetric tour of cities
    length = n_cities + 1
    First and last cities are index 0 e.g. for 5 cities
    [0, 1, 2, 3, 4, 0]
    """
    tour = [x for x in range(n_cities)]
    tour.append(0)
    return tour
    
def tour_cost(tour, matrix):
    """
    The total distance in the tour.
    
    @tour numpy array for tour
    @matrix numpy array of costs in travelling between 2 points.
    """
    cost = 0
    for i in range(len(tour) - 1):
        cost += matrix[tour[i]][tour[i+1]]
        
    return cost
    