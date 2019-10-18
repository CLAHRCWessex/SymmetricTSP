# -*- coding: utf-8 -*-
"""
Encapsulates objective functions for TSP
"""

from abc import ABC, abstractmethod
import numpy as np

def symmetric_tour_list(n_cities, start_index = 0):
    """
    Returns a numpy array representiung a symmetric tour of cities
    length = n_cities + 1
    First and last cities are index 'start_index' e.g. for
    start_index = 0 then for 5 cities tour = 
    [0, 1, 2, 3, 4, 0]
    """
    tour = [x for x in range(n_cities)]
    tour.remove(start_index)
    tour.insert(0, start_index)
    tour.append(start_index)
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


class AbstractObjective(ABC):   
    @abstractmethod
    def evaluate(self, solution):
        pass


class SimpleTSPObjective(AbstractObjective):
    '''
    Simple objective for the Symmetric TSP
    Evaluates the cost of a tour.
    '''
    def __init__(self, matrix):
        '''
        Constructor

        Parameters:
        -------
        matrix - numpy.array, matrix (2D array) representing the 
        edge costs between each city.
        '''
        self._matrix = matrix
    
    def evaluate(self, tour):
        """
        The eucidean total distance in the tour.
        
        Parameters: 
        --------
        tour -  numpy.array, vector (1D array) representing tour
                e.g. [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        Returns:
        -------
        float - cost of tour.  This is the euclidean difference between each
        city in @tour with the addition of looping back from the final city to the 
        first.

        """
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += self._matrix[tour[i]][tour[i+1]]

        cost += self._matrix[tour[len(tour)-1]][tour[0]]
            
        return cost

