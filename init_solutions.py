# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:57:46 2017

@author: tm3y13
"""

from random import shuffle

def random_tour(tour):
    """
    Initial solution to tour is psuedo random
    """
    rnd_tour = tour[1:len(tour)-1]
    base_city = tour[0]
    shuffle(rnd_tour)
    rnd_tour.append(base_city)
    rnd_tour.insert(0, base_city)
    return rnd_tour
    
    