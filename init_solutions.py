# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:57:46 2017

@author: tm3y13
"""

from random import shuffle

def random_tour(tour):
    rnd_tour = tour[1:len(tour)-1]
    shuffle(rnd_tour)
    rnd_tour.append(0)
    rnd_tour.insert(0,0)
    return rnd_tour
    
    