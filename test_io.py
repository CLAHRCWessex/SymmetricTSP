# -*- coding: utf-8 -*-
"""
Test script for reading of tsp data.
"""

import tsp_io as io
import euclidean as e
import objective as o
import init_solutions as init
from random import shuffle

import numpy as np

file_path = "Data\st70.tsp"
file_out = "Data\matrix.csv"
md_rows = 6

cities = io.read_coordinates(file_path, md_rows)
meta = io.read_meta_data(file_path, md_rows)

print(cities)
print(meta)

dist = e.euclidean_distance(cities[0], cities[1])
dist2 = e.euclidean_distance2(cities[0], cities[1])

print(dist, dist2)

matrix = e.gen_matrix(cities)

np.savetxt(file_out, matrix, delimiter=",")

tour = o.symmetric_tour_list(len(cities))

#random initial symmetric tour
tour = init.random_tour(tour)
cost = o.tour_cost(tour, matrix)
print(cost)


    

