# -*- coding: utf-8 -*-
"""
Test script for reading of tsp data.
"""

import tsp_io as io
import euclidean as e
import objective as o
import init_solutions as init
from bruteforce import BruteForceSolver

import numpy as np

file_path = "Data\st70.tsp"
file_out = "Data\matrix.csv"
md_rows = 6

cities = io.read_coordinates(file_path, md_rows)
meta = io.read_meta_data(file_path, md_rows)

print(cities)
print(meta)

#example of calculating a single euclidean distance
dist = e.euclidean_distance(cities[0], cities[1])
print(dist)

#generate matrix
matrix = e.gen_matrix(cities)

#output city matrix - to validate and use for manual calcs etc.
np.savetxt(file_out, matrix, delimiter=",")

#you can specify the start/end city index using
#the optional parameter start_index.  Default index = 0
tour = o.symmetric_tour_list(len(cities), 2) # city at index 2 is start/end
# tour = o.symmetric_tour_list(len(cities)) #  for city 0

print(tour)

#randomise the cities apart from start/end
tour = init.random_tour(tour)
print("\n", tour)

cost = o.tour_cost(tour, matrix)
print(cost)


#Brute force example for small TSP problems

#need somethign to produce "short tour from large".
size_trim = 10 #note slow beyond 10
base_city = tour[0]
tour = tour[0:size_trim]  #select a subset of the big problem.
tour.append(base_city)

print("\n\n**Short tour\n{0}".format(tour))
print("initial cost: {0}".format(o.tour_cost(tour, matrix)))

solver = BruteForceSolver(tour, matrix)
print("Enumerating...")
solver.solve()

print("\n** OUTPUT ***")
print("\nbest solutions:\t{0}".format(len(solver.best_solutions)))
print("best cost:\t{0}".format(solver.best_cost))
print("best solutions:")
[print(s) for s in solver.best_solutions]










    

