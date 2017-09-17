# -*- coding: utf-8 -*-
"""
Test script for reading of tsp data.
"""

import tsp_io as io
import euclidean as e
import objective as o
import init_solutions as init
from bruteforce import BruteForceSolver
from local_search import OrdinaryDecent, SteepestDecent
from multi_runner import MultiRunner

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
size_trim = 10 #note bruteforce is slow beyond 10
base_city = tour[0]
tour = tour[0:size_trim]  #select a subset of the big problem.
tour.append(base_city)

print("\n\n**Short tour\n{0}".format(tour))
print("initial cost: {0}".format(o.tour_cost(tour, matrix)))

solver = BruteForceSolver(tour, matrix)
print("Enumerating...")
#for size_trim 10 = 2.2s per loop
solver.solve()

print("\n** BRUTEFORCE OUTPUT ***")
print("\nbest solutions:\t{0}".format(len(solver.best_solutions)))
print("best cost:\t{0}".format(solver.best_cost))
cost1 = solver.best_cost
print("best solutions:")
[print(s) for s in solver.best_solutions]

#Local Search - Single Run of Ordinary Decent 
solver = OrdinaryDecent(tour, matrix)
print("Commencing Local Search using Ordinary Decent...")
#for trim_size = 10 = average 220ms 
solver.solve()

print("\n** ORDINARY DECENT OUTPUT ***")
print("\nbest solutions:\t{0}".format(len(solver.best_solutions)))
print("best cost:\t{0}".format(solver.best_cost))
cost2 = solver.best_cost
print("best solutions:")
[print(s) for s in solver.best_solutions]

#Local Search - Single Run of Steepest Decent 
solver = SteepestDecent(tour, matrix)
print("Commencing Local Search using Steepest Decent...")
#for trim_size = 10 = average 222ms 
solver.solve()

print("\n** STEEPEST DECENT OUTPUT ***")
print("\nbest solutions:\t{0}".format(len(solver.best_solutions)))
print("best cost:\t{0}".format(solver.best_cost))
cost3 = solver.best_cost
print("best solutions:")
[print(s) for s in solver.best_solutions]

#Local Search - multiple runs of Ordinary Decent 
runner = MultiRunner(OrdinaryDecent(tour, matrix))
n = 5
print("\nCommencing Local Search using Ordinary Decent (Best of {} runs)..."\
      .format(n))
runner.run(n)

print("\n** MULTIPLE RUNS OF ORDINARY DECENT OUTPUT ***")
cost4, solutions = runner.get_best_solutions()
print("\nbest solutions:\t{0}".format(len(solutions)))
print("best cost:\t{0}".format(cost4))
print("best solutions:")
[print(s[0]) for s in solutions]

#Local Search - multiple runs of Steepest Decent 
runner = MultiRunner(SteepestDecent(tour, matrix))
print("\nCommencing Local Search using Ordinary Decent (Best of {} runs)..."\
      .format(n))
runner.run(n)

print("\n** MULTIPLE RUNS OF STEEPEST DECENT OUTPUT ***")
cost5, solutions = runner.get_best_solutions()
print("\nbest solutions:\t{0}".format(len(solutions)))
print("best cost:\t{0}".format(cost5))
print("best solutions:")
[print(s[0]) for s in solutions]


#Summary of methods
print("\n** COST SUMMARY ***")
print("\nBrute Force:\t\t\t{0}".format(cost1))
print("Ordinary Decent:\t\t{0}".format(cost2))
print("Steepest Decent:\t\t{0}".format(cost3))
print("Ordinary Decent ({0} runs):\t{1}".format(n, cost4))
print("Steepest Decent ({0} runs):\t{1}".format(n, cost5))




    

