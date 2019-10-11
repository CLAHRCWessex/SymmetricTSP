# -*- coding: utf-8 -*-
"""
Test script for TSP algs against large instances of the problem.

"""

import tsp_io as io
import euclidean as e
import objective as o
import init_solutions as init
from local_search_2opt import OrdinaryDecent2Opt, SteepestDecent2Opt, LocalSearchArgs
from multi_runner import MultiRunner

file_path = "Data\st70.tsp"
file_out = "Data\matrix.csv"
md_rows = 6

cities = io.read_coordinates(file_path, md_rows)
meta = io.read_meta_data(file_path, md_rows)

#generate matrix
matrix = e.gen_matrix(cities)

#you can specify the start/end city index using
#the optional parameter start_index.  Default index = 0
tour = o.symmetric_tour_list(len(cities), 2) # city at index 2 is start/end


#randomise the cities apart from start/end
tour = init.random_tour(tour)

#need somethign to produce "short tour from large".
#size_trim = 50 #note bruteforce is slow beyond 10
#base_city = tour[0]
#tour = tour[0:size_trim]  #select a subset of the big problem.
#tour.append(base_city)


#Local Search - multiple runs of Ordinary Decent 
args = LocalSearchArgs()
args.init_solution = tour
args.matrix = matrix
runner = MultiRunner(OrdinaryDecent2Opt(args))
n = 2
print("\nCommencing Local Search using Ordinary Decent (Best of {} runs)..."\
      .format(n))
runner.run(n)

print("\n** MULTIPLE RUNS OF ORDINARY DECENT 2OPT OUTPUT ***")
cost4, solutions = runner.get_best_solutions()
print("\nbest solutions:\t{0}".format(len(solutions)))
print("best cost:\t{0}".format(cost4))
print("best solutions:")
[print(s[0]) for s in solutions]

#optimal is 676.  50 runs of ordinary decent 2Opt found - 690



#Local Search - multiple runs of steepest Decent 
args = LocalSearchArgs()
args.init_solution = tour
args.matrix = matrix
runner = MultiRunner(SteepestDecent2Opt(args))
n = 5
print("\nCommencing Local Search using Steepest Decent (Best of {} runs)..."\
      .format(n))
runner.run(n)

print("\n** MULTIPLE RUNS OF STEEPEST DECENT 2OPT OUTPUT ***")
cost4, solutions = runner.get_best_solutions()
print("\nbest solutions:\t{0}".format(len(solutions)))
print("best cost:\t{0}".format(cost4))
print("best solutions:")
[print(s[0]) for s in solutions]

#optimal is 676.  50 runs of ordinary decent 2Opt found - 679