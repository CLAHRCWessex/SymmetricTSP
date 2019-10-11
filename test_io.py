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
from local_search_2opt import OrdinaryDecent2Opt, SteepestDecent2Opt, LocalSearchArgs
from construction import NearestNeighbour, FurthestInsertion
from evolutionary import (EvolutionaryAlgorithm, MewLambdaEvolutionStrategy, 
                          MewPlusLambdaEvolutionStrategy, 
                          TwoOptMutator, TwoCityMutator)

import numpy as np

def mark_optimal(optimal_cost, cost):
    if(cost == optimal_cost):
        return '*'
    else:
        return ''
    

def print_output(solver):
    print("\nbest solutions:\t{0}".format(len(solver.best_solutions)))
    print("best cost:\t{0}".format(solver.best_cost))
    print("best solutions:")
    [print(s) for s in solver.best_solutions]


def print_multi_run(solver):
    print("\nbest solutions:\t{0}".format(len(solver.best_solutions)))
    print("best cost:\t{0}".format(solver.best_cost))
    print("best solutions:")
    [print(s[0]) for s in solver.best_solutions]
    
    
    

file_path = "Data/st70.tsp"
file_out = "Data/matrix.csv"
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
size_trim = 13 #note bruteforce is slow beyond 10
base_city = tour[0]
tour = tour[0:size_trim]  #select a subset of the big problem.
tour.append(base_city)
results = []

print("\n\n**Short tour\n{0}".format(tour))
print("initial cost: {0}".format(o.tour_cost(tour, matrix)))

solver = BruteForceSolver(tour, matrix)
print("Enumerating...")
#for size_trim 10 = 2.2s per loop
solver.solve()

print("\n** BRUTEFORCE OUTPUT ***")
print_output(solver)
cost1 = solver.best_cost


#Local Search - Single Run of Ordinary Decent 
solver = OrdinaryDecent(tour, matrix)
print("\nRunning Local Search using Ordinary Decent...")
#for trim_size = 10 = average 220ms 
solver.solve()

print("\n** ORDINARY DECENT OUTPUT ***")
print_output(solver)
cost2 = solver.best_cost


#Local Search - Single Run of Steepest Decent 
solver = SteepestDecent(tour, matrix)
print("\nRunning Local Search using Steepest Decent...")
#for trim_size = 10 = average 222ms 
solver.solve()

print("\n** STEEPEST DECENT OUTPUT ***")
print_output(solver)
cost3 = solver.best_cost

#Local Search - multiple runs of Ordinary Decent 
runner = MultiRunner(OrdinaryDecent(tour, matrix))
n = 10
print("\nRunning Local Search using Ordinary Decent (Best of {} runs)..."\
      .format(n))
runner.run(n)

print("\n** MULTIPLE RUNS OF ORDINARY DECENT OUTPUT ***")
print_multi_run(solver)
cost4, solutions = runner.get_best_solutions()

#Local Search - multiple runs of Steepest Decent 
runner = MultiRunner(SteepestDecent(tour, matrix))
print("\nRunning Local Search using Ordinary Decent (Best of {} runs)..."\
      .format(n))
runner.run(n)

print("\n** MULTIPLE RUNS OF STEEPEST DECENT OUTPUT ***")
print_multi_run(solver)
cost5, solutions = runner.get_best_solutions()



#Construction Heuristic - Nearest Neighbour
solver = NearestNeighbour(tour, matrix)
print("\nRunning Nearest Neighbour alg...")
solver.solve()

print("\n** NEAREST NEIGHBOUR OUTPUT ***")
print("\nbest solutions:\t{0}".format(1))
print("best cost:\t{0}".format(solver.best_cost))
cost6 = solver.best_cost
print("best solutions:")
print(solver.best_solution)
bs = solver.best_solution


#Local Search - Single Run of Ordinary Decent 
solver = OrdinaryDecent(bs, matrix)
print("\nRunning Local Search using Ordinary Decent...NN")
#for trim_size = 10 = average 220ms 
solver.solve()

print("\n** ORDINARY DECENT OUTPUT ***")
print_output(solver)
cost7 = solver.best_cost


#Local Search - Single Run of Ordinary Decent with 2 Opt Swap
args = LocalSearchArgs()
args.init_solution = tour
args.matrix = matrix
solver = OrdinaryDecent2Opt(args)
print("\nRunning Local Search using Ordinary Decent 2-Opt...")
solver.solve()

print("\n** ORDINARY DECENT 2 Opt OUTPUT ***")
print_output(solver)
cost8 = solver.best_cost



#Local Search - Single Run of Steepest Decent with 2 Opt Swap
args = LocalSearchArgs()
args.init_solution = tour
args.matrix = matrix
solver = SteepestDecent2Opt(args)
print("\nRunning Local Search using Ordinary Decent 2-Opt...")
solver.solve()

print("\n** ORDINARY DECENT 2 Opt OUTPUT ***")
print_output(solver)
cost9 = solver.best_cost

#Construction Heuristic - Furthest insertion
solver = FurthestInsertion(tour, matrix)
print("\nRunning Furthest Insertion alg...")
solver.solve()

print("\n** FURTHEST INSERTION OUTPUT ***")
print("best cost:\t{0}".format(solver.best_cost))
cost10 = solver.best_cost
print("best solutions:")
print(solver.best_solution)


#Evolutionary Algorithm - (mew, lambda) strategy
mew = 5
_lambda = 10
strategy = MewLambdaEvolutionStrategy(mew, _lambda, TwoCityMutator())
solver = EvolutionaryAlgorithm(tour, matrix,_lambda, strategy, max_iter=5000)
print("\nRunning (mew, lambda) evolutionary alg...")
solver.solve()

print("\n** (MEW, LAMBDA) OUTPUT ***")
print("best cost:\t{0}".format(solver.best_cost))
cost11 = solver.best_cost
print("best solutions:")
print(solver.best_solution)


#Evolutionary Algorithm - (mew+lambda) strategy
mew = 3
_lambda = 10
strategy = MewPlusLambdaEvolutionStrategy(mew, _lambda, TwoCityMutator())
solver = EvolutionaryAlgorithm(tour, matrix,_lambda, strategy)
print("\nRunning (mew + lambda) evolutionary alg...")
solver.solve()

print("\n** (MEW+LAMBDA) OUTPUT ***")
print("best cost:\t{0}".format(solver.best_cost))
cost12 = solver.best_cost
print("best solutions:")
print(solver.best_solution)


#Evolutionary Algorithm - (mew+lambda) strategy, TwoOpt Mutation
mew = 3
_lambda = 10
strategy = MewPlusLambdaEvolutionStrategy(mew, _lambda, TwoOptMutator())
solver = EvolutionaryAlgorithm(tour, matrix,_lambda, strategy)
print("\nRunning (mew + lambda) evolutionary alg with 2-Opt...")
solver.solve()

print("\n** (MEW+LAMBDA) OUTPUT ***")
print("best cost:\t{0}".format(solver.best_cost))
cost13 = solver.best_cost
print("best solutions:")
print(solver.best_solution)


#Summary of methods
print("\n** COST SUMMARY ***")


print("\nBrute Force:\t\t\t{0}".format(cost1))
print("Ordinary Decent:\t\t{0}\t{1}".format(cost2, mark_optimal(cost1, cost2)))
print("Steepest Decent:\t\t{0}\t{1}".format(cost3, mark_optimal(cost1, cost3)))
print("Ordinary Decent ({0} runs):\t{1}\t{2}".format(n, cost4, mark_optimal(cost1, cost4)))
print("Steepest Decent ({0} runs):\t{1}\t{2}".format(n, cost5, mark_optimal(cost1, cost5)))
print("Nearest Neighbour:\t\t{0}\t{1}".format(cost6, mark_optimal(cost1, cost6)))
print("Ordinary Decent NN init:\t{0}\t{1}".format(cost7, mark_optimal(cost1, cost7)))
print("Ordinary Decent 2-Opt\t\t{0}\t{1}".format(cost8, mark_optimal(cost1, cost8)))
print("Steepest Decent 2-Opt\t\t{0}\t{1}".format(cost9,mark_optimal(cost1, cost9)))
print("Furthest Insertion:\t\t{0}\t{1}".format(cost10, mark_optimal(cost1, cost10)))
print("EA: (Mew, Lambda) \t\t{0}\t{1}".format(cost11, mark_optimal(cost1, cost11)))
print("EA: (Mew+Lambda) \t\t{0}\t{1}".format(cost12, mark_optimal(cost1, cost12)))
print("EA: (Mew+Lambda)+2Opt \t\t{0}\t{1}".format(cost13, mark_optimal(cost1, cost13)))
print("\n*Optimal")
