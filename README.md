# SymmetricTSP
A collection of algorithms for solving the symmetric travelling salesman problem
Author: T.Monks 

TSP Formal Definition:

Given a undirected graph and cost for each edge, find the Hamiltonian Circuit of min total cost.

Informal Definition:

Given a set of cities and known distance between each pair of cities, 
find the tour that visits each city exactly once and minimises the total distance travelled

Solution methods included:

1. BruteForceSolver - enumerate all solutions.  Inefficient, but useful for small instance testing.
2. RandomSearch - a random tournament of solutions.  
3. OrdinarySearch - first improvement neighbourhood search - swaps two cities at a time
4. SteepestImprovement - best improvement neighbourhood search for best swap of two cities
5. MultipleRunner - runs any local search alg n times with a different rnd initial tour.  Selected best tour from n.
6. Nearest Neighbour - Greedy Contruction heuristic - starting from base add teh nearest city (terrible performance!)
7. OrdinaryDecent2Opt - first improvement neighbourhood search - 2 opt (edge) swap (reverse section of route)
8. SteepestDecent2Opt - best improvement neighbourhood search - 2 opt (edge) swap (reverse section of route)
9. FurthestInsertion - Construction heuristic. Bites the bullet early and adds in furthest points.
10. EvolutionaryAlgorithm - A family of evolutionary algorithm strategies
10a. MewLambdaEvolutionStrategy - Uses Truncation Selection to select the best Mew individuals in a population of solutions (of size Lambda) and mutate them. Mutation can be 2 city swap or 2-Opt.  
10b.  MewPlusLambdaEvolutionStrategy- Uses Truncation Selection to select the best Mew individuals in a population (of size Lambda) and mutate them. The Mew fittest are introduced in the next generation to compete against children.  Each generation is of size mew+lambda.  Mutation can be 2 city swap or 2-Opt. 
10c. GeneticAlgorithmStrategy - Uses selection with replacement to crossover individual chromosomes in a population and mutate them.  

All of these are tested in test.io.py
Look at test_io to see how to use each of the algorithms (easy setup for all)
Testing takes place with a slice of a bigger 70 city problem from TSPLib (st70.tsp)
set paramter trim_size to change slice.

Look at test_big.py to run against full algorithm (don't use brute force).

To read in data:

This is setup to work with data taken from TSPLib https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/

See \Data directory for st70.tsp

import tsp_io as io

use read_coordinates and read_meta_data to import data files.  
coordinates imported as numpy array.

See test_io.py for example