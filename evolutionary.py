
import numpy as np
from abc import ABC, abstractmethod

from objective import tour_cost, symmetric_tour_list
from tsp_utility import append_base, trim_base
from init_solutions import random_tour


def initiation_population(population_size, tour):
    '''
    Generate a list of @population_size tours.  Tours
    are randomly generated and unique to maximise
    diversity of the population.

    Parameters:
    ---------
    population_size -- the size of the population

    Returns:
    ---------
    List - a list of numpy arrays each of which is a unique tour.
    '''

    population = {}
    for i in range(population_size):
        
        #some code is legacy and uses python
        #lists instead of numpy arrays... to fix!
        new_tour = random_tour(tour)

        if str(new_tour) not in population:
            population[str(new_tour)] = np.array(new_tour)
        else:
            i = i - 1
    return list(population.values())
    

class TwoCityMutator(object):
    '''
    Mutates an individual tour by
    randomly swapping two cities.
    '''
    def mutate(self, tour):
        '''
        Randomly swap two cities
        
        Parameters:
        --------
        tour, np.array, tour

        '''
        #remember that index 0 and len(tour) are start/end city
        to_swap = np.random.randint(1, len(tour) - 1, 2)

        tour[to_swap[0]], tour[to_swap[1]] = \
            tour[to_swap[1]], tour[to_swap[0]]

        return tour


class TwoOptMutator(object):
    '''
    Mutates an individual tour by
    randomly swapping two cities.
    '''
    def mutate(self, tour):
        '''
        Randomly reverse a section of the route
        
        Parameters:
        --------
        tour, np.array, tour

        '''
        #remember that index 0 and len(tour) are start/end city
        to_swap = np.random.randint(1, len(tour) - 1, 2)

        if to_swap[1] < to_swap[0]:
            to_swap[0], to_swap[1] = to_swap[1], to_swap[0]

        return self._reverse_sublist(tour, to_swap[0], to_swap[1])


    def _reverse_sublist(self, lst, start, end):
        """
        Reverse a slice of the @lst elements between
        @start and @end
        """
        lst[start:end] = lst[start:end][::-1]
        return lst


class AbstractTourMutator(ABC):
    @abstractmethod
    def mutate(self, tour):
        pass

class AbstractEvolutionStrategy(ABC):
    @abstractmethod
    def evolve(self, population, costs):
        pass

class MewLambdaEvolutionStrategy(object):
    '''
    The (mew, lambda) evolution strategy
    The fittest mew of each generation
    produces lambda/mew offspring each of which is
    mutated.

    Each generation is of size lambda.
    '''
    def __init__(self, mew, _lambda, mutator):
        '''
        Constructor

        Parameters:
        --------
        mew -       int, controls how selectiveness the algorithm.  
                    Low values of mew relative to _lambsa mean that only the best 
                    breed in each generation and the algorithm becomes 
                    more exploitative.

        _lambda -   int, controls the size of each generation.

        mutator -   AbstractTourMutator, encapsulates the logic of mutation for a tour
        '''
        self._mew = mew
        self._lambda = _lambda
        self._mutator = mutator

    
    def evolve(self, population, costs):
        '''
        Only mew fittest individual survice.
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- list, of numpy arrays representing a generation of tours

        Returns:
        --------
        list of numpy.arrays - a new generation of tours.
        '''

        fittest_indexes = np.argpartition(costs, self._mew)[:self._mew]
        fittest = np.array(population)[fittest_indexes]

        population = []

        for parent in fittest:
            for child_n in range(int(self._lambda/self._mew)):
                child = self._mutator.mutate(parent.copy())
                population.append(child)

        return population
        

class MewPlusLambdaEvolutionStrategy(object):
    '''
    The (mew+lambda) evolution strategy
    The fittest mew of each generation
    produces lambda/mew offspring each of which is
    mutated.  The mew fittest parents compete with 
    their offspring int he new generation.

    The first generation is of size lambda.
    The second generation is of size mew+lambda
    '''
    def __init__(self, mew, _lambda, mutator):
        '''
        Constructor

        Parameters:
        --------
        mew -       int, controls how selectiveness the algorithm.  
                    Low values of mew relative to _lambsa mean that only the best 
                    breed in each generation and the algorithm becomes 
                    more exploitative.

        _lambda -   int, controls the size of each generation.

        mutator -   AbstractTourMutator, encapsulates the logic of mutation for a tour
        '''
        self._mew = mew
        self._lambda = _lambda
        self._mutator = mutator

    
    def evolve(self, population, costs):
        '''
        Only mew fittest individual survice.
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- list, of numpy arrays representing a generation of tours

        Returns:
        --------
        list of numpy.arrays - a new generation of tours.
        '''

        fittest_indexes = np.argpartition(costs, self._mew)[:self._mew]
        fittest = np.array(population)[fittest_indexes]

        population = fittest.tolist()  #this is the difference from (mew, lambda)

        for parent in fittest:
            for child_n in range(int(self._lambda/self._mew)):
                child = self._mutator.mutate(parent.copy())
                population.append(child)

        return population


class EvolutionaryAlgorithm(object):
    '''
    Encapsulates a simple Evolutionary algorithm
    with mutation at each generation.
    '''
    def __init__(self, tour, matrix, _lambda, 
                 strategy, max_iter=1000):
        '''
        Parameters:
        ---------
        n_cities    - int, the number of cities to visit
        matrix      - np.array, cost matrix travelling from city i to city j
        _lambda     - int, initial population size
        strategy    - AbstractEvolutionStrategy, evolution stratgy
        max_iter    - int, maximum number of iterations (default=1000)
        '''
        self._tour = tour
        self._max_iter = max_iter
        self._matrix = matrix
        self._strategy = strategy
        self._lambda = _lambda
        self._best = None
        self._best_cost = np.inf

    def _get_best(self):
        return self._best
    
    def _get_best_cost(self):
        return self._best_cost

    def solve(self):

        population = initiation_population(self._lambda, self._tour)
        costs = None
    
        for generation in range(self._max_iter):
            costs = self._fitness(population)
            
            min_index = np.argmin(costs)

            if self._best is None or (costs[min_index] < self._best_cost):
                self._best = population[min_index]
                self._best_cost = costs[min_index]
            
            population = self._strategy.evolve(population, costs)

    
                

    def _fitness(self, population):
        costs = np.full(len(population), -1.0, dtype=float)
        for i in range(len(population)):
            
            costs[i] = tour_cost(population[i], self._matrix)

        return costs
            
    best_solution = property(_get_best)
    best_cost = property(_get_best_cost)






            












    


