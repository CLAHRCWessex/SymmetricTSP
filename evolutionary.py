
import numpy as np
from abc import ABC, abstractmethod

from objective import tour_cost, symmetric_tour_list
from tsp_utility import append_base, trim_base
from init_solutions import random_tour

class AbstractMutator(ABC):
    @abstractmethod
    def mutate(self, individual):
        pass

class AbstractEvolutionStrategy(ABC):
    @abstractmethod
    def evolve(self, population, costs):
        pass

class AbstractSelector(ABC):
    @abstractmethod
    def select(self, population, costs):
        pass

class TruncationSelector(AbstractSelector):
    '''
    Simple truncation selection of the mew fittest 
    individuals in the population
    '''
    def __init__(self, mew):
        self._mew = mew

    def select(self, population, fitness):
        fittest_indexes = np.argpartition(fitness, fitness.size - self._mew)[-self._mew:]
        return population[fittest_indexes]

class TournamentSelector(AbstractSelector):
    '''
    Encapsulates a popular GA selection algorithm called
    Tournament Selection.  An individual is selected at random
    (with replacement) as the best from the population and competes against
    a randomly selected (with replacement) challenger.  If the individual is
    victorious they compete in the next round.  If the challenger is successful
    they become the best and are carried forward to the next round. This is repeated
    for t rounds.  Higher values of t are more selective.  
    '''
    def __init__(self, tournament_size=2):
        '''
        Constructor

        Parameters:
        ---------
        tournament_size, int, must be >=1, (default=2)
        '''
        if tournament_size < 1:
            raise ValueError('tournamant size must int be >= 1')
        
        self._tournament_size = tournament_size
        
    def select(self, population, fitness):
        '''
        Select individual from population for breeding using
        a tournament approach.  t tournaments are conducted.
        '''
        best_index = np.random.randint(0, population.shape[0])
        best, best_fitness = population[best_index], fitness[best_index]
        

        for i in range(2, self._tournament_size + 1):
            challenger_index = np.random.randint(0, population.shape[0])
            challenger = population[challenger_index]
            
            if fitness[challenger_index] > best_fitness:
                best = challenger
                best_fitness = fitness[challenger_index]

        return best
            


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
    np.array. matrix size = (population_size, len(tour)). Contains
              the initial generation of tours
    '''

    population = {}
    #for i in range(population_size):
    i = 0
    while i < population_size:
        #some code is legacy and uses python
        #lists instead of numpy arrays... to fix!
        new_tour = random_tour(tour)

        #return data as
        population_arr = np.full((population_size, len(tour)), -1, dtype=int)

        if str(new_tour) not in population:
            population[str(new_tour)] = np.array(new_tour)
            i = i + 1

    population_arr[:,] = list(population.values())

    return population_arr
    

class TwoCityMutator(AbstractMutator):
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


class TwoOptMutator(AbstractMutator):
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
        Truncation selection - only mew fittest individuals survive.  
        
        Each of these breed lambda/mew children who are mutations
        of the parent.

        Parameters:
        --------
        population -- numpy array, matrix representing a generation of tours
                      size (lambda, len(tour))

        costs -- numpy.array, vector, size lambda, representing the cost of the 
                 tours in population

        Returns:
        --------
        numpy.array - matrix of new generation of tours, size (lambda, len(tour))
        '''

        fittest_indexes = np.argpartition(costs, costs.size - self._mew)[-self._mew:]
        fittest = population[fittest_indexes]

        population = np.full((self._lambda, fittest[0].shape[0]),
                             fill_value=-1, dtype=int)

        index = 0
        for parent in fittest:
            for child_n in range(int(self._lambda/self._mew)):
                child = self._mutator.mutate(parent.copy())
                population[index] = child
                index += 1

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
        population -- numpy array, matrix representing a generation of tours
                      size (lambda+mew, len(tour))

        costs -- numpy.array, vector, size lambda, representing the cost of the 
                 tours in population

        Returns:
        --------
        numpy.arrays - matric a new generation of tours, size (lambda+mew, len(tour))
        '''

        fittest_indexes = np.argpartition(costs, costs.size - self._mew)[-self._mew:]
        fittest = population[fittest_indexes]
        
        #this is the difference from (mew, lambda)
        population = np.full((self._lambda+self._mew, fittest[0].shape[0]),
                             0, dtype=int)

        population[:len(fittest),] = fittest
    
        index = self._mew
        for parent in range(len(fittest)):
            for child_n in range(int(self._lambda/self._mew)):
                child = self._mutator.mutate(fittest[parent].copy())
                population[index] = child
                index += 1

        return population





class EvolutionaryAlgorithm(object):
    '''
    Encapsulates a simple Evolutionary algorithm
    with mutation at each generation.
    '''
    def __init__(self, tour, matrix, _lambda, 
                 strategy, maximisation=True, generations=1000):
        '''
        Parameters:
        ---------
        tour        - np.array, cities to visit
        matrix      - np.array, cost matrix travelling from city i to city j
        _lambda     - int, initial population size
        strategy    - AbstractEvolutionStrategy, evolution stratgy
        maximisation- bool, True if the objective is a maximisation and 
                      False if objective is minimisation (default=True)
        generations - int, maximum number of generations  (default=1000)
        '''
        self._tour = tour
        self._max_generations = generations
        self._matrix = matrix
        self._strategy = strategy
        self._lambda = _lambda
        self._best = None
        self._best_cost = np.inf
        
        if maximisation:
            self._negate = 1
        else:
            self._negate = -1

    def _get_best(self):
        return self._best
    
    def _get_best_cost(self):
        return self._best_cost * self._negate

    def solve(self):

        population = initiation_population(self._lambda, self._tour)
        costs = None
    
        for generation in range(self._max_generations):
            costs = self._fitness(population)
            
            max_index = np.argmax(costs)

            if self._best is None or (costs[max_index] > self._best_cost):
                self._best = population[max_index]
                self._best_cost = costs[max_index]
            
            population = self._strategy.evolve(population, costs)

    
                
    def _fitness(self, population):
        costs = np.full(len(population), -1.0, dtype=float)
        for i in range(len(population)):
            
            costs[i] = tour_cost(population[i], self._matrix)

        return costs * self._negate
            
    best_solution = property(_get_best)
    best_cost = property(_get_best_cost)






            












    


