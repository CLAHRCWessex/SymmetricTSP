'''
Unit tests for evolutionary algorithms module 'evolutionary.py'

Work in progress!
'''

import numpy as np
import pytest

from evolutionary import PartiallyMappedCrossover


def test_pmx_child_a():
    
    test_seed = 101

    np.random.seed(seed=test_seed)
    #sets up swap between index 1 and 3 for child a
    #sets up swap between index 3 and 5 for child b

    parent_a = np.array([1, 2, 3, 4, 5, 12])
    parent_b = np.array([2, 5, 12, 3, 4, 1])

    #expected answer c_a:
    #0: c_a = [1, 2, 3, 4, 5, 12]
    #1: c_a = [1, 5, 3, 4, 2, 12]
    #2: c_a = [1, 5, 12, 4, 2, 3]
    #3: c_a = [1, 5, 12, 3, 2, 4]

    #expected answer c_b:
    #0: c_a = [2, 5, 12, 3, 4, 1]
    #1: c_a = [2, 5, 12, 4, 3, 1]
    #2: c_a = [2, 3, 12, 4, 5, 1]
    #3: c_a = [2, 3, 1, 4, 5, 12]

    expected_c_a = np.array([1, 5, 12, 3, 2, 4])
    expected_c_b = np.array([2, 3, 1, 4, 5, 12])

    print(np.sort(np.random.randint(0, len(parent_a), size = 2)))

    print('before: {0} {1}'.format(parent_a, parent_b))

    np.random.seed(seed=test_seed)
    x_operator = PartiallyMappedCrossover()
    c_a, c_b = x_operator.crossover(parent_a, parent_b)

    print('children: {0} {1}'.format(c_a, c_b))
    print('after: {0} {1}'.format(parent_a, parent_b))

    assert np.array_equal(expected_c_a, c_a)

def test_pmx_child_b():
    
    test_seed = 101

    np.random.seed(seed=test_seed)
    #sets up swap between index 1 and 3 for child a
    #sets up swap between index 3 and 5 for child b

    parent_a = np.array([1, 2, 3, 4, 5, 12])
    parent_b = np.array([2, 5, 12, 3, 4, 1])

    #expected answer c_a:
    #0: c_a = [1, 2, 3, 4, 5, 12]
    #1: c_a = [1, 5, 3, 4, 2, 12]
    #2: c_a = [1, 5, 12, 4, 2, 3]
    #3: c_a = [1, 5, 12, 3, 2, 4]

    #expected answer c_b:
    #0: c_a = [2, 5, 12, 3, 4, 1]
    #1: c_a = [2, 5, 12, 4, 3, 1]
    #2: c_a = [2, 3, 12, 4, 5, 1]
    #3: c_a = [2, 3, 1, 4, 5, 12]

    expected_c_a = np.array([1, 5, 12, 3, 2, 4])
    expected_c_b = np.array([2, 3, 1, 4, 5, 12])

    print(np.sort(np.random.randint(0, len(parent_a), size = 2)))

    print('before: {0} {1}'.format(parent_a, parent_b))

    np.random.seed(seed=test_seed)
    x_operator = PartiallyMappedCrossover()
    c_a, c_b = x_operator.crossover(parent_a, parent_b)

    print('children: {0} {1}'.format(c_a, c_b))
    print('after: {0} {1}'.format(parent_a, parent_b))

    assert np.array_equal(expected_c_b, c_b)


if __name__ == '__main__':
    test_pmx_child_a()
    test_pmx_child_b()
    