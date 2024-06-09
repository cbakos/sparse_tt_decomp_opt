import math
from typing import List

import numpy as np
from primePy import primes
from itertools import chain, combinations


def prime_factors(n: int) -> List[int]:
    """
    Returns the prime factors of n (including multiplicities).
    :param n: non-negative integer to be factored
    :return: prime factors of n
    """
    factors = primes.factors(n)
    return factors


def possible_tile_sizes_from_factors(factors: List[int]) -> np.array:
    """
    Returns the possible tile sizes from factors.
    :param factors: prime factors of n
    :return: unique tile sizes constructed as products of factors as a np.aray of size (k,)
    """

    factor_combinations = chain.from_iterable(combinations(factors, r) for r in range(1, len(factors) + 1))
    products = np.unique([math.prod(t) for t in factor_combinations])
    return products
