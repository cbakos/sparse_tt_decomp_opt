from typing import List

import numpy as np
from primePy import primes


def prime_factors(n: int) -> List[int]:
    """
    Returns the prime factors of n (including multiplicities).
    :param n: non-negative integer to be factored
    :return: prime factors of n
    """
    factors = primes.factors(n)
    return factors
